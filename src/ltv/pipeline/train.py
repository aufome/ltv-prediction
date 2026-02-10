import numpy as np

from ltv.logging import get_logger
from ltv.config import (
    PREPROCESSOR_PARAMS,
    FEATURES,
    DATE_COLUMN,
    TARGET,
    TARGET_RAW,
    MODEL_DIR,
    P_MIN_THRESHOLD
)
from ltv.features.preprocess import Preprocessor, PreprocessorConfig
from ltv.features.encoders import CountryAvgRevenueEncoder
from ltv.data.split import temporal_split
from ltv.models.xgb import train_classifier, train_regressor
from ltv.models.calibration import calibrate_classifier_isotonic
from ltv.models.hurdle import HurdleModel
from ltv.models.save_load import SaveLoad

logger = get_logger(__name__)


class TrainPipeline:
    def __init__(
        self,
        preprocessor_params: dict = PREPROCESSOR_PARAMS,
        bundle_name: str = "ltv_bundle.pkl",
        p_min_threshold: float | None = P_MIN_THRESHOLD,
        clf_params: dict | None = None,
        reg_params: dict | None = None,
    ):
        self.preprocessor_params = preprocessor_params
        self.bundle_name = bundle_name
        self.p_min_threshold = p_min_threshold

        # sensible defaults; you can tune later
        self.clf_params = clf_params or dict(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        self.reg_params = reg_params or dict(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    def run(self):
        logger.info("=== Training pipeline started ===")

        # Preprocess
        logger.info("Running Preprocessor (training mode)")
        config = PreprocessorConfig(**self.preprocessor_params)
        df = Preprocessor(config).run()
        logger.info(f"Preprocessed user table shape: {df.shape}")

        # Temporal split (based on final regression target column)
        # We split using TARGET so the split indices match regression evaluation later.
        base_features = [c for c in FEATURES if c != "country_avg_revenue"]
        split_df = df[[DATE_COLUMN, "country", TARGET, TARGET_RAW] + base_features].copy()


        X_train, X_val, X_test, y_train_t, y_val_t, y_test_t = temporal_split(
            df=split_df,
            date_column=DATE_COLUMN,
            target_column=TARGET
        )

        # Build payer labels from RAW revenue (always in dollars)
        y_train_payer = (X_train[TARGET_RAW] > 0).astype(int)
        y_val_payer = (X_val[TARGET_RAW] > 0).astype(int)

        # Fit country encoder on TRAIN ONLY
        encoder = CountryAvgRevenueEncoder().fit(
            X_train[["country"]],
            y_train_t if "_log" not in TARGET else np.expm1(y_train_t),
            country_col="country",
        )

        # Add country_avg_revenue to all splits
        X_train["country_avg_revenue"] = X_train["country"].map(encoder.mapping_).fillna(encoder.global_mean_)
        X_val["country_avg_revenue"] = X_val["country"].map(encoder.mapping_).fillna(encoder.global_mean_)
        X_test["country_avg_revenue"] = X_test["country"].map(encoder.mapping_).fillna(encoder.global_mean_)

        # Drop non-feature cols to make matrices
        drop_cols = [DATE_COLUMN, TARGET, TARGET_RAW, "country"]
        X_train_m = X_train.drop(columns=drop_cols, errors="ignore")[FEATURES].copy()
        X_val_m = X_val.drop(columns=drop_cols, errors="ignore")[FEATURES].copy()
        X_test_m = X_test.drop(columns=drop_cols, errors="ignore")[FEATURES].copy()

        # Train classifier (raw)
        logger.info("Training payer classifier (XGBClassifier)")
        clf_raw = train_classifier(X_train_m, y_train_payer, X_val_m, y_val_payer, params=self.clf_params)

        # Calibrate classifier (isotonic)
        logger.info("Calibrating classifier (isotonic on validation)")
        clf_cal = calibrate_classifier_isotonic(clf_raw, X_val_m, y_val_payer)

        # Train regressor on payers only
        # Regression target column is TARGET
        payer_mask_train = (X_train[TARGET_RAW] > 0).values
        payer_mask_val = (X_val[TARGET_RAW] > 0).values

        X_train_pos = X_train_m.loc[payer_mask_train]
        X_val_pos = X_val_m.loc[payer_mask_val]

        y_train_reg = y_train_t.loc[payer_mask_train]
        y_val_reg = y_val_t.loc[payer_mask_val]

        logger.info(f"Training regressor on payers only: train={len(X_train_pos)}, val={len(X_val_pos)}")
        reg = train_regressor(X_train_pos, y_train_reg, X_val_pos, y_val_reg, params=self.reg_params)

        # Wrap into HurdleModel
        hurdle = HurdleModel(
            clf_raw=clf_raw,
            clf_cal=clf_cal,
            reg=reg,
            p_min_threshold=self.p_min_threshold,
            reg_in_log_space=("_log" in TARGET),
        )
        
        from ltv.evaluation.metrics import (
            regression_metrics,
            payer_metrics,
            classifier_metrics,
            revenue_capture_metrics,
        )

        
        # Evaluate on test
        logger.info("Evaluating on test set")

        # payer labels in test (based on raw dollars)
        y_test_payer = (X_test[TARGET_RAW] > 0).astype(int).values

        # true revenue in dollars for evaluation
        if "_log" in TARGET:
            y_test_dollars = np.expm1(y_test_t.values)
        else:
            y_test_dollars = y_test_t.values

        # predicted probability + predicted LTV
        p_test = hurdle.predict_payer_proba(X_test_m)
        y_pred_dollars = hurdle.predict_ltv(X_test_m)

        metrics_out = {}
        metrics_out.update(regression_metrics(y_test_dollars, y_pred_dollars))
        metrics_out.update(payer_metrics(y_test_dollars, y_pred_dollars))
        metrics_out.update(classifier_metrics(y_test_payer, p_test))
        metrics_out.update(revenue_capture_metrics(y_test_dollars, y_pred_dollars))

        logger.info(
            "Test metrics: " + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics_out.items()
                if k in ["mae", "rmse", "nmae", "revenue_ratio", "payer_rate", "clf_auc", "clf_prauc", "capture_top_10pct"]
            )
        )



        # Save bundle
        bundle = {
            "hurdle_model": hurdle,
            "clf_raw": clf_raw,            # optional, but convenient for SHAP
            "clf_cal": clf_cal,
            "reg": reg,
            "country_encoder": encoder,
            "features": FEATURES,
            "target": TARGET,
            "date_column": DATE_COLUMN,
            "p_min_threshold": self.p_min_threshold,
            "clf_params": self.clf_params,
            "reg_params": self.reg_params,
            "test_metrics": metrics_out
        }

        path = SaveLoad(model_dir=str(MODEL_DIR)).save_bundle(bundle, filename=self.bundle_name)
        logger.info(f"Saved model bundle to: {path}")

        logger.info("=== Training pipeline finished ===")
        return bundle
