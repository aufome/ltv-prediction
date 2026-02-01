import os
import pandas as pd

from ltv.logging import get_logger
from ltv.config import FEATURES, INPUT_DIR, INPUT_FILE_NAME, MODEL_DIR
from ltv.features.preprocess import Preprocessor, PreprocessorConfig
from ltv.models.save_load import SaveLoad

logger = get_logger(__name__)


class PredictPipeline:
    def __init__(
        self,
        clean: bool = False,
        input_dir: str = INPUT_DIR,
        input_file_name: str = INPUT_FILE_NAME,
        bundle_name: str = "ltv_bundle.pkl",
    ):
        self.clean = clean
        self.input_dir = input_dir
        self.input_file_name = input_file_name
        self.bundle_name = bundle_name

    def run(self):
        logger.info("=== Prediction pipeline started ===")

        # Load bundle
        saver = SaveLoad(model_dir=str(MODEL_DIR))
        bundle = saver.load_bundle(self.bundle_name)

        hurdle_model = bundle["hurdle_model"]
        country_encoder = bundle["country_encoder"]

        logger.info("Loaded model bundle successfully")

        # Load input data
        if self.clean:
            logger.info("Running Preprocessor on raw input file")
            
            config = PreprocessorConfig(
                input_file=True,
                file_dir=str(self.input_dir),
                file_name=self.input_file_name,
            )
            df_model_input = Preprocessor(config).run()

        else:
            path = os.path.join(self.input_dir, self.input_file_name)
            logger.info(f"Loading preprocessed input from {path}")
            df_model_input = pd.read_parquet(path)

        logger.info(f"Input data shape: {df_model_input.shape}")

        # Add country prior feature
        df_model_input["country_avg_revenue"] = (
            df_model_input["country"]
            .map(country_encoder.mapping_)
            .fillna(country_encoder.global_mean_)
        )


        # Feature matrix
        missing = [c for c in FEATURES if c not in df_model_input.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = df_model_input[FEATURES]


        # Predict LTV
        preds = hurdle_model.predict_ltv(X)

        logger.info("Prediction completed successfully")


        # Save outputs
        saver.save_user_predictions(df_model_input, pd.Series(preds))
        saver.save_cohort_predictions(df_model_input, pd.Series(preds))

        logger.info("Saved predictions successfully")
        logger.info("=== Prediction pipeline finished ===")

        return preds
