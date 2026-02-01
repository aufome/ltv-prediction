import os
import joblib
import pandas as pd
from typing import Any, Dict, Optional


class SaveLoad:
    """
    Model bundles and output artifacts.
    """
    def __init__(
        self,
        model_dir: str = "models",
        output_dir: str = os.path.join("data", "output"),
    ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    # Bundle (models + metadata)
    def save_bundle(self, bundle: Dict[str, Any], filename: str = "ltv_bundle.pkl") -> str:
        path = os.path.join(self.model_dir, filename)
        joblib.dump(bundle, path)
        return path

    def load_bundle(self, filename: str = "ltv_bundle.pkl") -> Dict[str, Any]:
        path = os.path.join(self.model_dir, filename)
        return joblib.load(path)

    # Outputs
    def save_user_predictions(
        self,
        users_df: pd.DataFrame,
        y_pred: pd.Series,
        filename: str = "ltv_predictions_users.parquet",
        extra_cols: Optional[list] = None,
    ) -> str:
        path = os.path.join(self.output_dir, filename)
        y_pred = y_pred.rename("predicted_ltv")

        cols = ["user_id", "country", "first_event_date"]
        if extra_cols:
            cols += extra_cols

        users_df[cols].join(y_pred).to_parquet(path=path, index=False)
        return path

    def save_cohort_predictions(
        self,
        users_df: pd.DataFrame,
        y_pred: pd.Series,
        filename: str = "ltv_predictions_cohort.parquet",
    ) -> str:
        path = os.path.join(self.output_dir, filename)
        y_pred = y_pred.rename("predicted_ltv")

        (
            users_df[["user_id", "country", "first_event_date"]]
            .join(y_pred)
            .groupby("first_event_date", as_index=False)
            .agg(avg_cohort_predicted_ltv=("predicted_ltv", "mean"))
            .to_parquet(path=path, index=False)
        )
        return path
