import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict


@dataclass
class CountryAvgRevenueEncoder:
    """
    Simple target-mean encoder for country using training labels only.
    Stores a mapping + global mean for unseen countries.
    """
    mapping_: Dict[str, float] = None
    global_mean_: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, country_col: str = "country"):
        tmp = X_train[[country_col]].copy()
        tmp["target"] = y_train.values
        means = tmp.groupby(country_col)["target"].mean()
        self.mapping_ = means.to_dict()
        self.global_mean_ = float(y_train.mean())
        return self

    def transform(self, X: pd.DataFrame, country_col: str = "country", new_col: str = "country_avg_revenue") -> pd.DataFrame:
        X = X.copy()
        X[new_col] = X[country_col].map(self.mapping_).fillna(self.global_mean_)
        return X
