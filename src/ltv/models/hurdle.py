import numpy as np
import pandas as pd
from dataclasses import dataclass
from ltv.config import TARGET, P_MIN_THRESHOLD


@dataclass
class HurdleModel:
    clf_raw: object
    clf_cal: object
    reg: object
    p_min_threshold: float = P_MIN_THRESHOLD
    reg_in_log_space: bool = "LOG" in TARGET   # reg predicts log1p(revenue) for payers

    def predict_payer_proba(self, X: pd.DataFrame) -> np.ndarray:
        # assumes clf_cal has predict_proba
        return self.clf_cal.predict_proba(X)[:, 1]

    def predict_conditional_revenue(self, X: pd.DataFrame) -> np.ndarray:
        # reg predicts log-space or dollar-space depending on flag
        y_hat = self.reg.predict(X)
        if self.reg_in_log_space:
            y_hat = np.expm1(y_hat)
        return np.maximum(y_hat, 0.0)

    def predict_ltv(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_payer_proba(X)
        r = self.predict_conditional_revenue(X)
        ltv = p * r
        if self.p_min_threshold is not None:
            ltv = np.where(p < self.p_min_threshold, 0.0, ltv)
        return ltv
