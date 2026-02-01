import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mean_true = float(y_true.mean() + 1e-9)
    nmae = float(mae / mean_true)
    nrmse = float(rmse / mean_true)

    true_total = float(y_true.sum())
    pred_total = float(y_pred.sum())
    revenue_diff = float(pred_total - true_total)
    revenue_ratio = float(pred_total / (true_total + 1e-9))

    return {
        "mae": float(mae),
        "rmse": rmse,
        "nmae": nmae,
        "nrmse": nrmse,
        "true_total_revenue": true_total,
        "pred_total_revenue": pred_total,
        "revenue_diff": revenue_diff,
        "revenue_ratio": revenue_ratio,
    }

def payer_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    payer_mask = y_true > 0
    payer_rate = float(payer_mask.mean())

    out = {"payer_rate": payer_rate}

    if payer_mask.any():
        out["mae_payers"] = float(mean_absolute_error(y_true[payer_mask], y_pred[payer_mask]))
        out["rmse_payers"] = float(np.sqrt(mean_squared_error(y_true[payer_mask], y_pred[payer_mask])))
    else:
        out["mae_payers"] = np.nan
        out["rmse_payers"] = np.nan

    # How much predicted revenue goes to true-zero users
    zero_mask = ~payer_mask
    out["total_pred_on_true_zeros"] = float(y_pred[zero_mask].sum())
    out["mean_pred_on_true_zeros"] = float(y_pred[zero_mask].mean()) if zero_mask.any() else 0.0

    return out

def classifier_metrics(y_true_payer: np.ndarray, p_pred: np.ndarray) -> dict:
    y_true_payer = np.asarray(y_true_payer).astype(int)
    p_pred = np.asarray(p_pred)

    # Some edge cases: if all y are one class, AUC isn't defined
    out = {}
    if len(np.unique(y_true_payer)) == 2:
        out["clf_auc"] = float(roc_auc_score(y_true_payer, p_pred))
        out["clf_prauc"] = float(average_precision_score(y_true_payer, p_pred))
    else:
        out["clf_auc"] = np.nan
        out["clf_prauc"] = np.nan
    return out

def revenue_capture_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float) -> float:
    df = pd.DataFrame({"true": y_true, "pred": y_pred}).sort_values("pred", ascending=False)
    n_top = max(1, int(len(df) * k))
    return float(df.iloc[:n_top]["true"].sum() / (df["true"].sum() + 1e-9))

def revenue_capture_metrics(y_true: np.ndarray, y_pred: np.ndarray, ks=(0.01, 0.05, 0.10, 0.20)) -> dict:
    out = {}
    for k in ks:
        out[f"capture_top_{int(k*100)}pct"] = revenue_capture_at_k(y_true, y_pred, k)
        out[f"lift_top_{int(k*100)}pct"] = float(out[f"capture_top_{int(k*100)}pct"] / k)
    return out
