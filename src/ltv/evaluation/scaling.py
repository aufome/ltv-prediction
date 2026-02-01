import numpy as np

def revenue_scaling_dollars(y_true_dollars, y_pred_dollars):
    y_true_dollars = np.asarray(y_true_dollars)
    y_pred_dollars = np.asarray(y_pred_dollars)

    y_pred_dollars = np.maximum(y_pred_dollars, 0.0)
    scale = float(y_true_dollars.sum() / (y_pred_dollars.sum() + 1e-9))
    return y_pred_dollars * scale, scale
