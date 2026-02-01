from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate_classifier_isotonic(clf_prefit, X_val, y_val):
    """
    Calibrate a 'prefit' classifier on a held-out validation set
    using isotonic regression.
    """
    cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(clf_prefit),
        method="isotonic",
        cv=2,
        n_jobs=-1
    )
    cal.fit(X_val, y_val)
    return cal



