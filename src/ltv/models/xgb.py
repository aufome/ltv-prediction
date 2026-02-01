from xgboost import XGBClassifier, XGBRegressor
import pandas as pd


def train_classifier(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    params: dict,
):
    
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    
    model = XGBClassifier(
        **params,
        random_state=314,
        n_jobs=-1,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )
    return model


def train_regressor(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    params: dict,
):
    model = XGBRegressor(
        **params,
        random_state=314,
        n_jobs=-1,
        eval_metric="rmse",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )
    return model
