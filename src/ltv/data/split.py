import pandas as pd
from typing import Tuple


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Time-based split using the date_column. No shuffling.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    total_days = (max_date - min_date).days
    val_start = min_date + pd.Timedelta(days=int(total_days * train_size))
    test_start = val_start + pd.Timedelta(days=int(total_days * val_size))

    train_mask = df[date_column] < val_start
    val_mask = (df[date_column] >= val_start) & (df[date_column] < test_start)
    test_mask = df[date_column] >= test_start

    X_train = df.loc[train_mask].copy()
    X_val = df.loc[val_mask].copy()
    X_test = df.loc[test_mask].copy()

    y_train = X_train[target_column].copy()
    y_val = X_val[target_column].copy()
    y_test = X_test[target_column].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test
