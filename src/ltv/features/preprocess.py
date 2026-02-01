import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from ..config import RAW_FILE_NAME, REVENUE_CLIP_QUANTILE
from ..logging import get_logger


logger = get_logger(__name__)

@dataclass
class PreprocessorConfig:
    file_dir: str
    input_file: bool = False
    file_name: str = RAW_FILE_NAME
    revenue_clip_quantile: float = REVENUE_CLIP_QUANTILE
    random_state: int = 314
    day_intervals: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2),
        (3, 5),
        (6, 8),
        (9, 11),
        (12, 14),
    ])

class Preprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        
        self._df_raw: Optional[pd.DataFrame] = None
        self._df_15d: Optional[pd.DataFrame] = None
        self._users_model_input: Optional[pd.DataFrame] = None
        self._users_event_encoded: Optional[pd.DataFrame] = None
        self._users_event_freq: Optional[pd.DataFrame] = None
        self._users_revenue: Optional[pd.DataFrame] = None
        self._users_engagement: Optional[pd.DataFrame] = None

    @property
    def users_model_input(self) -> pd.DataFrame:
        return self._users_model_input.copy()

    def _load_data(self):
        try:
            path = os.path.join(self.config.file_dir, self.config.file_name)
            self._df_raw = pd.read_parquet(path)
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"{self.config.file_dir} is not found.")
        if self._df_raw.empty:
            logger.error("Loaded data frame is empty.")
            raise ValueError("Loaded data frame is empty.")
        return self
    

    def _clean_data(self):
        df = self._df_raw.copy()
        df = df.drop_duplicates()
        df = df.dropna(subset=["country"])
        df = df.assign(
            revenue=df["revenue"].fillna(0),
            event_date=pd.to_datetime(df["event_date"]),
            first_event_date=pd.to_datetime(df["first_event_date"]),
        )
        
        # Only works when training the raw file
        if not self.config.input_file:
            # Filter unrealistic first_year_revenue
            df = df[(df.first_year_revenue == 0) | (df.first_year_revenue > 0.001)]
        
        self._df_raw = df
        return self

    def _prepare_15d_data(self):
        max_date = self._df_raw.event_date.max()
        cutoff_date = max_date - pd.Timedelta(days=15)
        df_15d = self._df_raw[self._df_raw.first_event_date <= cutoff_date].copy()
        df_15d["days_since_signup"] = (
            df_15d.event_date - df_15d.first_event_date
        ).dt.days
        df_15d = df_15d[df_15d["days_since_signup"].between(0, 14)]
        self._df_15d = df_15d
        return self

    def _encode_user_events(self):
        df = self._df_15d

        # Binary event encoding
        users_event_encoded = (
            df.groupby(["user_id", "event_name"])
            .size()
            .unstack(fill_value=0)
            .gt(0)
            .astype(int)
            .reset_index()
        )

        # Interaction features
        interactions = {
            "subscribe_renewal": ("subscribe", "renewal"),
            "paywall_refund": ("paywall", "refund"),
        }
        for new_col, (col1, col2) in interactions.items():
            if (
                col1 in users_event_encoded.columns
                and col2 in users_event_encoded.columns
            ):
                users_event_encoded[new_col] = (
                    users_event_encoded[col1] & users_event_encoded[col2]
                ).astype(int)
            else:
                users_event_encoded[new_col] = 0
        
        # Only works when training the raw file
        if not self.config.input_file:
            # Merge first_year_revenue
            users_event_encoded = users_event_encoded.merge(
                df[["user_id", "first_year_revenue"]].drop_duplicates("user_id"),
                on="user_id",
                how="left",
            )

        self._users_event_encoded = users_event_encoded
        return self

    def _event_frequency_features(self):
        df = self._df_15d
        users_event_freq = (
            df.groupby(["user_id", "event_name"])
            .size()
            .unstack(fill_value=0)
            .astype(int)
        )
        users_event_freq.columns = [f"{c}_freq" for c in users_event_freq.columns]
        users_event_freq.reset_index(inplace=True)
        self._users_event_freq = users_event_freq
        return self
    
    def _two_day_revenue_feature(self):
        df = self._df_15d
        users_revenue = df.groupby("user_id", as_index=False).agg(
            two_week_revenue=("revenue", "sum")
        )
        
        self._users_revenue = users_revenue
        
        return self

    def _active_days_features(self):
        df = self._df_15d
        users_active_days = (
            df.groupby(["user_id", "first_event_date", "country", "days_since_signup"])[
                "event_name"
            ]
            .count()
            .unstack(fill_value=0)
            .gt(0)
            .astype(int)
        )
        users_active_days.drop(columns=[0], inplace=True, errors="ignore")

        # Day intervals
        features_set = {}
        for start, end in self.config.day_intervals:
            features_set[f"active_days_{start}_{end}"] = users_active_days.loc[
                :, start:end
            ].sum(axis=1)
        features_set["total_active_days_14"] = users_active_days.loc[:, 1:14].sum(
            axis=1
        )

        # Longest consecutive active days
        def longest_consecutive(row):
            longest_cons = cons = 0
            for val in row:
                if val == 1:
                    cons += 1
                    longest_cons = max(longest_cons, cons)
                else:
                    cons = 0
            return longest_cons

        features_set["max_consecutive_active_days_14"] = users_active_days.apply(
            longest_consecutive, axis=1
        )

        users_active_days.columns = ["day_" + str(c) for c in users_active_days.columns]
        users_active_days.reset_index(inplace=True)
        users_active_days.drop(
            columns=["first_event_date", "country"], inplace=True, errors="ignore"
        )

        users_engagement = pd.DataFrame(features_set).reset_index()
        users_engagement = users_engagement.merge(users_active_days, on="user_id")

        self._users_engagement = users_engagement
        return self

    def _merge_features(self):
        df = self._users_engagement.merge(self._users_event_encoded, on="user_id")
        df = df.merge(self._users_event_freq, on="user_id")
        df = df.merge(self._users_revenue, on="user_id")

        # Only works when training the raw file
        if not self.config.input_file:
            # Revenue features
            df["first_year_revenue_capped"] = np.clip(
                df["first_year_revenue"],
                a_min=0,
                a_max=df["first_year_revenue"].quantile(self.config.revenue_clip_quantile),
            )
            df["first_year_revenue_log"] = np.log1p(df["first_year_revenue"])
            df["first_year_revenue_capped_log"] = np.log1p(df["first_year_revenue_capped"])

        self._users_model_input = df
        return self

    def run(self):
        return (
            self._load_data()
            ._clean_data()
            ._prepare_15d_data()
            ._encode_user_events()
            ._event_frequency_features()
            ._two_day_revenue_feature()
            ._active_days_features()
            ._merge_features()
            ._users_model_input
        )