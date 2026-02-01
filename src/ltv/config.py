from __future__ import annotations

from pathlib import Path


# Paths
PACKAGE_DIR = Path(__file__).resolve().parent          # .../src/ltv
SRC_DIR = PACKAGE_DIR.parent                           # .../src
PROJECT_ROOT = SRC_DIR.parent                          # .../project_root

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

for p in [RAW_DIR, PROCESSED_DIR, INPUT_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# Filenames
RAW_FILE_NAME = "ltv_prediction_raw.parquet"
INPUT_FILE_NAME = "ltv_prediction_input.parquet"


# Columns
DATE_COLUMN = "first_event_date"
COUNTRY_COLUMN = "country"
USER_ID_COLUMN = "user_id"

# Raw target (dollars)
TARGET_RAW = "first_year_revenue"


# Preprocessor settings
REVENUE_CLIP_QUANTILE = 0.99 # cap at 99th percentile

# Engineered targets (produced in training mode by your Preprocessor)
TARGET_LOG = TARGET_RAW + "_log"
TARGET_CAPPED = TARGET_RAW + "_capped"
TARGET_CAPPED_LOG = TARGET_CAPPED + "_log"


# Final choice
TARGET = TARGET_CAPPED_LOG


PREPROCESSOR_PARAMS = {
    "file_dir": str(RAW_DIR),
    "file_name": RAW_FILE_NAME,
}


# Feature list (full set)
DAY_COLS = [f"day_{i}" for i in range(1, 15)]

ACTIVE_INTERVAL_COLS = [
    "active_days_1_2",
    "active_days_3_5",
    "active_days_6_8",
    "active_days_9_11",
    "active_days_12_14",
]

ENGAGEMENT_COLS = [
    "total_active_days_14",
    "max_consecutive_active_days_14",
]

# Binary event columns created by _encode_user_events()
BINARY_EVENT_COLS = [
    "auto_renew_off",
    "free_trial",
    "paywall",
    "refund",
    "renewal",
    "subscribe",
    "subscribe_renewal",
    "paywall_refund",
]

# Frequency columns created by _event_frequency_features()
FREQ_EVENT_COLS = [
    "auto_renew_off_freq",
    "free_trial_freq",
    "paywall_freq",
    "refund_freq",
    "renewal_freq",
    "subscribe_freq",
]

EARLY_MONETIZATION_COLS = ["two_week_revenue"]

# Added after split using training-only mapping
COUNTRY_PRIOR_COLS = ["country_avg_revenue"]

FEATURES = (
    DAY_COLS
    + ACTIVE_INTERVAL_COLS
    + ENGAGEMENT_COLS
    + BINARY_EVENT_COLS
    + FREQ_EVENT_COLS
    + EARLY_MONETIZATION_COLS
    + COUNTRY_PRIOR_COLS
)

# Columns never used as model inputs
DROP_COLUMNS = [
    USER_ID_COLUMN,
    DATE_COLUMN,
    COUNTRY_COLUMN,
    TARGET_RAW,
    TARGET_LOG,
    TARGET_CAPPED,
    TARGET_CAPPED_LOG
]

