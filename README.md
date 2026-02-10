# LTV Prediction with a Zero-Inflated Hurdle Model (XGBoost)

## Project Overview

This project predicts **first-year customer Lifetime Value (LTV)** using only the first 15 days of user behavior.

A key challenge is that the dataset is **zero-inflated**:

- Most users generate **no revenue**
- A small fraction of payers generate the majority of revenue

To handle this, we build a **two-stage hurdle model**:

1. **Classifier** → predicts whether a user becomes a payer  
2. **Regressor** → predicts revenue *conditional on paying*  

Final expected LTV:

$$
\hat{LTV} = \hat{p}(\text{payer}) \cdot \hat{r}(\text{revenue} \mid \text{payer})
$$

---

## Features

All features are observed within the first 15 days:

- Engagement (`active_days_*`, longest streaks, total active days)
- Funnel events (`subscribe_freq`, `free_trial_freq`, `refund_freq`, etc.)
- Early monetization (`two_week_revenue`)
- Geographic baseline (`country_avg_revenue` (leakage-free))

---

## Train/Test Strategy

We apply a **temporal split** based on `first_event_date`:

- Train on earliest users  
- Validation on intermediate period  
- Test on most recent users  

This avoids future leakage and closely mimics real-world deployment conditions.

---

## Modeling Approach

### Stage 1: Payer Classification

- Model: `XGBClassifier`
- Handles strong class imbalance via `scale_pos_weight`
- Metrics: AUC ≈ 0.95, PR-AUC ≈ 0.88
- Probabilities calibrated using **isotonic regression**

### Stage 2: Conditional Revenue Regression

- Model: `XGBRegressor`
- Trained only on paying users
- Target: 99th-percentile capped revenue

---

## Thresholding and Zero-Inflation Control

The hurdle model naturally produces many small positive predictions due to the soft combination:

$$
\hat{LTV} = P(\text{payer}) \cdot E(\text{revenue} \mid \text{payer})
$$

We experimented with applying a minimum probability threshold:

$$
\hat{LTV} = 0 \text{ if } P(\text{payer}) < p_{min}
$$

A focused sweep around $p_{min} \in [0.005, 0.01]$ revealed a clear trade-off:

- Lower thresholds preserve calibration but over-predict on zero users

- Higher thresholds reduce false positives but under-estimate revenue

Because the unthresholded model already achieves near-perfect revenue calibration, the final model uses the soft hurdle without hard thresholding.

---

## Target Engineering Results

| Target | MAE | NMAE | RMSE | Revenue Ratio | Predicted Revenue on True Zeros |
|-------|-----|------|------|-----|---------------------------|
| Raw revenue | 2.46 | 0.91 | 14.06 | 1.03 | 123k |
| **Capped 99%** | **1.27** | **0.71** | **5.78** | **1.00** | **93k** |
| Capped 99% + log | 1.16 | 0.65 | 5.95 | 0.80 | 63k |

Final target choice: `first_year_revenue_capped_99`

While the log-transformed target reduces error metrics slightly, it introduces systematic revenue underestimation. The capped (non-log) target achieves better revenue calibration while maintaining strong accuracy, making it more suitable for business use.

---

## SHAP Interpretability

SHAP values were computed separately for both components of the hurdle model.

### Classifier Drivers

Top features:

- `two_week_revenue`
- `free_trial_freq`
- `subscribe_freq`
- `country_avg_revenue`
- `total_active_days_14`

SHAP summary plot:

<img src="reports/shap_clf_summary.png" width="700">


---

### Regressor Drivers

Top features:

- `auto_renew_off_freq`
- `two_week_revenue`
- `country_avg_revenue`
- `renewal_freq`
- sustained engagement features

SHAP summary plot:

<img src="reports/shap_reg_summary.png" width="700">



---

## Revenue Capture Performance

Beyond pointwise error metrics, we evaluate the model’s ranking power, which is often more relevant for business decisions.

| Segment | Revenue Captured |
|--------|------------------|
| Top 1% | 28.8% |
| Top 5% | 82.5% |
| Top 10% | 88.9% |
| Top 20% | 91.9% |


Cumulative revenue capture curve:

<img src="reports/cumulative_revenue_curve.png" width="400">


The model successfully concentrates revenue among a small fraction of high-value users, making it well-suited for targeting, retention, and marketing use cases.

---

## Conclusion

This project delivers a production-style LTV system with:

- explicit hurdle modeling for zero inflation

- strong payer classification with calibrated probabilities

- capped revenue regression for stability and calibration

- SHAP-based interpretability

- excellent revenue-ranking performance

The final model balances accuracy, calibration, interpretability, and business usefulness.

---

## Repository Structure

```
ltv-prediction/
│
├── src/ltv/ # Core package
│ ├── pipeline/ # Train/predict orchestration
│ ├── models/ # Classifier, regressor, calibration
│ ├── features/ # Preprocessing + feature engineering
│ ├── evaluation/ # Metrics + revenue capture tools
│ ├── config.py # Paths + constants
│ └── logging.py # Logging utilities
│
├── scripts/
│ ├── train.py # Run training pipeline
│ └── predict.py # Run prediction pipeline (--clean supported)
│
├── data/
│ ├── raw/ # Raw event-level parquet files
│ ├── input/ # Prediction-time input files
│ └── output/ # Saved predictions
│
├── models/ # Saved hurdle model bundle
├── reports/ # SHAP plots + evaluation figures
├── notebooks/ # Development + experimentation notebooks
└── README.md
```

---

## Usage

### 1. Install

From the project root:

```
pip install -e .
````

### 2. Training

Run the full training pipeline:

```
python scripts/train.py
```

Training inputs:
- Raw parquet file: `data/raw/ltv_prediction_raw.parquet`

Training outputs:
- Saved model bundle: ```models/ltv_bundle.pkl```
- Logs: ```logs/pipeline.log```

### 3. Prediction

Run prediction using a trained model:

```
python scripts/predict.py
```

If prediction input requires preprocessing:

```
python scripts/predict.py --clean
```

Prediction inputs:
- Processed parquet file: `data/input/ltv_prediction_input.parquet`

Or
- Raw parquet file: `data/input/ltv_prediction_input_raw.parquet`

Prediction outputs:

- User-level predictions: ```data/outputs/ltvprediction_users.parquet```
- Cohort-level predictions: ```data/outputs/ltvprediction_cohort.parquet```

### 4. Data

The `data/` folder comes with sample Parquet files. These are just small subsets so you can see the structure of the data, they’re not really meant for training.

That said, if you still want to use them for experiments, you’ll need to:

- either remove the `sample` part from the filenames,
- or adjust the paths in your config and notebooks accordingly.

The **original data folder** (used during model training and notebook experiments) has been compressed into a single archive and can be downloaded [here](https://drive.google.com/file/d/1qIs4lAYSMoA33X0Qy2hD5WyDBd7aftji/view?usp=sharing).


## Notes

- Make sure parquet formatted raw data is placed in ```data/raw/``` before training.

- Make sure parquet formatted input data is placed in ```data/input/``` before predicting.

- You can configure file paths and parameters in ```src/ltv/config.py```

- There are two Jupyter Notebook files available for you to explore how this model is developed. Please check ```notebooks``` folder.

