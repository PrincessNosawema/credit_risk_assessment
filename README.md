# **Credit Risk Assessment** — ML Pipeline & Streamlit Dashboard

> Enterprise-ready credit risk assessment system with interactive model analysis and individual risk prediction.
> Implements a full ML pipeline (data cleaning → preprocessing → resampling → model training & evaluation) and a Streamlit web UI for exploration and single-case predictions.

**Key highlights**

* Cleaned and focused dataset: removes marketing columns that would leak information into a credit-risk target.
* Models trained: Logistic Regression, Random Forest, XGBoost.
* Class imbalance handled with SMOTE.
* Interactive Streamlit dashboard for model comparison, dataset analysis, and single-case risk prediction.
* Includes export of predicted probabilities (`df_predict_CORRECTED.csv`) from the standalone pipeline.

---

## Table of contents

1. [Features](#features)
2. [Project structure](#project-structure)
3. [Installation](#installation)
4. [Run the Streamlit app](#run-the-streamlit-app)
5. [Run the ML pipeline (CLI / script)](#run-the-ml-pipeline-cli--script)
6. [Dataset & cleaning rules](#dataset--cleaning-rules)
7. [ML approach & components](#ml-approach--components)
8. [Visualizations & dashboard pages](#visualizations--dashboard-pages)
9. [Outputs](#outputs)
10. [Troubleshooting & notes](#troubleshooting--notes)
11. [Contributing](#contributing)
12. [License](#license)

---

## Features

* End-to-end reproducible pipeline: load → clean → EDA → preprocess → resample → train → evaluate → predict → export.
* Multiple model comparison using cross-validated ROC AUC (Logistic Regression, Random Forest, XGBoost).
* Automated data cleaning tailored to the banking dataset (standardizes categories, removes marketing columns, fixes outliers).
* Handles categorical encoding via a mixed strategy (OneHot for low-cardinality, TargetEncoder for higher-cardinality).
* Imbalanced target handling using SMOTE.
* Streamlit dashboard with:

  * Overview and key metrics
  * Model performance comparison (CV vs test)
  * ROC curves, confusion matrix, probability distributions
  * Dataset analysis and cleaning report
  * Interactive single-customer risk predictor using top features

---

## Project structure

```
credit-risk-assessment/
├── app.py                          # Streamlit web application (dashboard)
├── credit_risk_assessment.py       # Standalone ML pipeline script (runs end-to-end + exports)
├── credit_risk_assessment.ipynb    # Notebook (EDA & experiments)
├── requirements.txt                # Python dependencies
├── dataset_banco.csv               # Banking dataset
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_processor.py           # Data cleaning, validation, dynamic input form helpers
│   ├── model_loader.py             # Model training, evaluation, preprocessor, prediction helpers
│   └── visualizations.py           # Plotting helpers used by Streamlit
├── .gitignore
├── LICENSE                         # MIT license
├── README.md                       # Project Documentation
└── CONTRIBUTING.md                 # Contributing guidelines
```

---

## Installation

1. Clone or copy the repository into a working folder.

2. Create and activate a Python virtual environment (recommended):

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes**

* `requirements.txt` contains the main packages used (pandas, numpy, scikit-learn, xgboost, imbalanced-learn, category-encoders, streamlit, plotly, joblib, ...).
* The code targets Python 3.8+ (see [requirements.txt](requirements.txt)).

---

## Run the Streamlit app

1. Run:

```bash
streamlit run app.py
```

2. The app runs locally and opens a dashboard with pages:

* Overview
* Model Performance
* Risk Predictor
* Dataset Analysis

**Streamlit caching**

* The app uses `st.cache_data` and `st.cache_resource` to cache dataset loading and trained model artifacts. Use the sidebar **Clear Cache & Retrain** button when you want to force retraining during development.

---

## Run the ML pipeline (CLI / script)

A standalone pipeline is available in `credit_risk_assessment.py`. Running it will execute the end-to-end process and export per-record predictions.

```bash
python credit_risk_assessment.py
```

What it does:

* Loads `dataset_banco.csv`
* Applies the cleaning rules (removes marketing columns, fixes categories, removes nulls/empty strings, fixes age outliers)
* Performs EDA (prints summary stats)
* Preprocesses (pipelines for numeric and categorical features)
* Balances training set using SMOTE
* Trains Logistic Regression, Random Forest, and XGBoost (3-fold CV)
* Evaluates on a held-out test set and prints ROC-AUC
* Generates predictions for the dataset and exports `df_predict_CORRECTED.csv`

---

## Dataset & cleaning rules

**Dataset:** `dataset_banco.csv` — banking dataset used for credit risk modeling. This repository does **not** include the dataset contents in the README; place the CSV in the project root for scripts/app to find it.

**Cleaning rules applied (automated)**

* Remove marketing-related columns that leak campaign information into a credit-risk target:
  `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, `y`

  * Note: `duration` is explicitly excluded because it causes data leakage (call length is a marketing success metric).
* Age correction: values > 100 are replaced with the rounded mean age (valid range is 18–100).
* Categorical normalization:

  * `marital`: `div.` → `divorced`
  * `education`: `sec.` → `secondary`, `UNK` → `unknown`
* Remove rows with nulls and rows with empty strings in object columns.

**Kept — credit-relevant features**

* `age`, `job`, `marital`, `education`, `default` (target), `balance`, `housing`, `loan`

---

## ML approach & components

**Target**: `default` (mapped to binary: `yes` → 1, `no` → 0) — the pipeline explicitly uses `default` as the credit-risk target.

**Preprocessing**

* Numeric: median imputation + `StandardScaler`
* Categorical:

  * One-Hot Encoding for low-cardinality categorical variables (<= 10 unique values)
  * `TargetEncoder` for higher-cardinality categorical variables
* Implementation via scikit-learn `ColumnTransformer` and Pipelines.

**Imbalance handling**

* SMOTE applied to the training set after preprocessing to create a balanced training set for model fitting.

**Models trained**

* `LogisticRegression` (liblinear)
* `RandomForestClassifier` (n\_estimators=100)
* `XGBClassifier` (XGBoost with `eval_metric='logloss'`)

**Model selection & evaluation**

* 3-fold `StratifiedKFold` cross-validation to compute CV ROC-AUC (mean & std).
* Models trained on balanced training set and evaluated on a held-out test set. Selection by highest test ROC-AUC.
* Script & app report CV vs test scores and stability (difference between CV mean and test AUC).

**Feature importance**

* Visualized where available (random forest / XGBoost feature importances or logistic coefficients).
* Additional analysis tries to combine multiple methods (mutual information, chi-square, permutation importance) to propose the top features used by the compact predictor.

---

## Visualizations & dashboard pages

The Streamlit dashboard (app.py) provides:

* **Overview**

  * Best model, default rate, total records, model stability summary.
  * Data quality summary & business interpretation including standard risk thresholds:

    * Low: 0–20% (auto-approve)
    * Medium: 20–40% (manual review)
    * High: 40%+ (likely reject)

* **Model Performance**

  * CV vs Test ROC-AUC comparison (bar chart)
  * ROC curve and confusion matrix for the selected best model
  * Feature importance (if available)
  * Prediction probability distributions and business impact metrics by threshold

* **Risk Predictor**

  * Compact dynamic form using the top statistically-important features
  * Single-case prediction with a gauge chart and textual recommendation
  * Displays which key feature values the prediction used

* **Dataset Analysis**

  * Cleaning report (what was removed/kept)
  * Dataset overview visualizations (target, age, balance, job distribution)
  * Basic statistics (numerical describe, categorical counts)
  * Data quality table (missing values, types & unique counts)

All plots are Plotly-based and the UI uses Streamlit for interactivity.

---

## Outputs

* `df_predict_CORRECTED.csv` — produced by the `credit_risk_assessment.py` script; contains the dataset with predicted default probabilities.
* Streamlit app displays interactive outputs and does not write to disk by default (except when you explicitly run the standalone script).

---

## Troubleshooting & notes

* **Dataset not found**: the utils loaders catch `FileNotFoundError` and Streamlit will show an error. Ensure `dataset_banco.csv` is in the project root.
* **Dependency issues**:

  * Make sure to install packages from `requirements.txt`.
  * Very large XGBoost builds or platform-specific wheels may require a compatible Python version and platform (use Python 3.8+ as targeted).
* **Model training is cached in Streamlit**: use the sidebar **Clear Cache & Retrain** button to clear caches if you update code or dataset.
* **Data leakage**: the pipeline explicitly removes features that represent marketing campaign outcomes (like `duration`) to avoid leakage into the credit-risk target — do not reintroduce marketing columns unless you change the modeling intention.
* **Reproducibility**: Random seeds (`random_state=42`) are used across the pipeline for reproducibility (splitting, SMOTE, CV shuffling, model seeds).

---

## Contributing

* Please open issues or PRs with clear descriptions.
* Keep changes limited to a single concern per PR (bugfix, new feature, docs).
* Follow the project's style (clear docstrings, type hints where helpful, tests if added later).

**Check [CONTRIBUTING.md](CONTRIBUTING.md) for more details**

---

## License

This repository is licensed under the [MIT LICENSE](LICENSE)