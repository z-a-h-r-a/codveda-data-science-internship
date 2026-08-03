# Codveda Data Science Internship

Three-level internship project built with Python, pandas, matplotlib, seaborn and
scikit-learn.

## Progress Summary

| Level | Status      |
|-------|-------------|
| Level 1 | **Complete** |
| Level 2 | **Complete** |
| Level 3 | **Not started** |

## Repository Layout

```
codveda-data-science-internship/
├── data/                          # raw datasets used by the notebooks/scripts
├── level1/
│   ├── task2_cleaning.py          # data cleaning & preprocessing (sentiment)
│   ├── task3_eda.py               # exploratory data analysis (iris)
│   ├── cleaned_sentiment_data.csv # cleaned output of Task 2
│   ├── iris_summary_stats.csv     # per-species summary statistics
│   ├── eda_summary.md             # 3-5 bullet EDA insights
│   └── figures/                   # all EDA plots (.png)
├── level2/
│   ├── task1_regression.py        # house-price regression (LR / DT / RF)
│   ├── task2_classification.py    # churn classification (LogReg / RF)
│   ├── regression_results.csv     # model comparison (MSE / R2)
│   ├── classification_results.csv # model comparison (Acc / P / R / F1 / AUC)
│   └── figures/                   # predicted-vs-actual & ROC plots
└── level3/                        # NOT STARTED - intentionally empty
```

## Level 1

### Task 2 - Data Cleaning & Preprocessing (sentiment dataset)
- Stripped whitespace from all string columns (Text, User, Platform, Hashtags,
  Country, Sentiment, Timestamp).
- Dropped two duplicate spreadsheet index columns (`Unnamed: 0`, `Unnamed: 0.1`).
- Missing values: reported counts; strategy = median imputation for numerics,
  row-drop for core text columns (no rows actually needed imputation).
- Outliers in Retweets (18) and Likes (18) detected via the IQR fences and
  capped (winsorized) rather than dropped.
- Encoded categoricals: one-hot for Platform, label encoding for Country and
  Sentiment.
- Standardized Retweets and Likes with StandardScaler (mean 0, std 1).
- Output: `level1/cleaned_sentiment_data.csv`.

### Task 3 - Exploratory Data Analysis (iris dataset)
- Per-species summary stats (mean / median / std / min / max).
- Histograms, scatter plots (petal & sepal length vs width, colored by species),
  box plots and a correlation heatmap - all saved under `level1/figures/`.
- Key insights written to `level1/eda_summary.md`.

## Level 2

### Task 1 - Regression (Boston-Housing-style dataset)
- Loaded headerless file and assigned the 14 columns
  (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV).
- Trained Linear Regression, Decision Tree and Random Forest on an 80/20 split.
- Best model: **Random Forest** (Test MSE = 7.90, R2 = 0.89).
- Predicted-vs-actual plot saved under `level2/figures/`.
- Results saved to `level2/regression_results.csv`.

### Task 2 - Classification (customer churn)
- Used the pre-split churn-bigml-80.csv (train) and churn-bigml-20.csv (test)
  without re-splitting.
- Encoded State (one-hot) and the two Yes/No plan columns; scaled numerics with
  StandardScaler (fitted on train only).
- Trained Logistic Regression and Random Forest; reported accuracy, precision,
  recall, F1 and AUC, plus ROC curves.
- Best model: **Random Forest** (Accuracy = 0.94, F1 = 0.72, AUC = 0.90).
- Results saved to `level2/classification_results.csv`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Level 1
python level1/task2_cleaning.py
python level1/task3_eda.py

# Level 2
python level2/task1_regression.py
python level2/task2_classification.py
```
