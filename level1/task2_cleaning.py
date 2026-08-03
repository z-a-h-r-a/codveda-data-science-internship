# =============================================================================
# LEVEL 1 - TASK 2: Data Cleaning and Preprocessing
# Dataset: sentiment_dataset.csv (messy string columns + missing values)
# -----------------------------------------------------------------------------
# Pipeline overview:
#   1. Load the dataset and inspect it
#   2. Drop duplicate index columns
#   3. Strip leading/trailing whitespace from all string columns
#   4. Handle missing values (report counts, decide impute vs drop)
#   5. Detect & handle outliers in numeric columns (Retweets, Likes) via IQR
#   6. Encode categorical columns (Platform, Country, Sentiment)
#   7. Standardize numeric columns with StandardScaler
#   8. Save the cleaned dataset to level1/cleaned_sentiment_data.csv
# =============================================================================

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# -----------------------------------------------------------------------------
# 1) LOAD AND INSPECT
# -----------------------------------------------------------------------------
DATA_PATH = "../data/sentiment_dataset.csv"
OUT_PATH = "cleaned_sentiment_data.csv"

df = pd.read_csv(DATA_PATH)
print("=== ORIGINAL SHAPE ===", df.shape)
print("\n=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== DIMS / DTYPES ===")
print(df.dtypes)

# The file was exported from a spreadsheet and carries two leftover index
# columns ("Unnamed: 0" and "Unnamed: 0.1"). They are pure row counters that
# add no information, so we drop them.
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

# -----------------------------------------------------------------------------
# 2) STRIP WHITESPACE FROM STRING COLUMNS
# -----------------------------------------------------------------------------
# Many object columns have leading/trailing spaces (e.g. " Positive  ",
# " Twitter  "). We strip every string cell so categories match after encoding.
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
print("\n=== STRING COLUMNS STRIPPED ===", list(str_cols))
print("Sample Platform values after strip:", df["Platform"].unique())

# -----------------------------------------------------------------------------
# 3) HANDLE MISSING VALUES
# -----------------------------------------------------------------------------
# First, also treat fully-empty strings (whitespace-only) as missing.
df = df.replace(r"^\s*$", np.nan, regex=True)

missing = df.isna().sum()
print("\n=== MISSING VALUES PER COLUMN (after strip) ===")
print(missing[missing > 0] if missing.any() else "No missing values found.")

# DECISION / JUSTIFICATION
# -------------------------
# * Numeric columns: if any were missing we would impute with the MEDIAN,
#   because Retweets/Likes are right-skewed and the median is robust to
#   outliers. (Median imputation keeps all rows for modeling.)
# * Object columns: if any were missing we would DROP those rows, because
#   the Text / User / Hashtags content is the core of the record and cannot
#   be invented; imputation would fabricate data.
# * In practice this file contains no missing cells, so no rows are dropped
#   and no numeric imputation is triggered. The logic is kept so the pipeline
#   is safe on messier versions of the same dataset.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
df = df.dropna(subset=["Text", "User", "Hashtags"])

# -----------------------------------------------------------------------------
# 4) DETECT & HANDLE OUTLIERS IN NUMERIC COLUMNS (IQR METHOD)
# -----------------------------------------------------------------------------
# Box-plots of Retweets/Likes show high-value spikes (e.g. Retweets up to 40
# while Q3 = 25, Likes up to 80 while Q3 = 50). We cap (winsorize) values
# beyond the IQR fences instead of dropping rows, so no tweets are lost while
# extreme values can no longer distort scaling / models.
for col in ["Retweets", "Likes"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_before = int(((df[col] < lower) | (df[col] > upper)).sum())
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"\n[IQR] {col}: fences [{lower:.2f}, {upper:.2f}] -> "
          f"{n_before} outliers detected and capped.")

# -----------------------------------------------------------------------------
# 5) ENCODE CATEGORICAL COLUMNS
# -----------------------------------------------------------------------------
# * Platform  -> one-hot encoding (only 4 values, perfect for one-hot).
# * Country   -> label encoding (115 countries; one-hot would create a huge
#                sparse matrix on just 732 rows, so integer labels are used).
# * Sentiment -> label encoding (many fine-grained classes, same reasoning).
print("\n=== CATEGORICAL ENCODING ===")

platform_dummies = pd.get_dummies(df["Platform"], prefix="Platform").astype(int)
print("One-hot Platform columns:", list(platform_dummies.columns))

le_country = LabelEncoder()
country_enc = le_country.fit_transform(df["Country"])
print("Label-encoded Country classes:", le_country.classes_.size)

le_sentiment = LabelEncoder()
sentiment_enc = le_sentiment.fit_transform(df["Sentiment"])
print("Label-encoded Sentiment classes:", le_sentiment.classes_.size)

# -----------------------------------------------------------------------------
# 6) STANDARDIZE NUMERIC COLUMNS (StandardScaler)
# -----------------------------------------------------------------------------
# Retweets and Likes are the real numeric features. Year/Month/Day/Hour are
# temporal components kept untouched. We standardize to mean=0, std=1 so the
# values are directly comparable for distance-based models.
scaler = StandardScaler()
retweets_scaled = scaler.fit_transform(df[["Retweets"]])
likes_scaled = scaler.fit_transform(df[["Likes"]])
print("\n=== STANDARDIZATION ===")
print("Retweets after scaling  -> mean {:.4f}, std {:.4f}".format(
    retweets_scaled.mean(), retweets_scaled.std()))
print("Likes    after scaling  -> mean {:.4f}, std {:.4f}".format(
    likes_scaled.mean(), likes_scaled.std()))

# -----------------------------------------------------------------------------
# 7) ASSEMBLE & SAVE CLEANED DATASET
# -----------------------------------------------------------------------------
cleaned = pd.DataFrame({
    "Text": df["Text"],
    "User": df["User"],
    "Hashtags": df["Hashtags"],
    **platform_dummies,
    "Country_encoded": country_enc,
    "Sentiment_encoded": sentiment_enc,
    "Retweets_scaled": retweets_scaled.ravel(),
    "Likes_scaled": likes_scaled.ravel(),
    "Year": df["Year"],
    "Month": df["Month"],
    "Day": df["Day"],
    "Hour": df["Hour"],
})
cleaned.to_csv(OUT_PATH, index=False)

print("\n=== CLEANED DATASET ===")
print("Final shape:", cleaned.shape)
print("Saved to:", OUT_PATH)
print(cleaned.head())
