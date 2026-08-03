# =============================================================================
# LEVEL 2 - TASK 1: Predictive Modeling - Regression
# Dataset: house_prediction_data_set.csv (Boston-Housing style, NO header row)
# -----------------------------------------------------------------------------
# 1. Load data and assign the 14 column names
# 2. Sanity checks (shape, dtypes, missing values, describe)
# 3. Train/test split
# 4. Linear Regression  -> MSE & R^2
# 5. Decision Tree Regressor & Random Forest Regressor, compare in a table
# 6. Plot predicted vs actual for the best model
# 7. Save comparison results to level2/regression_results.csv
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "../data/house_prediction_data_set.csv"
OUT_CSV = "regression_results.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
           "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# -----------------------------------------------------------------------------
# 1) LOAD DATA (no header; whitespace-separated) & assign column names
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
print("=== SHAPE ===", df.shape)
print("\n=== HEAD ===")
print(df.head())

# -----------------------------------------------------------------------------
# 2) SANITY CHECKS
# -----------------------------------------------------------------------------
print("\n=== DTYPES ===")
print(df.dtypes)
print("\n=== MISSING VALUES ===")
print(df.isna().sum().sum())
print("\n=== DESCRIBE ===")
print(df.describe())
print("\n=== DUPLICATES ===", df.duplicated().sum())

# -----------------------------------------------------------------------------
# 3) TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------
X = df.drop(columns=["MEDV"])
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\n=== SPLIT ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -----------------------------------------------------------------------------
# 4+5) TRAIN MODELS AND COMPARE
# -----------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

rows = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    tr_pred = model.predict(X_train)
    tr_mse = mean_squared_error(y_train, tr_pred)
    tr_r2 = r2_score(y_train, tr_pred)
    fitted[name] = (model, y_pred)
    rows.append({"Model": name, "Test MSE": mse, "Test R2": r2,
                 "Train MSE": tr_mse, "Train R2": tr_r2})
    print(f"[{name}] Test MSE = {mse:.4f}, Test R2 = {r2:.4f} | "
          f"Train MSE = {tr_mse:.4f}, Train R2 = {tr_r2:.4f}")

results = pd.DataFrame(rows).sort_values("Test MSE")
print("\n=== MODEL COMPARISON (sorted by Test MSE) ===")
print(results.to_string(index=False))
results.to_csv(OUT_CSV, index=False)
print("Saved comparison table to:", OUT_CSV)

# -----------------------------------------------------------------------------
# 6) PREDICTED VS ACTUAL FOR THE BEST MODEL (lowest test MSE)
# -----------------------------------------------------------------------------
best_name = results.iloc[0]["Model"]
best_model, best_pred = fitted[best_name]

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(y_test, best_pred, alpha=0.6, edgecolor="k", s=45)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction (y = x)")
ax.set_xlabel("Actual MEDV")
ax.set_ylabel("Predicted MEDV")
ax.set_title(f"Predicted vs Actual MEDV - {best_name}\n"
             f"Test MSE = {results.iloc[0]['Test MSE']:.3f}, "
             f"Test R2 = {results.iloc[0]['Test R2']:.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "predicted_vs_actual_best_model.png"), dpi=120)
plt.close(fig)
print(f"\nBest model: {best_name}")
print("Saved figure: figures/predicted_vs_actual_best_model.png")
