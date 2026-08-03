# =============================================================================
# LEVEL 2 - TASK 2: Classification - Customer Churn
# Datasets: churn-bigml-80.csv (train) / churn-bigml-20.csv (test)
#           -> already pre-split, they are used AS-IS (no re-split)
# -----------------------------------------------------------------------------
# 1. Load train/test files and inspect them
# 2. Preprocess: encode categorical features + scale numeric features
#    (encoders/scalers are fit on the TRAIN split only)
# 3. Logistic Regression -> accuracy, precision, recall, F1 + ROC curve
# 4. Random Forest -> same metrics, comparison table
# 5. Save comparison results to level2/classification_results.csv
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_PATH = "../data/churn-bigml-80.csv"
TEST_PATH = "../data/churn-bigml-20.csv"
OUT_CSV = "classification_results.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) LOAD & INSPECT
# -----------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print("=== SHAPES ===", train.shape, test.shape)
print("\n=== TRAIN HEAD ===")
print(train.head())
print("\n=== DTYPES ===")
print(train.dtypes)
print("\n=== MISSING VALUES (train / test) ===",
      train.isna().sum().sum(), "/", test.isna().sum().sum())
print("\n=== CHURN DISTRIBUTION (train / test) ===")
print(train["Churn"].value_counts(), test["Churn"].value_counts())
print("\n=== STATES train vs test (overlap) ===")
tr_st, te_st = set(train["State"]), set(test["State"])
print("states in test but not train:", sorted(te_st - tr_st))

# -----------------------------------------------------------------------------
# 2) PREPROCESSING
# -----------------------------------------------------------------------------
TARGET = "Churn"          # boolean column, kept as-is (False/True)
BINARY_PLAN = ["International plan", "Voice mail plan"]   # Yes/No -> 0/1
CATEGORICAL = ["State"]                                  # one-hot encoded
NUMERIC = [c for c in train.columns
           if c not in [TARGET] + BINARY_PLAN + CATEGORICAL]

# Map the two Yes/No plan columns to integers (fit mapping on train only).
plan_map = {col: {"Yes": 1, "No": 0} for col in BINARY_PLAN}
for col in BINARY_PLAN:
    train[col] = train[col].map(plan_map[col])
    test[col] = test[col].map(plan_map[col])

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].astype(int)

# ColumnTransformer: scale numerics, one-hot encode State.
# fitted on the TRAIN split, then transform test (handle_unknown='ignore'
# covers any state that only appears in the test split).
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ])

X_train_pp = preprocessor.fit_transform(X_train)
X_test_pp = preprocessor.transform(X_test)
print("\n=== PREPROCESSED SHAPES ===", X_train_pp.shape, X_test_pp.shape)

# -----------------------------------------------------------------------------
# 3+4) TRAIN CLASSIFIERS AND COMPARE
# -----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

rows = []
fitted_proba = {}
for name, model in models.items():
    model.fit(X_train_pp, y_train)
    y_pred = model.predict(X_test_pp)
    y_proba = model.predict_proba(X_test_pp)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    fitted_proba[name] = y_proba
    rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                 "Recall": rec, "F1 Score": f1, "AUC": auc})
    print(f"[{name}] Accuracy={acc:.4f} Precision={prec:.4f} "
          f"Recall={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

results = pd.DataFrame(rows)
print("\n=== CLASSIFIER COMPARISON ===")
print(results.to_string(index=False))
results.to_csv(OUT_CSV, index=False)
print("Saved comparison table to:", OUT_CSV)

# -----------------------------------------------------------------------------
# 5) ROC CURVES (both models, with AUC in the legend)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, y_proba in fitted_proba.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Customer Churn Classification")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/roc_curves.png")
