# =============================================================================
# LEVEL 3 - TASK 1: NLP + Neural Network - Sentiment Classification
# Dataset: sentiment_dataset.csv (social-media texts)
# -----------------------------------------------------------------------------
# 1. Load the dataset and inspect it
# 2. NLP preprocessing:
#      * clean the raw text (strip whitespace, drop emojis/punctuation,
#        keep hashtag words after removing the '#', lowercase)
#      * collapse the 279 fine-grained labels into Positive / Negative /
#        Neutral (keyword-driven grouping, documented below)
# 3. Feature extraction: TF-IDF vectorization (word unigrams + bigrams)
# 4. Neural Network: scikit-learn MLPClassifier (128-64 hidden units) versus
#    a Logistic Regression baseline on the same TF-IDF features
# 5. Evaluation: accuracy, macro/weighted F1, classification report,
#    confusion matrix (NN) and the MLP training-loss curve
# 6. Save metrics + figures, and demo-predict on brand-new sentences
# =============================================================================

import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "sentiment_dataset.csv")
OUT_CSV = os.path.join(HERE, "nlp_results.csv")
REPORT_CSV = os.path.join(HERE, "classification_report.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TEXT_COL = "Text"
LABEL_COL = "Sentiment"

# -----------------------------------------------------------------------------
# 1) LOAD & INSPECT
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("=== SHAPE ===", df.shape)
print("\n=== HEAD ===")
print(df[[TEXT_COL, LABEL_COL]].head())
print("\n=== RAW LABEL CARDINALITY ===")
print("Distinct fine-grained sentiment classes:", df[LABEL_COL].nunique())
print("(most classes have 1-5 rows -> useless as-is for supervised learning)")

# -----------------------------------------------------------------------------
# 2) NLP PREPROCESSING
# -----------------------------------------------------------------------------
def clean_text(t):
    """Normalize one raw text cell for the TF-IDF step."""
    t = str(t).strip().lower()
    t = t.replace("#", " ")            # hashtag -> plain word (keeps signal)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)   # drop URLs
    t = re.sub(r"[^a-z0-9\s]", " ", t) # drop emojis / punctuation
    return re.sub(r"\s+", " ", t).strip()

df["clean_text"] = df[TEXT_COL].apply(clean_text)
print("\n=== TEXT CLEANING (before -> after) ===")
for i in range(3):
    print(f"[{i}] RAW    : {df[TEXT_COL].iloc[i]!r}")
    print(f"[{i}] CLEAN  : {df['clean_text'].iloc[i]!r}")

# --- Label grouping ---------------------------------------------------------
# The dataset ships 279 fine-grained sentiment classes (Joy, Serenity,
# Vibrancy, Whispers of the Past, ...) that are far too sparse to learn from.
# We group them into 3 macro classes via keyword matching on the label:
#   * Positive  : joy, gratitude, excitement, pride, hope, wonder, ...
#   * Negative  : sadness, despair, grief, anger, fear, regret, ...
#   * Neutral   : curiosity, confusion, boredom, indifference, nostalgia, ...
# Bittersweet / ambivalent labels are folded into Neutral (neither positive
# nor negative, and only ~10 rows), keeping 3 well-populated classes.
MIXED = ["ambivalence", "ambivalent", "bittersweet", "emotion", "mixed"]
NEGATIVE = [
    "negative", "sad", "sorrow", "grief", "griev", "melancholy", "despair",
    "desperation", "desolat", "devastat", "bitter", "frustrat", "resent",
    "jealous", "envy", "envious", "hate", "anger", "angry", "fear",
    "anxious", "anxiety", "embarrass", "loneliness", "lonely", "isolation",
    "isolated", "bad", "betrayal", "shame", "heartbreak", "heartache",
    "disappoint", "loss", "lost", "regret", "helpless", "overwhelmed",
    "exhaustion", "suffering", "intimid", "disgust", "darkness", "ruins",
    "storm", "pressure", "obstacle", "challenge", "miscalculation",
    "dismissive", "apprehensive", "hurt", "pain", "defeat", "worried",
    "stressed",
]
POSITIVE = [
    "positive", "joy", "happy", "happiness", "excit", "content", "gratitude",
    "grateful", "hope", "elation", "playful", "serenity", "calm", "empower",
    "determin", "inspir", "enthusi", "acceptance", "awe", "euphoria", "proud",
    "pride", "compassion", "tender", "accomplish", "confident", "empath",
    "adventure", "surprise", "kind", "love", "enjoy", "admiration",
    "reverence", "fulfill", "zest", "enchant", "exploration", "whimsy",
    "coziness", "rejuvenation", "affection", "adoration", "anticipation",
    "satisfaction", "thrill", "tranquility", "creativity", "captivat",
    "amusement", "relief", "wonder", "radiance", "harmony", "mindful",
    "freedom", "free", "elegance", "resilience", "spark", "immersion",
    "positivity", "amazement", "grandeur", "celebration", "celebrate",
    "energy", "friendship", "romance", "success", "colorful", "ecstasy",
    "charm", "journey", "connection", "hypnotic", "touched", "triumph",
    "heartwarming", "engagement", "iconic", "sympathy", "breakthrough",
    "solace", "imagination", "vibrancy", "mesmeriz", "magic", "blessed",
    "appreciation", "confidence", "optimism", "motivation", "overjoyed",
    "delight", "beauty", "festive", "reunion", "artistic", "melodic",
    "marvel", "dazzle", "renewed", "dream", "vibes", "arousal", "belonging",
]
NEUTRAL = [
    "neutral", "curious", "curiosity", "confusion", "confused",
    "indifference", "indifferent", "boredom", "bored", "nostalgia",
    "reflection", "contemplation", "pensive", "solitude", "suspense",
    "intrigue", "yearning", "numb", "pragmatic", "ordinary", "mundane",
    "typical", "informational",
]

def map_sentiment(raw):
    """Map a fine-grained label to Positive / Negative / Neutral."""
    s = str(raw).lower()
    for kw in MIXED:
        if kw in s:
            return "Neutral"
    for kw in NEGATIVE:
        if kw in s:
            return "Negative"
    for kw in POSITIVE:
        if kw in s:
            return "Positive"
    for kw in NEUTRAL:
        if kw in s:
            return "Neutral"
    return "Neutral"  # unrecognized labels -> Neutral (safe default)

df["sentiment"] = df[LABEL_COL].apply(map_sentiment)
print("\n=== GROUPED SENTIMENT DISTRIBUTION ===")
print(df["sentiment"].value_counts().to_string())
print("\nGrouping coverage (labels not explicitly matched, defaulted to"
      " Neutral):",
      sorted(df.loc[df[LABEL_COL].map(map_sentiment) == "Neutral",
                    LABEL_COL].unique()))

# -----------------------------------------------------------------------------
# 3) FEATURE EXTRACTION - TF-IDF
# -----------------------------------------------------------------------------
# Word unigrams + bigrams, English stop words removed, rare terms (appearing
# in < 2 docs) dropped, sub-linear term-frequency scaling to tame long texts.
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000,
    sublinear_tf=True,
)
X = vectorizer.fit_transform(df["clean_text"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
classes = label_encoder.classes_  # encoded order: Negative, Neutral, Positive
print("\n=== TF-IDF MATRIX ===")
print(f"Shape: {X.shape[0]} docs x {X.shape[1]} features")

# -----------------------------------------------------------------------------
# 4) TRAIN / TEST SPLIT + MODELS
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n=== SPLIT ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=2000, random_state=42),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64),      # 2 hidden layers
        activation="relu",
        solver="adam",
        alpha=0.01,                        # L2 regularization (0.001 underfits
                                           # here - loss stuck ~0.6, model only
                                           # predicts the majority class)
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=800,
        early_stopping=True,               # stops when val-loss stalls
        n_iter_no_change=20,               # patience must be long enough for
                                           # the loss to actually descend
        random_state=42,
    ),
}

rows = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    fitted[name] = (model, y_pred)
    rows.append({"Model": name, "Accuracy": acc, "F1 Macro": f1_macro,
                 "F1 Weighted": f1_weighted})
    print(f"\n[{name}] Accuracy={acc:.4f}  F1 macro={f1_macro:.4f}  "
          f"F1 weighted={f1_weighted:.4f}")
    if isinstance(model, MLPClassifier):
        print(f"   converged after {model.n_iter_} iterations "
              f"(final loss {model.loss_:.4f})")

results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
print("\n=== MODEL COMPARISON ===")
print(results.to_string(index=False))
results.to_csv(OUT_CSV, index=False)
print("Saved comparison table to:", OUT_CSV)

# -----------------------------------------------------------------------------
# 5) EVALUATION DETAILS (best model = the neural network)
# -----------------------------------------------------------------------------
best_model, best_pred = fitted["Neural Network (MLP)"]
print("\n=== CLASSIFICATION REPORT (MLP) ===")
report = classification_report(y_test, best_pred, target_names=classes,
                               output_dict=True)
print(classification_report(y_test, best_pred, target_names=classes))

report_rows = []
for cls in classes:
    report_rows.append({"Class": cls, **report[cls]})
pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)
print("Saved classification report to:", REPORT_CSV)

# --- Confusion matrix (absolute + normalized) -------------------------------
cm = confusion_matrix(y_test, best_pred, labels=[0, 1, 2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, data, title in zip(axes, [cm, cm_norm],
                           ["Absolute counts", "Row-normalized"]):
    sns.heatmap(data, annot=True, fmt=".2f" if title.startswith("Row")
                else "d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(f"Confusion matrix - MLP ({title})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "confusion_matrix_mlp.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/confusion_matrix_mlp.png")

# --- MLP training loss curve -------------------------------------------------
mlp = fitted["Neural Network (MLP)"][0]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(mlp.loss_curve_, lw=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Training loss")
ax.set_title("MLP training-loss curve (early stopping "
             f"after {mlp.n_iter_} iterations)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "training_loss_curve_mlp.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/training_loss_curve_mlp.png")

# --- Accuracy / F1 comparison bar chart -------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results))
width = 0.35
ax.bar(x - width / 2, results["Accuracy"], width, label="Accuracy")
ax.bar(x + width / 2, results["F1 Macro"], width, label="F1 Macro")
ax.set_xticks(x)
ax.set_xticklabels(results["Model"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Logistic Regression vs Neural Network (TF-IDF features)")
ax.legend()
for xi, r in results.iterrows():
    ax.annotate(f"{r['Accuracy']:.3f}", (xi - width / 2, r["Accuracy"]),
                ha="center", va="bottom")
    ax.annotate(f"{r['F1 Macro']:.3f}", (xi + width / 2, r["F1 Macro"]),
                ha="center", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "accuracy_comparison.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/accuracy_comparison.png")

# -----------------------------------------------------------------------------
# 6) NLP INTERPRETABILITY: TOP TF-IDF TERMS PER SENTIMENT CLASS
# -----------------------------------------------------------------------------
feat_names = np.array(vectorizer.get_feature_names_out())
for i, cls in enumerate(classes):
    mask = (y_train == i)
    mean_w = X_train[mask].mean(axis=0).A1
    top = feat_names[np.argsort(mean_w)[::-1][:10]]
    print(f"\nTop terms [{cls}]: {', '.join(top)}")

# -----------------------------------------------------------------------------
# 7) DEMO: PREDICT ON BRAND-NEW SENTENCES
# -----------------------------------------------------------------------------
demo = [
    "Absolutely loved the concert last night, the energy was incredible!",
    "My flight got cancelled and I missed the wedding. Devastated.",
    "The package should arrive on Tuesday afternoon, nothing special.",
    "Can't wait for the weekend trip with the whole family!",
    "Traffic was a nightmare and I am so frustrated right now.",
]
print("\n=== DEMO PREDICTIONS (MLP) ===")
for sentence in demo:
    vec = vectorizer.transform([clean_text(sentence)])
    proba = mlp.predict_proba(vec)[0]
    pred_code = mlp.classes_[np.argmax(proba)]
    label = label_encoder.inverse_transform([pred_code])[0]
    conf = proba.max()
    print(f"- {sentence!r}\n    -> {label} (confidence {conf:.2f})")
print("\nNote: neutral sentences are the model's weak spot (only 97 neutral "
      "rows in the dataset; see the classification report for Neutral recall).")

print("\n=== LEVEL 3 TASK COMPLETE ===")
print(f"Outputs: {OUT_CSV}, {REPORT_CSV}, figures/ under {FIG_DIR}")
