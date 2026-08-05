# =============================================================================
# LEVEL 3 - TACHE 2: NLP / Classification de texte
# Dataset: sentiment_dataset.csv (colonne Text + cible Sentiment)
# -----------------------------------------------------------------------------
# 1. Pretraitement texte (nltk):
#      * tokenisation          -> nltk.word_tokenize
#      * stopwords             -> nltk.corpus.stopwords (langue anglaise)
#      * stemming              -> nltk.PorterStemmer
#      * lemmatisation         -> nltk.WordNetLemmatizer
#    Les 279 labels fins sont regroupes en Positive / Negative / Neutral
#    (meme logique que la tache precedente).
# 2. Vectorisation TF-IDF (unigrammes + bigrammes, stop words supprimes).
# 3. Modeles: Multinomial Naive Bayes vs Logistic Regression.
# 4. Evaluation: precision / recall / F1 par classe + moyennes macro/weighted,
#    matrices de confusion.
# 5. Sorties: nlp_results.csv, figures/.
# =============================================================================

import sys
import os
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "sentiment_dataset.csv")
OUT_CSV = os.path.join(HERE, "nlp_results.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 0) NLTK RESOURCES (auto-download if missing, so it runs on any machine)
# -----------------------------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] nltk resource '{resource}' failed: {exc}")

STOPWORDS = set(nltk.corpus.stopwords.words("english"))
STEMMER = nltk.PorterStemmer()
LEMMATIZER = nltk.WordNetLemmatizer()

# -----------------------------------------------------------------------------
# 1) PRETRAITEMENT TEXTE
# -----------------------------------------------------------------------------
def tokenize(text):
    """Lowercase, strip punctuation/emojis, keep alphanumeric tokens."""
    text = str(text).strip().lower().replace("#", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.findall(r"\b[a-z0-9]+\b", text)

def preprocess_pipeline(text, stem=True, lemmatize=True, keep_stops=False):
    """Full NLP chain: tokenise -> drop stopwords -> stem -> lemmatize."""
    tokens = tokenize(text)
    if not keep_stops:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------------------------------------------------------
# 2) CHARGEMENT + REGROUPEMENT DE LA CIBLE
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("=== SHAPE ===", df.shape)
print("Distinct fine-grained sentiment classes:", df["Sentiment"].nunique())

# Le dataset contient 279 labels fins tres peu remplis. On les regroupe en
# 3 classes macro via un matching par mots-cles sur le label.
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
    return "Neutral"

df["sentiment"] = df["Sentiment"].apply(map_sentiment)
print("\n=== SENTIMENT DISTRIBUTION (apres regroupement) ===")
print(df["sentiment"].value_counts().to_string())

# -----------------------------------------------------------------------------
# 3) VECTORISATION TF-IDF
# -----------------------------------------------------------------------------
# Les textes passent par le pipeline complet (tokenisation + stopwords +
# stemming + lemmatisation) avant la vectorisation TF-IDF.
df["processed"] = df["Text"].apply(preprocess_pipeline)
print("\n=== EXEMPLE DE PRETRAITEMENT ===")
for i in range(3):
    print(f"[{i}] RAW      : {df['Text'].iloc[i]!r}")
    print(f"[{i}] PROCESSED: {df['processed'].iloc[i]!r}")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)
X = vectorizer.fit_transform(df["processed"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
classes = label_encoder.classes_  # Negative, Neutral, Positive
print("\n=== TF-IDF MATRIX ===")
print(f"Shape: {X.shape[0]} docs x {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n=== SPLIT ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -----------------------------------------------------------------------------
# 4) MODELES: NAIVE BAYES vs LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
models = {
    "Naive Bayes (Multinomial)": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000,
                                              random_state=42),
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

results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
print("\n=== COMPARAISON DES MODELES ===")
print(results.to_string(index=False))
results.to_csv(OUT_CSV, index=False)
print("Saved comparison table to:", OUT_CSV)

# -----------------------------------------------------------------------------
# 5) EVALUATION DETAILLEE (precision / recall / F1)
# -----------------------------------------------------------------------------
for name, (_, y_pred) in fitted.items():
    print(f"\n=== CLASSIFICATION REPORT [{name}] ===")
    print(classification_report(y_test, y_pred, target_names=classes,
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, ttl in zip(axes, [cm, cm_norm],
                             ["Absolute counts", "Row-normalized"]):
        sns.heatmap(data, annot=True, fmt=".2f" if ttl.startswith("Row")
                    else "d", cmap="Blues", ax=ax,
                    xticklabels=classes, yticklabels=classes)
        ax.set_title(f"Confusion matrix - {name} ({ttl})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    tag = name.split()[0].lower().replace("(", "").replace(")", "")
    fig.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{tag}.png"), dpi=120)
    plt.close(fig)
    print("Saved figure:", f"figures/confusion_matrix_{tag}.png")

# -----------------------------------------------------------------------------
# 6) BONUS: COMPARAISON DES VARIANTES DE PRETRAITEMENT (impact stemming)
# -----------------------------------------------------------------------------
print("\n=== IMPACT DU PRETRAITEMENT (TF-IDF, Logistic Regression) ===")
preproc_variants = {
    "brut (sans NLP)": lambda t: " ".join(tokenize(t)),
    "sans stemming": lambda t: preprocess_pipeline(t, stem=False),
    "pipeline complet": preprocess_pipeline,
}
base_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
for label, fn in preproc_variants.items():
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000,
                        sublinear_tf=True)
    Xv = v.fit_transform(df["Text"].apply(fn))
    Xvt, Xve, yvt, yve = train_test_split(
        Xv, y, test_size=0.2, random_state=42, stratify=y)
    base_lr.fit(Xvt, yvt)
    acc = accuracy_score(yve, base_lr.predict(Xve))
    print(f"- {label:24s} -> accuracy = {acc:.4f}")

print("\n=== TACHE 2 TERMINEE ===")
print(f"Outputs: {OUT_CSV}, figures/ sous {FIG_DIR}")
