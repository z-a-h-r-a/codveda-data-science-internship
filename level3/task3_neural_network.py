# =============================================================================
# LEVEL 3 - TACHE 3: Reseaux de neurones (Keras)
# Dataset: iris.csv (classification simple, 4 features -> 3 especes)
# -----------------------------------------------------------------------------
# NOTE: TensorFlow n'a pas de wheel pour Python 3.14 sur cette machine
# ("No matching distribution found"). On utilise donc Keras 3 avec le backend
# PyTorch (CPU) - l'API Keras reste identique a TensorFlow/Keras.
#
# 1. Chargement + standardisation des features (StandardScaler)
# 2. Reseau feed-forward (Sequential) entraine par retropropagation
#      Input(4) -> Dense(16, relu) -> Dense(8, relu) -> Dense(3, softmax)
#    Optimiseur Adam, perte sparse-categorical-crossentropy.
# 3. Courbes accuracy / loss (train + validation)
# 4. Tuning d'hyperparametres (unites cachees, taux d'apprentissage, epochs)
# 5. Evaluation finale: accuracy, classification report, confusion matrix
# 6. Sorties: nn_results.csv, figures/.
# =============================================================================

import sys
import os
import random

# Keras 3 doit connaitre son backend AVANT l'import de keras/torch.
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import keras
from keras import layers, callbacks

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Reproducibilite
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "iris.csv")
OUT_CSV = os.path.join(HERE, "nn_results.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("Backend Keras:", keras.backend.backend())

# -----------------------------------------------------------------------------
# 1) CHARGEMENT + PREPARATION
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("\n=== IRIS DATASET ===")
print(df.head())
print("Shape:", df.shape)
print("Classes:", df["species"].value_counts().to_dict())

feature_cols = [c for c in df.columns if c != "species"]
X = df[feature_cols].to_numpy(dtype="float32")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["species"])
classes = label_encoder.classes_  # setosa, versicolor, virginica
print("Features:", feature_cols, "| Target classes:", classes)

# Standardiser (les reseaux sont sensibles aux echelles des features).
scaler = StandardScaler()
X = scaler.fit_transform(X).astype("float32")

# Split 60% train / 20% val / 20% test (stratifie)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train)
print(f"\n=== SPLIT ===")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -----------------------------------------------------------------------------
# 2) FABRIQUE DE MODELES (feed-forward)
# -----------------------------------------------------------------------------
def make_model(hidden_units, learning_rate, input_dim):
    """Simple reseau feed-forward: couches Dense relu + sortie softmax."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(len(classes), activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------------------------------------------------------
# 3) TUNING D'HYPERPARAMETRES (petite grille manuelle)
# -----------------------------------------------------------------------------
grid = [
    {"hidden": (16,),         "lr": 0.001, "name": "16 units, lr=1e-3"},
    {"hidden": (32,),         "lr": 0.001, "name": "32 units, lr=1e-3"},
    {"hidden": (16, 8),       "lr": 0.001, "name": "16-8 units, lr=1e-3"},
    {"hidden": (16,),         "lr": 0.01,  "name": "16 units, lr=1e-2"},
    {"hidden": (32,),         "lr": 0.01,  "name": "32 units, lr=1e-2"},
    {"hidden": (16, 8),       "lr": 0.01,  "name": "16-8 units, lr=1e-2"},
]

EPOCHS = 200
EARLY = callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True)

print("\n=== TUNING D'HYPERPARAMETRES (sur la validation) ===")
tuning_rows = []
histories = {}
for cfg in grid:
    model = make_model(cfg["hidden"], cfg["lr"], X.shape[1])
    hist = model.fit(
        X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val),
        verbose=0, callbacks=[EARLY])
    val_acc = float(max(hist.history["val_accuracy"]))
    val_loss = float(min(hist.history["val_loss"]))
    epochs_used = len(hist.history["loss"])
    tuning_rows.append({"Config": cfg["name"], "Val Accuracy": val_acc,
                        "Val Loss": val_loss, "Epochs": epochs_used})
    histories[cfg["name"]] = hist
    print(f"- {cfg['name']:24s} -> val_acc={val_acc:.4f} "
          f"val_loss={val_loss:.4f} ({epochs_used} epochs)")

tuning = pd.DataFrame(tuning_rows).sort_values(
    "Val Accuracy", ascending=False)
print("\n=== CLASSEMENT (tuning) ===")
print(tuning.to_string(index=False))
tuning.to_csv(OUT_CSV, index=False)
print("Saved tuning table to:", OUT_CSV)

best_cfg = tuning.iloc[0]["Config"]
print(f"\nBest config: {best_cfg}")

# -----------------------------------------------------------------------------
# 4) ENTRAINEMENT FINAL DU MEILLEUR MODELE (train + val)
# -----------------------------------------------------------------------------
best = grid[[r["name"] for r in grid].index(best_cfg)]
final_model = make_model(best["hidden"], best["lr"], X.shape[1])
X_final_train = np.concatenate([X_train, X_val], axis=0)
y_final_train = np.concatenate([y_train, y_val], axis=0)

print("\n=== ENTRAINEMENT FINAL (train+val, validation=test) ===")
final_hist = final_model.fit(
    X_final_train, y_final_train, epochs=EPOCHS, validation_data=(X_test, y_test),
    verbose=0, callbacks=[EARLY])
print(f"Entraine sur {len(X_final_train)} exemples, "
      f"{len(final_hist.history['loss'])} epochs")

y_pred = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n=== EVALUATION FINALE SUR TEST ===")
print(f"Accuracy = {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=classes))

# -----------------------------------------------------------------------------
# 5) FIGURES
# -----------------------------------------------------------------------------
# Courbes accuracy / loss (meilleur modele)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(final_hist.history["accuracy"], label="train")
axes[0].plot(final_hist.history["val_accuracy"], label="test (val)")
axes[0].set_title(f"Accuracy - {best_cfg}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(final_hist.history["loss"], label="train")
axes[1].plot(final_hist.history["val_loss"], label="test (val)")
axes[1].set_title(f"Loss - {best_cfg}")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "nn_training_curves.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/nn_training_curves.png")

# Barplot du tuning
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(tuning["Config"], tuning["Val Accuracy"], color="steelblue")
ax.axhline(test_acc, ls="--", color="red",
           label=f"Best model on test = {test_acc:.3f}")
ax.set_xticks(range(len(tuning)))
ax.set_xticklabels(tuning["Config"], rotation=30, ha="right")
ax.set_ylabel("Validation accuracy")
ax.set_title("Hyperparameter tuning - validation accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "nn_hyperparameter_tuning.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/nn_hyperparameter_tuning.png")

# Confusion matrix du meilleur modele
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=classes, yticklabels=classes)
ax.set_title(f"Confusion matrix - Neural Network (test, acc={test_acc:.3f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "nn_confusion_matrix.png"), dpi=120)
plt.close(fig)
print("Saved figure: figures/nn_confusion_matrix.png")

print("\n=== TACHE 3 TERMINEE ===")
print(f"Outputs: {OUT_CSV}, figures/ sous {FIG_DIR}")
