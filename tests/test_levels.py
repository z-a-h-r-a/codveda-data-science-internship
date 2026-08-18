# =============================================================================
# Tests de validation des 3 niveaux du projet Codveda Data Science Internship
# -----------------------------------------------------------------------------
# 1. Execution des 6 scripts (subprocess, depuis leur propre dossier)
# 2. Validation du contenu des sorties (CSV, figures, fichiers auxiliaires)
# 3. Sanity checks sur les datasets bruts
#
# Lancer depuis la racine du projet :
#   python -m pytest tests -v
# =============================================================================

import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Scripts a executer : (cle, chemin relatif depuis la racine) -------------
SCRIPTS = [
    ("level1/task2_cleaning.py",),
    ("level1/task3_eda.py",),
    ("level2/task1_regression.py",),
    ("level2/task2_classification.py",),
    ("level3/task2_nlp_classification.py",),
    ("level3/task3_neural_network.py",),
]

LEVEL1_FIGURES = ["histograms.png", "scatter_plots.png", "boxplots.png",
                  "correlation_heatmap.png"]
LEVEL2_FIGURES = ["predicted_vs_actual_best_model.png", "roc_curves.png"]
LEVEL3_FIGURES = ["confusion_matrix_naive.png", "confusion_matrix_logistic.png",
                  "nn_training_curves.png", "nn_hyperparameter_tuning.png",
                  "nn_confusion_matrix.png"]


def _p(*parts):
    return os.path.join(ROOT, *parts)


def _run_script(rel_script, timeout=1500):
    """Execute un script dans son propre dossier et renvoie le CompletedProcess."""
    script = _p(rel_script)
    cwd = os.path.dirname(script)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, script], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def _check_returncode(result, name):
    assert result.returncode == 0, (
        f"[{name}] a echoue (code {result.returncode})\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")


def _check_figures(fig_dir, names):
    for name in names:
        path = _p(fig_dir, name)
        assert os.path.exists(path), f"figure absente: {path}"
        assert os.path.getsize(path) > 5000, f"figure vide/tronquee: {path}"


def _csv(rel_path):
    path = _p(rel_path)
    assert os.path.exists(path), f"CSV absent: {path}"
    return pd.read_csv(path)


# =============================================================================
# Fixture session : execute tous les scripts une seule fois
# =============================================================================
@pytest.fixture(scope="session")
def run_all_scripts():
    results = {}
    for (rel,) in SCRIPTS:
        results[rel] = _run_script(rel)
    return results


# -----------------------------------------------------------------------------
# 1) EXECUTION
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("rel", [s[0] for s in SCRIPTS])
def test_script_runs(run_all_scripts, rel):
    _check_returncode(run_all_scripts[rel], rel)


# -----------------------------------------------------------------------------
# 2) SANITY CHECKS DES DATASETS
# -----------------------------------------------------------------------------
def test_dataset_sentiment():
    df = pd.read_csv(_p("data", "sentiment_dataset.csv"))
    assert df.shape[0] == 732, "sentiment_dataset devrait avoir 732 lignes"
    assert "Text" in df.columns and "Sentiment" in df.columns


def test_dataset_iris():
    df = pd.read_csv(_p("data", "iris.csv"))
    assert df.shape == (150, 5), "iris.csv devrait faire 150x5"
    counts = df["species"].value_counts().to_dict()
    assert all(counts[s] == 50 for s in ["setosa", "versicolor", "virginica"])


def test_dataset_house():
    df = pd.read_csv(_p("data", "house_prediction_data_set.csv"),
                     sep=r"\s+", header=None)
    assert df.shape[0] == 506, "house dataset devrait avoir 506 lignes"
    assert df.shape[1] == 14, "house dataset devrait avoir 14 colonnes"


def test_dataset_churn_splits():
    for name in ["churn-bigml-80.csv", "churn-bigml-20.csv"]:
        assert os.path.exists(_p("data", name)), f"dataset absent: {name}"


# -----------------------------------------------------------------------------
# 3) LEVEL 1
# -----------------------------------------------------------------------------
def test_level1_cleaning_outputs(run_all_scripts):
    _check_returncode(run_all_scripts["level1/task2_cleaning.py"],
                      "task2_cleaning")
    path = _p("level1", "cleaned_sentiment_data.csv")
    assert os.path.exists(path), f"cleaned CSV absent: {path}"
    df = pd.read_csv(path)

    assert df.shape[0] == 732, "le dataset nettoye doit garder 732 lignes"
    required = {"Text", "User", "Hashtags", "Country_encoded",
                "Sentiment_encoded", "Retweets_scaled", "Likes_scaled",
                "Year", "Month", "Day", "Hour"}
    assert required.issubset(df.columns), (
        f"colonnes manquantes: {required - set(df.columns)}")
    assert any(c.startswith("Platform_") for c in df.columns), \
        "encodage one-hot Platform attendu"
    assert df.isna().sum().sum() == 0, "le dataset nettoye contient des NaN"
    assert df["Sentiment_encoded"].dtype.kind in "iu", \
        "Sentiment_encoded doit etre entier"

    # Retweets/Likes standardises -> moyenne ~0, ecart-type ~1
    for col in ["Retweets_scaled", "Likes_scaled"]:
        assert abs(df[col].mean()) < 0.05, f"{col} moyenne ~0 attendue"
        assert 0.9 < df[col].std() < 1.1, f"{col} std ~1 attendu"


def test_level1_eda_outputs(run_all_scripts):
    _check_returncode(run_all_scripts["level1/task3_eda.py"], "task3_eda")
    _check_figures("level1/figures", LEVEL1_FIGURES)

    stats = pd.read_csv(_p("level1", "iris_summary_stats.csv"),
                        index_col=0, header=[0, 1])
    assert stats.shape == (3, 20), "3 especes x 4 features x 5 stats attendues"
    assert list(stats.index) == ["setosa", "versicolor", "virginica"]
    assert stats.notna().all().all(), "stats manquantes dans iris_summary_stats"

    summary_path = _p("level1", "eda_summary.md")
    assert os.path.exists(summary_path), "eda_summary.md absent"
    with open(summary_path, encoding="utf-8") as f:
        content = f.read()
    assert content.strip().startswith("#"), "eda_summary.md doit etre du markdown"


# -----------------------------------------------------------------------------
# 4) LEVEL 2
# -----------------------------------------------------------------------------
def test_level2_regression_outputs(run_all_scripts):
    _check_returncode(run_all_scripts["level2/task1_regression.py"],
                      "task1_regression")
    _check_figures("level2/figures", LEVEL2_FIGURES[:1])

    df = _csv("level2/regression_results.csv")
    assert len(df) == 3, "3 modeles attendus (LR / DT / RF)"
    expected = {"Model", "Test MSE", "Test R2", "Train MSE", "Train R2"}
    assert expected.issubset(df.columns), f"colonnes manquantes: {expected}"
    assert set(df["Model"]) == {"Linear Regression", "Decision Tree",
                                "Random Forest"}
    assert (df["Test MSE"] > 0).all() and (df["Test R2"] > 0).all(), \
        "MSE>0 et R2>0 attendus pour la regression immobiliere"
    # Le meilleur modele (MSE le plus faible) doit etre Random Forest
    best = df.sort_values("Test MSE").iloc[0]["Model"]
    assert best == "Random Forest", f"meilleur modele inattendu: {best}"


def test_level2_classification_outputs(run_all_scripts):
    _check_returncode(run_all_scripts["level2/task2_classification.py"],
                      "task2_classification")
    _check_figures("level2/figures", LEVEL2_FIGURES)

    df = _csv("level2/classification_results.csv")
    assert len(df) == 2, "2 modeles attendus (LogReg / RF)"
    expected = {"Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"}
    assert expected.issubset(df.columns), f"colonnes manquantes: {expected}"
    assert set(df["Model"]) == {"Logistic Regression", "Random Forest"}
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    assert df[metric_cols].to_numpy().min() >= 0.0, "metriques >= 0 attendues"
    assert df[metric_cols].to_numpy().max() <= 1.0, "metriques <= 1 attendues"
    # Le meilleur modele doit etre Random Forest (accuracy la plus haute)
    best = df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    assert best == "Random Forest", f"meilleur modele inattendu: {best}"


# -----------------------------------------------------------------------------
# 5) LEVEL 3
# -----------------------------------------------------------------------------
def test_level3_nlp_outputs(run_all_scripts):
    _check_returncode(run_all_scripts["level3/task2_nlp_classification.py"],
                      "task2_nlp_classification")
    _check_figures("level3/figures",
                   ["confusion_matrix_naive.png", "confusion_matrix_logistic.png"])

    df = _csv("level3/nlp_results.csv")
    assert len(df) == 2, "2 modeles attendus (Naive Bayes / LogReg)"
    assert set(df["Model"]) == {"Naive Bayes (Multinomial)",
                                "Logistic Regression"}
    for col in ["Accuracy", "F1 Macro", "F1 Weighted"]:
        assert df[col].between(0.0, 1.0).all(), f"metrique {col} hors [0,1]"
    # L'evaluation doit etre meilleure que le hasard (>0.5 sur ce jeu)
    assert df["Accuracy"].min() > 0.5, "accuracy anormalement basse"


def test_level3_nn_outputs(run_all_scripts):
    _check_returncode(run_all_scripts["level3/task3_neural_network.py"],
                      "task3_neural_network")
    _check_figures("level3/figures",
                   ["nn_training_curves.png", "nn_hyperparameter_tuning.png",
                    "nn_confusion_matrix.png"])

    df = _csv("level3/nn_results.csv")
    assert len(df) == 6, "6 configurations de tuning attendues"
    assert "Config" in df.columns and "Val Accuracy" in df.columns
    assert df["Val Accuracy"].between(0.0, 1.0).all()
    assert (df["Epochs"] > 0).all(), "epochs doivent etre positifs"
