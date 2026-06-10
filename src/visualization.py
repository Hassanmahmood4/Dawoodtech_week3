from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="crest")


def _save_current_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_regression_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = metrics_df.melt(
        id_vars="model",
        value_vars=["mae", "rmse"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.title("Regression Model Error Comparison")
    plt.xlabel("Model")
    plt.ylabel("Error score")
    _save_current_figure(output_path)


def plot_regression_predictions(
    actual: pd.Series, predictions, output_path: Path, model_name: str
) -> None:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=actual, y=predictions, s=56, color="#1f7a68")
    min_value = min(actual.min(), predictions.min())
    max_value = max(actual.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#334155")
    plt.title(f"Actual vs Predicted Values ({model_name})")
    plt.xlabel("Actual disease progression")
    plt.ylabel("Predicted disease progression")
    _save_current_figure(output_path)


def plot_classification_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.title("Classification Model Metric Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    _save_current_figure(output_path)


def plot_confusion_matrix(
    matrix, labels: list[str], output_path: Path, model_name: str
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    _save_current_figure(output_path)
