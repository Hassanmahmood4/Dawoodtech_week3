from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET,
    FIGURE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    REGRESSION_FEATURES,
    REGRESSION_TARGET,
    REPORT_DIR,
    TEST_SIZE,
)
from src.data_preparation import prepare_all_datasets
from src.visualization import (
    plot_classification_metrics,
    plot_confusion_matrix,
    plot_regression_metrics,
    plot_regression_predictions,
)


def _ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _regression_models() -> dict[str, Pipeline]:
    linear_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    return {
        "Linear Regression": linear_pipeline,
        "Random Forest Regressor": forest_pipeline,
    }


def _classification_models() -> dict[str, Pipeline]:
    logistic_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_pipeline,
        "Random Forest Classifier": forest_pipeline,
    }


def train_regression_model(regression_df: pd.DataFrame) -> dict[str, Any]:
    x = regression_df[REGRESSION_FEATURES]
    y = regression_df[REGRESSION_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in _regression_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        predictions_by_model[model_name] = predictions
        trained_models[model_name] = model

        mse = mean_squared_error(y_test, predictions)
        results.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_test, predictions),
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2": r2_score(y_test, predictions),
            }
        )

    metrics_df = pd.DataFrame(results).sort_values("rmse", ascending=True)
    best_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    best_predictions = predictions_by_model[best_name]

    joblib.dump(best_model, MODEL_DIR / "regression_model.joblib")
    metrics_df.to_csv(REPORT_DIR / "regression_metrics.csv", index=False)
    plot_regression_metrics(metrics_df, FIGURE_DIR / "regression_metrics.png")
    plot_regression_predictions(
        y_test,
        best_predictions,
        FIGURE_DIR / "regression_predictions.png",
        best_name,
    )

    return {
        "best_model_name": best_name,
        "metrics": metrics_df.to_dict(orient="records"),
        "feature_defaults": x.median().to_dict(),
    }


def train_classification_model(classification_df: pd.DataFrame) -> dict[str, Any]:
    x = classification_df[CLASSIFICATION_FEATURES]
    y = classification_df[CLASSIFICATION_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in _classification_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        predictions_by_model[model_name] = predictions
        trained_models[model_name] = model

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, pos_label="malignant", zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, pos_label="malignant", zero_division=0
                ),
            }
        )

    metrics_df = pd.DataFrame(results).sort_values(
        ["recall", "precision", "accuracy"], ascending=False
    )
    best_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    best_predictions = predictions_by_model[best_name]

    joblib.dump(best_model, MODEL_DIR / "classification_model.joblib")
    metrics_df.to_csv(REPORT_DIR / "classification_metrics.csv", index=False)
    plot_classification_metrics(
        metrics_df, FIGURE_DIR / "classification_metrics.png"
    )
    plot_confusion_matrix(
        confusion_matrix(y_test, best_predictions, labels=["malignant", "benign"]),
        ["malignant", "benign"],
        FIGURE_DIR / "classification_confusion_matrix.png",
        best_name,
    )

    return {
        "best_model_name": best_name,
        "metrics": metrics_df.to_dict(orient="records"),
        "feature_defaults": x.median().to_dict(),
    }


def train_all_models() -> dict[str, Any]:
    _ensure_directories()
    regression_df, classification_df = prepare_all_datasets()

    metadata = {
        "project": "Week 3 ML Prediction System",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "regression": train_regression_model(regression_df),
        "classification": train_classification_model(classification_df),
    }

    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


if __name__ == "__main__":
    training_metadata = train_all_models()
    print(json.dumps(training_metadata, indent=2))
