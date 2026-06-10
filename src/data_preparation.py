from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from src.config import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET,
    DATA_DIR,
    REGRESSION_FEATURES,
    REGRESSION_TARGET,
)


def prepare_regression_dataset() -> pd.DataFrame:
    """Load and save the diabetes regression dataset."""
    dataset = load_diabetes(as_frame=True)
    frame = dataset.frame.copy()
    frame = frame.rename(columns={"target": REGRESSION_TARGET})
    frame = frame[REGRESSION_FEATURES + [REGRESSION_TARGET]]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / "diabetes_regression.csv", index=False)
    return frame


def prepare_classification_dataset() -> pd.DataFrame:
    """Load and save the breast cancer classification dataset."""
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame[CLASSIFICATION_TARGET] = dataset.target
    frame[CLASSIFICATION_TARGET] = frame[CLASSIFICATION_TARGET].map(
        {0: "malignant", 1: "benign"}
    )
    frame = frame[CLASSIFICATION_FEATURES + [CLASSIFICATION_TARGET]]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / "breast_cancer_classification.csv", index=False)
    return frame


def prepare_all_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    regression_df = prepare_regression_dataset()
    classification_df = prepare_classification_dataset()
    return regression_df, classification_df


if __name__ == "__main__":
    prepare_all_datasets()
    print("Datasets saved to data/.")
