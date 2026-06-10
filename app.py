from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CLASSIFICATION_FEATURES,
    DATA_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    REGRESSION_FEATURES,
    REPORT_DIR,
)
from src.train_models import train_all_models


st.set_page_config(
    page_title="Week 3 ML Prediction System",
    page_icon="ML",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: oklch(0.985 0.006 180);
            --ink: oklch(0.235 0.025 220);
            --muted: oklch(0.48 0.025 220);
            --accent: oklch(0.58 0.115 170);
            --accent-soft: oklch(0.93 0.035 170);
            --line: oklch(0.86 0.012 210);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, oklch(0.94 0.04 170), transparent 34rem),
                var(--surface);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        p, li, label, [data-testid="stMarkdownContainer"] {
            line-height: 1.5;
        }

        .hero {
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 2.4rem;
            background: linear-gradient(135deg, oklch(0.98 0.008 180), oklch(0.94 0.025 175));
            box-shadow: 0 24px 70px oklch(0.62 0.03 220 / 0.18);
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .subcopy {
            color: var(--muted);
            max-width: 68ch;
            font-size: 1.05rem;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            background: oklch(0.995 0.003 180 / 0.78);
            min-height: 132px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.45rem;
        }

        .metric-card h3 {
            margin: 0;
            min-height: 3.25rem;
            display: flex;
            align-items: center;
            color: var(--ink);
            font-size: clamp(1.35rem, 2vw, 1.75rem);
            line-height: 1.15;
        }

        div[data-testid="stTabs"] button {
            color: var(--ink);
            font-weight: 650;
            opacity: 1;
        }

        div[data-testid="stTabs"] button:hover {
            color: var(--accent);
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent);
        }

        div[data-testid="stWidgetLabel"] p {
            color: var(--ink);
            font-weight: 650;
        }

        .form-section-title {
            margin: 1.2rem 0 0.4rem;
            color: var(--ink);
            font-size: 1rem;
            font-weight: 750;
            letter-spacing: -0.01em;
        }

        .form-section-copy {
            margin: -0.1rem 0 0.85rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .field-helper {
            margin: 0.25rem 0 0.25rem;
        }

        .field-helper strong {
            display: block;
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.25;
        }

        .field-helper span {
            display: block;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
            max-width: 52ch;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        div[data-testid="stAlert"] p {
            color: oklch(0.24 0.07 155);
            font-weight: 700;
        }

        div[data-testid="stAlert"][data-baseweb="notification"] {
            background-color: oklch(0.92 0.055 150);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts() -> tuple[object, object, dict]:
    metadata_path = MODEL_DIR / "metadata.json"
    regression_path = MODEL_DIR / "regression_model.joblib"
    classification_path = MODEL_DIR / "classification_model.joblib"

    if not metadata_path.exists() or not regression_path.exists() or not classification_path.exists():
        train_all_models()

    regression_model = joblib.load(regression_path)
    classification_model = joblib.load(classification_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return regression_model, classification_model, metadata


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_df = pd.read_csv(DATA_DIR / "diabetes_regression.csv")
    classification_df = pd.read_csv(DATA_DIR / "breast_cancer_classification.csv")
    regression_metrics = pd.read_csv(REPORT_DIR / "regression_metrics.csv")
    classification_metrics = pd.read_csv(REPORT_DIR / "classification_metrics.csv")
    return regression_df, classification_df, regression_metrics, classification_metrics


def feature_label(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


REGRESSION_LABELS = {
    "age": "Age Index",
    "sex": "Sex Index",
    "bmi": "Body Mass Index",
    "bp": "Blood Pressure",
    "s1": "Serum Measure 1",
    "s2": "Serum Measure 2",
    "s3": "Serum Measure 3",
    "s4": "Serum Measure 4",
    "s5": "Serum Measure 5",
    "s6": "Serum Measure 6",
}


REGRESSION_DESCRIPTIONS = {
    "age": "Normalized patient age used as a general health indicator.",
    "sex": "Encoded biological sex value from the diabetes dataset.",
    "bmi": "Body mass index value, one of the strongest progression signals.",
    "bp": "Average blood pressure measurement.",
    "s1": "First normalized blood serum measurement.",
    "s2": "Second normalized blood serum measurement.",
    "s3": "Third normalized blood serum measurement.",
    "s4": "Fourth normalized blood serum measurement.",
    "s5": "Fifth normalized blood serum measurement.",
    "s6": "Sixth normalized blood serum measurement.",
}


def build_regression_form(defaults: dict[str, float]) -> pd.DataFrame:
    st.markdown("### Predict Disease Progression")
    st.caption("Adjust the normalized health indicators and get an instant regression prediction.")
    values: dict[str, float] = {}

    sections = [
        (
            "Clinical Profile",
            "Core patient indicators used by the regression model.",
            ["age", "sex", "bmi", "bp"],
        ),
        (
            "Blood Serum Measurements",
            "Normalized laboratory measures from the diabetes dataset.",
            ["s1", "s2", "s3", "s4", "s5", "s6"],
        ),
    ]

    for title, copy, features in sections:
        st.markdown(
            f'<div class="form-section-title">{title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="form-section-copy">{copy}</div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(2)

        for index, feature in enumerate(features):
            with columns[index % 2]:
                label = REGRESSION_LABELS.get(feature, feature_label(feature))
                st.markdown(
                    f"""
                    <div class="field-helper">
                        <strong>{label}</strong>
                        <span>{REGRESSION_DESCRIPTIONS[feature]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                values[feature] = st.number_input(
                    label,
                    value=float(defaults[feature]),
                    step=0.0001,
                    format="%.4f",
                    key=f"regression_{feature}",
                    help=REGRESSION_DESCRIPTIONS[feature],
                    label_visibility="collapsed",
                )

    return pd.DataFrame([values], columns=REGRESSION_FEATURES)


CLASSIFICATION_LABELS = {
    "mean radius": "Mean Radius",
    "mean texture": "Mean Texture",
    "mean perimeter": "Mean Perimeter",
    "mean area": "Mean Area",
    "mean smoothness": "Mean Smoothness",
    "worst radius": "Worst Radius",
    "worst texture": "Worst Texture",
    "worst perimeter": "Worst Perimeter",
    "worst area": "Worst Area",
    "worst concave points": "Worst Concave Points",
}


CLASSIFICATION_DESCRIPTIONS = {
    "mean radius": "Average distance from the center to the tumor boundary.",
    "mean texture": "Average variation in gray-scale values across the tumor image.",
    "mean perimeter": "Average perimeter length of the tumor boundary.",
    "mean area": "Average tumor area measurement.",
    "mean smoothness": "Average local variation in radius lengths.",
    "worst radius": "Largest radius measurement among the most abnormal cells.",
    "worst texture": "Highest texture measurement among the most abnormal cells.",
    "worst perimeter": "Largest perimeter measurement among the most abnormal cells.",
    "worst area": "Largest area measurement among the most abnormal cells.",
    "worst concave points": "Highest count of concave boundary points.",
}


def build_classification_form(defaults: dict[str, float]) -> pd.DataFrame:
    st.markdown("### Predict Cancer Diagnosis")
    st.caption("Enter tumor measurement values to classify the case as benign or malignant.")
    values: dict[str, float] = {}

    sections = [
        (
            "Mean Tumor Measurements",
            "Average measurements calculated from the cell nuclei image.",
            [
                "mean radius",
                "mean texture",
                "mean perimeter",
                "mean area",
                "mean smoothness",
            ],
        ),
        (
            "Worst-Case Tumor Measurements",
            "Largest or most severe values observed for the selected tumor features.",
            [
                "worst radius",
                "worst texture",
                "worst perimeter",
                "worst area",
                "worst concave points",
            ],
        ),
    ]

    for title, copy, features in sections:
        st.markdown(
            f'<div class="form-section-title">{title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="form-section-copy">{copy}</div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(2)

        for index, feature in enumerate(features):
            with columns[index % 2]:
                label = CLASSIFICATION_LABELS.get(feature, feature.title())
                st.markdown(
                    f"""
                    <div class="field-helper">
                        <strong>{label}</strong>
                        <span>{CLASSIFICATION_DESCRIPTIONS[feature]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                values[feature] = st.number_input(
                    label,
                    value=float(defaults[feature]),
                    min_value=0.0,
                    step=0.0001,
                    format="%.4f",
                    key=f"classification_{feature}",
                    help=CLASSIFICATION_DESCRIPTIONS[feature],
                    label_visibility="collapsed",
                )

    return pd.DataFrame([values], columns=CLASSIFICATION_FEATURES)


def show_metric_strip(metadata: dict) -> None:
    regression_best = metadata["regression"]["best_model_name"]
    classification_best = metadata["classification"]["best_model_name"]
    regression_score = metadata["regression"]["metrics"][0]["rmse"]
    classification_score = metadata["classification"]["metrics"][0]["recall"]

    columns = st.columns(4)
    cards = [
        ("Regression model", regression_best),
        ("Regression RMSE", f"{regression_score:.2f}"),
        ("Classification model", classification_best),
        ("Malignant recall", f"{classification_score:.2%}"),
    ]

    for column, (label, value) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="eyebrow">{label}</div>
                    <h3>{value}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_dataset_overview(regression_df: pd.DataFrame, classification_df: pd.DataFrame) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Regression Dataset")
        st.write("Diabetes progression dataset with ten normalized medical indicators.")
        st.dataframe(regression_df.head(8), use_container_width=True)
        st.write(regression_df.describe().T)

    with right:
        st.subheader("Classification Dataset")
        st.write("Breast cancer Wisconsin dataset using selected tumor measurement features.")
        st.dataframe(classification_df.head(8), use_container_width=True)
        st.write(classification_df["diagnosis"].value_counts().rename("count"))


def show_predictions(regression_model, classification_model, metadata: dict) -> None:
    regression_defaults = metadata["regression"]["feature_defaults"]
    classification_defaults = metadata["classification"]["feature_defaults"]

    regression_tab, classification_tab = st.tabs(
        ["Regression Prediction", "Classification Prediction"]
    )

    with regression_tab:
        regression_input = build_regression_form(regression_defaults)
        if st.button("Predict progression", type="primary"):
            prediction = regression_model.predict(regression_input)[0]
            st.success(f"Predicted disease progression score: {prediction:.2f}")
            st.download_button(
                "Download regression input",
                regression_input.assign(predicted_progression=prediction).to_csv(index=False),
                file_name="regression_prediction.csv",
                mime="text/csv",
            )

    with classification_tab:
        classification_input = build_classification_form(classification_defaults)
        if st.button("Predict diagnosis", type="primary"):
            prediction = classification_model.predict(classification_input)[0]
            st.success(f"Predicted diagnosis: {prediction.title()}")

            if hasattr(classification_model, "predict_proba"):
                probabilities = classification_model.predict_proba(classification_input)[0]
                classes = classification_model.classes_
                probability_df = pd.DataFrame(
                    {"diagnosis": classes, "probability": probabilities}
                )
                st.dataframe(probability_df, use_container_width=True)

            st.download_button(
                "Download classification input",
                classification_input.assign(predicted_diagnosis=prediction).to_csv(index=False),
                file_name="classification_prediction.csv",
                mime="text/csv",
            )


def show_performance(regression_metrics: pd.DataFrame, classification_metrics: pd.DataFrame) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Regression Performance")
        st.dataframe(regression_metrics, use_container_width=True)
        st.image(FIGURE_DIR / "regression_metrics.png")
        st.image(FIGURE_DIR / "regression_predictions.png")

    with right:
        st.subheader("Classification Performance")
        st.dataframe(classification_metrics, use_container_width=True)
        st.image(FIGURE_DIR / "classification_metrics.png")
        st.image(FIGURE_DIR / "classification_confusion_matrix.png")


def main() -> None:
    inject_styles()
    regression_model, classification_model, metadata = load_artifacts()
    regression_df, classification_df, regression_metrics, classification_metrics = load_data()

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Week 3 Machine Learning Project</div>
            <h1>Prediction System for Regression and Classification</h1>
            <p class="subcopy">
                Train, compare, and use supervised machine learning models in one polished app.
                The project includes preprocessing, model evaluation, saved artifacts, and instant predictions.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    show_metric_strip(metadata)
    st.write("")

    overview_tab, prediction_tab, performance_tab, documentation_tab = st.tabs(
        ["Dataset Overview", "Prediction System", "Model Performance", "Documentation"]
    )

    with overview_tab:
        show_dataset_overview(regression_df, classification_df)

    with prediction_tab:
        show_predictions(regression_model, classification_model, metadata)

    with performance_tab:
        show_performance(regression_metrics, classification_metrics)

    with documentation_tab:
        st.subheader("Model Workflow")
        st.markdown(
            """
            1. Load real-world scikit-learn datasets and export them to CSV.
            2. Select useful features for each supervised learning task.
            3. Split the data into training and testing sets.
            4. Impute missing values and scale numeric features where the model needs it.
            5. Train baseline and ensemble models.
            6. Evaluate regression with MAE, MSE, RMSE, and R-squared.
            7. Evaluate classification with accuracy, precision, recall, and confusion matrix.
            8. Save the best models and load them for real-time predictions.
            """
        )


if __name__ == "__main__":
    main()
