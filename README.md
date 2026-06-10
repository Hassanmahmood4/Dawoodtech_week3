# Week 3 ML Prediction System

An end-to-end machine learning project for the Week 3 internship task. The project trains supervised ML models, evaluates performance, saves trained artifacts, and provides a Streamlit app for instant predictions.

## Project Overview

This repository includes two supervised learning workflows:

- Regression: predict diabetes disease progression using medical indicators.
- Classification: predict whether a breast cancer diagnosis is benign or malignant using selected tumor measurements.

The implementation uses scikit-learn pipelines so preprocessing happens consistently during training and prediction.

## Features

- Dataset preparation and CSV export
- Missing value handling with median imputation
- Feature selection for regression and classification
- Data scaling for linear models
- Train/test split
- Linear Regression and Random Forest Regressor
- Logistic Regression and Random Forest Classifier
- Regression metrics: MAE, MSE, RMSE, R-squared
- Classification metrics: accuracy, precision, recall, confusion matrix
- Saved models with joblib
- Streamlit app with dataset overview, prediction forms, metrics, and charts
- Downloadable prediction records

## Project Structure

```text
Week3-ML-Prediction-System/
├── app.py
├── data/
│   ├── breast_cancer_classification.csv
│   └── diabetes_regression.csv
├── models/
│   ├── classification_model.joblib
│   ├── metadata.json
│   └── regression_model.joblib
├── notebook/
│   └── model_training.ipynb
├── reports/
│   ├── classification_metrics.csv
│   ├── regression_metrics.csv
│   └── figures/
├── screenshots/
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── train_models.py
│   └── visualization.py
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models

Run the full training workflow:

```bash
python3 -m src.train_models
```

This command will:

- Export datasets into `data/`
- Train regression and classification models
- Save model files into `models/`
- Save metric CSV files into `reports/`
- Save performance charts into `reports/figures/`

## Run Streamlit App

```bash
streamlit run app.py
```

The app includes:

- Dataset overview
- Regression prediction form
- Classification prediction form
- Model performance visualizations
- Prediction result downloads

## Datasets

The project uses stable datasets available through scikit-learn:

- Diabetes dataset for regression
- Breast cancer Wisconsin dataset for classification

Both datasets are loaded locally from scikit-learn and exported to CSV so the repository includes dataset files after training.

## Model Summary

Regression models compared:

- Linear Regression
- Random Forest Regressor

Classification models compared:

- Logistic Regression
- Random Forest Classifier

The best regression model is selected by lowest RMSE. The best classification model is selected by recall, then precision, then accuracy. Recall is prioritized because missing malignant cases is the riskiest classification error in this dataset.

Current results:

- Regression best model: Linear Regression with RMSE 53.85 and MAE 42.79.
- Classification best model: Logistic Regression with accuracy 96.49%, precision 95.24%, and recall 95.24%.

## Deliverables Checklist

- Machine learning notebook: `notebook/model_training.ipynb`
- Trained model files: `models/`
- Streamlit prediction app: `app.py`
- Dataset files: `data/`
- Documentation: `README.md`
- Performance charts: `reports/figures/`
- Screenshots: `screenshots/`

## Git Commit Plan

After the project is complete, use clean logical commits:

1. `chore: initialize project structure and dependencies`
2. `feat: add dataset preparation workflow`
3. `feat: implement model training pipelines`
4. `feat: add model evaluation visualizations`
5. `feat: build streamlit prediction app`
6. `docs: add notebook and project documentation`
7. `docs: add screenshots for final submission`
