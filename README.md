# Heart Disease Prediction using XGBoost

This project predicts the likelihood of heart disease using patient data. It uses XGBoost, cross-validation, and evaluation metrics including ROC-AUC and feature importance.

## Dataset
- The dataset used is `heart.csv` inside `archive.zip`.

## Features
- Categorical features are one-hot encoded.
- Numerical features are scaled using StandardScaler.

## Model
- XGBoost classifier with tuned hyperparameters.
- Evaluation includes:
  - Confusion Matrix
  - Classification Report
  - ROC-AUC score and curve
  - Feature importance visualization

## How to Run
1. Open `heart_disease_xgboost.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `archive.zip` when prompted.
3. Run all notebook cells sequentially.
