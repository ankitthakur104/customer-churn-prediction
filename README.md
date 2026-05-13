# Customer Churn Prediction System

End-to-end ML churn prediction with XGBoost/LightGBM ensemble, data drift detection, and FastAPI real-time scoring API.

## Features
- 40+ engineered behavioral and engagement features
- XGBoost + LightGBM ensemble with stacking
- Automated retraining triggered by statistical drift detection (KS test, PSI)
- Real-time scoring API via FastAPI
- SHAP explainability for predictions
- Docker + CI/CD deployment

## Metrics
- F1 Score: 0.89
- AUC-ROC: 0.93
- Automated drift detection in production

## Stack
Python · XGBoost · LightGBM · Scikit-learn · FastAPI · Docker · SHAP

## Setup
```bash
pip install -r requirements.txt
python train.py          # Train and save models
uvicorn api:app --reload # Start scoring API
```
