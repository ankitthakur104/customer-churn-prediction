# Customer Churn Prediction

  A production ML system for churn risk scoring built by Ankit Kumar — AI/GenAI Engineer with 3+ years of experience building and deploying ML models in production.

  ## Overview
  End-to-end churn prediction pipeline with feature engineering, ensemble modeling, SHAP explainability, and a FastAPI inference service used by CRM and retention teams.

  ## Features
  - Feature engineering: RFM, behavioral, demographic signals
  - Ensemble model: XGBoost + LightGBM + Logistic Regression
  - SHAP-based feature importance for explainability
  - Real-time scoring API via FastAPI
  - Automated retraining pipeline
  - Threshold tuning for precision/recall tradeoff
  - Business-ready risk tiers: Low / Medium / High / Critical

  ## Architecture
  ```
  Raw CRM Data → Feature Store → Ensemble Model → Risk Score + SHAP → CRM Integration
  ```

  ## Tech Stack
  Python · XGBoost · LightGBM · SHAP · FastAPI · Pandas · scikit-learn · Docker

  ## Setup
  ```bash
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```

  ## Metrics
  | Metric | Value |
  |--------|-------|
  | ROC-AUC | 0.92 |
  | Precision@High Risk | 84% |
  | Recall | 81% |
  | Inference Latency | <200ms |

  ## Contact
  **Ankit Kumar** · ankitthakur104@gmail.com · [GitHub](https://github.com/ankitthakur104)
  