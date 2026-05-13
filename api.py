"""Real-time churn scoring API."""
  import joblib, numpy as np
  from fastapi import FastAPI
  from pydantic import BaseModel

  app = FastAPI(title="Churn Prediction API", version="1.0.0")
  model = joblib.load("churn_model.joblib")
  scaler = joblib.load("scaler.joblib")

  class CustomerFeatures(BaseModel):
      features: list[float]

  @app.post("/predict")
  def predict(data: CustomerFeatures):
      x = scaler.transform([data.features])
      proba = model.predict_proba(x)[0][1]
      return {"churn_probability": round(float(proba), 4), "will_churn": proba > 0.5}

  @app.get("/health")
  def health(): return {"status": "online"}
  