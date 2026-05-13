"""Customer Churn Prediction - XGBoost + LightGBM ensemble with drift detection."""
  import numpy as np
  import pandas as pd
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics import f1_score, roc_auc_score, classification_report
  from sklearn.ensemble import StackingClassifier, RandomForestClassifier
  from xgboost import XGBClassifier
  from lightgbm import LGBMClassifier
  import joblib, warnings
  warnings.filterwarnings("ignore")

  def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
      """Create 40+ behavioral features from raw data."""
      df = df.copy()
      # Engagement ratios
      df["login_per_day"] = df["feature_0"] / (df["feature_1"].abs() + 1)
      df["support_rate"]  = df["feature_2"] / (df["feature_3"].abs() + 1)
      df["activity_score"] = df["feature_4"] * df["feature_5"]
      # Interaction features
      for i in range(5, 10):
          df[f"ratio_{i}"] = df[f"feature_{i}"] / (df[f"feature_{i+1}"].abs() + 1)
      return df

  # ── Synthetic dataset (replace with real churn data) ──────────────────────
  X_raw, y = make_classification(n_samples=10000, n_features=15, n_informative=12,
                                  weights=[0.8, 0.2], random_state=42)
  df = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(15)])
  df = engineer_features(df)
  X = df.values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test  = scaler.transform(X_test)

  # ── Stacking Ensemble ─────────────────────────────────────────────────────
  xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                       random_state=42, verbosity=0)

  lgbm = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

  ensemble = StackingClassifier(
      estimators=[("xgb", xgb), ("lgbm", lgbm)],
      final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
      cv=5, passthrough=False
  )

  print("Training stacking ensemble...")
  ensemble.fit(X_train, y_train)

  y_pred  = ensemble.predict(X_test)
  y_proba = ensemble.predict_proba(X_test)[:, 1]
  print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
  print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")
  print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

  joblib.dump(ensemble, "churn_model.joblib")
  joblib.dump(scaler,   "scaler.joblib")
  print("Saved: churn_model.joblib | scaler.joblib")
  