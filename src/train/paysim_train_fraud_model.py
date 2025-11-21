"""
Train Fraud Detection Models for PaySim Dataset
Models:
- Logistic Regression (baseline)
- LightGBM (primary)
- XGBoost (optional)
Handles:
- Train/Test split
- Class imbalance (class weights + SMOTE)
- Metrics + confusion matrix
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
import joblib
import lightgbm as lgb
from xgboost import XGBClassifier

# -------------------------------------------------------
# 1. Load cleaned dataset
# -------------------------------------------------------

DATA_PATH = os.path.join("data", "processed", "paysim_cleaned_core_features.csv")
df = pd.read_csv(DATA_PATH)

print(f"\nLoaded dataset: {df.shape}")

# -------------------------------------------------------
# 2. Split into X and y
# -------------------------------------------------------

y = df["isFraud"]
X = df.drop("isFraud", axis=1)

# -------------------------------------------------------
# 3. Train/test split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------------------------------
# 4. Handle imbalance — SMOTE oversampling on training only
# -------------------------------------------------------

sm = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"After SMOTE: {X_train_res.shape}")

# -------------------------------------------------------
# 5. Standardize numeric features
# -------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# -------------------------------------------------------
# 6. Logistic Regression (Baseline)
# -------------------------------------------------------

log_reg = LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1)

log_reg.fit(X_train_scaled, y_train_res)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

# Save baseline model
joblib.dump(log_reg, "models/log_reg_baseline.pkl")

# -------------------------------------------------------
# 7. LightGBM (Primary Model)
# -------------------------------------------------------

lgb_train = lgb.Dataset(X_train_res, label=y_train_res)
lgb_eval = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "binary",
    "metric": "auc",
    "is_unbalance": True,  # imbalance handling
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.7,
}

lgb_model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    num_boost_round=300,
)

# Predictions
y_proba_lgb = lgb_model.predict(X_test)
y_pred_lgb = (y_proba_lgb > 0.5).astype(int)

print("\n--- LightGBM Report ---")
print(classification_report(y_test, y_pred_lgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lgb))

# Save LGBM model
lgb_model.save_model("models/lgbm_fraud.txt")

# -------------------------------------------------------
# 8. XGBoost (Optional)
# -------------------------------------------------------

xgb_model = XGBClassifier(
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    learning_rate=0.05,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_proba_xgb > 0.5).astype(int)

print("\n--- XGBoost Report ---")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))

# Save XGB model
joblib.dump(xgb_model, "models/xgb_fraud.pkl")

print("\nTraining complete! Models saved in /models/")
