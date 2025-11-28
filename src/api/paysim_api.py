# ============================================================
# Adaptive Risk Twin – Fraud Detection API (FINAL VERSION)
# Includes:
#   ✔ Prediction
#   ✔ Simulation
#   ✔ SHAP Explainability
#   ✔ 15-feature alignment to training dataset
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import os

# ---------------------- CONFIG ----------------------
MODEL_PATH = "models/lgbm_fraud.txt"
CI_MODE = os.getenv("CI", "false").lower() == "true"

model = None
explainer = None

# EXACT 15 FEATURES USED DURING TRAINING
ALL_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "step",
    "balance_drop_ratio_orig",
    "balance_jump_ratio_dest",
    "zero_to_zero_dest",
    "unchanged_orig_balance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

# ---------------------- CREATE APP ----------------------
app = FastAPI(
    title="Adaptive Risk Twin – Fraud API",
    description="Fraud prediction + simulation + explainability",
    version="1.0.0"
)

# ---------------------- LOAD MODEL ----------------------
def load_dependencies():
    global model, explainer

    if CI_MODE:
        print("CI MODE ENABLED — skipping model loading.")
        return

    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = lgb.Booster(model_file=MODEL_PATH)

    # SHAP Explainer
    print("Initializing SHAP TreeExplainer…")
    explainer = shap.TreeExplainer(model)

    print("Loaded schema columns:", ALL_COLUMNS)
    print("SHAP explainer ready.")


# ---------------------- INPUT MODELS ----------------------
class TransactionInput(BaseModel):
    amount: float
    type: str
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    step: int


class SimulationInput(BaseModel):
    transaction: TransactionInput
    amount_shock_percent: float = 0
    force_type: str | None = None


class SHAPInput(BaseModel):
    amount: float
    type: str
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    step: int


class FraudPrediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    message: str


# ---------------------- FEATURE BUILDING ----------------------
def build_feature_row(tx: TransactionInput) -> pd.DataFrame:

    df = pd.DataFrame([{
        "amount": tx.amount,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "step": tx.step,
        "type": tx.type.upper()
    }])

    # EXACT engineered features used during training
    df["balance_drop_ratio_orig"] = (
        (df["oldbalanceOrg"] - df["newbalanceOrig"]) / (df["oldbalanceOrg"] + 1)
    )

    df["balance_jump_ratio_dest"] = (
        (df["newbalanceDest"] - df["oldbalanceDest"]) / (df["oldbalanceDest"] + 1)
    )

    df["zero_to_zero_dest"] = (
        (df["oldbalanceDest"] == 0) &
        (df["newbalanceDest"] == 0) &
        (df["amount"] > 0)
    ).astype(int)

    df["unchanged_orig_balance"] = (
        (df["oldbalanceOrg"] == df["newbalanceOrig"]) &
        (df["amount"] > 0)
    ).astype(int)

    # One-hot encode TYPE
    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=False)

    # Ensure all required columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    df = df[ALL_COLUMNS]

    # Clean
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


# ---------------------- ENDPOINT: HEALTH ----------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features": len(ALL_COLUMNS)
    }


# ---------------------- ENDPOINT: PREDICT ----------------------
@app.post("/predict_fraud", response_model=FraudPrediction)
def predict_fraud(tx: TransactionInput):

    X = build_feature_row(tx)

    if CI_MODE:
        prob = 0.1234
    else:
        prob = float(model.predict(X)[0])

    threshold = 0.5
    is_fraud = prob >= threshold
    msg = "High fraud risk" if is_fraud else "Low fraud risk"

    return FraudPrediction(
        fraud_probability=prob,
        is_fraud=is_fraud,
        threshold=threshold,
        message=msg
    )


# ---------------------- ENDPOINT: SIMULATION ----------------------
@app.post("/simulate")
def simulate(sim: SimulationInput):

    # Baseline prediction
    baseline_X = build_feature_row(sim.transaction)
    baseline_prob = float(model.predict(baseline_X)[0])

    # Shocked transaction
    shocked_tx = sim.transaction.model_copy()

    # Apply amount shock
    shocked_tx.amount = shocked_tx.amount * (1 + sim.amount_shock_percent / 100)

    # Optional forced type
    if sim.force_type:
        shocked_tx.type = sim.force_type

    # Predict simulated
    shock_X = build_feature_row(shocked_tx)
    shock_prob = float(model.predict(shock_X)[0])

    return {
        "baseline_fraud_probability": baseline_prob,
        "simulated_fraud_probability": shock_prob,
        "delta": shock_prob - baseline_prob,
        "applied_shock_percent": sim.amount_shock_percent,
        "forced_type": sim.force_type
    }


# ---------------------- ENDPOINT: SHAP ----------------------
@app.post("/explain_shap")
def explain_shap(tx: SHAPInput):

    X = build_feature_row(tx)

    # Prediction
    prob = float(model.predict(X)[0])

    # SHAP values
    shap_vals = explainer.shap_values(X)[0].tolist()

    contributions = list(zip(ALL_COLUMNS, shap_vals))
    contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    return {
        "fraud_probability": prob,
        "base_value": explainer.expected_value,
        "shap_values": shap_vals,
        "feature_names": ALL_COLUMNS,
        "top_features": contributions_sorted[:10]
    }


# ---------------------- INITIALIZE ----------------------
load_dependencies()
