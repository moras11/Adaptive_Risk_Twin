"""
FastAPI service for PaySim fraud detection (Adaptive Risk Twin – Fraud Module)

Endpoints:
- GET /health         : health check
- POST /predict_fraud : predict fraud probability for a single transaction (raw input)
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# ---------------------------------------------------
# 1. FastAPI app
# ---------------------------------------------------

app = FastAPI(
    title="Adaptive Risk Twin – Fraud API",
    description="Fraud prediction service built on PaySim + LightGBM",
    version="0.1.0",
)

# ---------------------------------------------------
# 2. Load model + feature schema at startup
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # go up from src/api/
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_fraud.txt")
SCHEMA_PATH = os.path.join(BASE_DIR, "data", "processed", "paysim_cleaned_core_features.csv")

print(f"Loading model from: {MODEL_PATH}")
model = lgb.Booster(model_file=MODEL_PATH)

print(f"Loading feature schema from: {SCHEMA_PATH}")
schema_df = pd.read_csv(SCHEMA_PATH, nrows=5)  # just to get columns
ALL_COLUMNS = [c for c in schema_df.columns if c != "isFraud"]  # features only

# ---------------------------------------------------
# 3. Pydantic request/response models
# ---------------------------------------------------

class TransactionInput(BaseModel):
    amount: float = Field(..., example=50000)
    type: str = Field(..., example="TRANSFER")
    oldbalanceOrg: float = Field(..., example=100000)
    newbalanceOrig: float = Field(..., example=50000)
    oldbalanceDest: float = Field(..., example=0)
    newbalanceDest: float = Field(..., example=50000)
    step: int = Field(..., example=245)


class FraudPrediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float = 0.5
    message: str


# ---------------------------------------------------
# 4. Helper: build model-ready dataframe from raw input
# ---------------------------------------------------

def build_feature_row(tx: TransactionInput) -> pd.DataFrame:
    """
    Convert raw transaction input into a single-row DataFrame
    with all engineered features and one-hot encoded type,
    aligned to training schema (ALL_COLUMNS).
    """
    # 4.1 raw base fields
    row = {
        "amount": tx.amount,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "step": tx.step,
        "type": tx.type.upper(),  # normalize
    }

    df = pd.DataFrame([row])

    # 4.2 engineered fraud signatures (same logic as training)
    df["balance_drop_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_drop_ratio_orig"] = df["balance_drop_orig"] / (df["oldbalanceOrg"] + 1)

    df["balance_jump_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["balance_jump_ratio_dest"] = df["balance_jump_dest"] / (df["oldbalanceDest"] + 1)

    df["zero_to_zero_dest"] = (
        (df["oldbalanceDest"] == 0)
        & (df["newbalanceDest"] == 0)
        & (df["amount"] > 0)
    ).astype(int)

    df["unchanged_orig_balance"] = (
        (df["oldbalanceOrg"] == df["newbalanceOrig"])
        & (df["amount"] > 0)
    ).astype(int)

    # 4.3 one-hot encode type
    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=False)

    # 4.4 align with training feature columns
    #    Any missing columns -> 0, any extra -> dropped
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # keep only training columns, in correct order
    df = df[ALL_COLUMNS]

    # replace inf/nan (consistent with training)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


# ---------------------------------------------------
# 5. Endpoints
# ---------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "features": len(ALL_COLUMNS)}


@app.post("/predict_fraud", response_model=FraudPrediction)
def predict_fraud(tx: TransactionInput):
    """
    Predict fraud probability for a single transaction.
    """
    # build feature row
    X = build_feature_row(tx)

    # LightGBM was trained on raw (unscaled) features
    prob = float(model.predict(X)[0])

    threshold = 0.5  # you can tune this later
    is_fraud = prob >= threshold

    if is_fraud:
        msg = "High fraud risk – transaction should be flagged/reviewed."
    else:
        msg = "Low fraud risk – transaction appears normal."

    return FraudPrediction(
        fraud_probability=prob,
        is_fraud=is_fraud,
        threshold=threshold,
        message=msg,
    )

from typing import Optional

class SimulationInput(BaseModel):
    transaction: TransactionInput
    amount_shock_percent: Optional[float] = Field(0, example=20)
    force_type: Optional[str] = Field(None, example="CASH_OUT")

@app.post("/simulate", response_model=dict)
def simulate_risk(sim: SimulationInput):
    """
    Run a simple risk simulation:
    - get baseline fraud probability
    - apply shocks (amount %, type override)
    - recompute fraud probability
    - return delta
    """
    # ---- 1) BASELINE ----
    baseline_X = build_feature_row(sim.transaction)
    baseline_prob = float(model.predict(baseline_X)[0])

    # ---- 2) APPLY SHOCKS ----
    shocked_tx = sim.transaction.model_copy()

    # amount shock
    shocked_tx.amount = shocked_tx.amount * (1 + sim.amount_shock_percent / 100)

    # type override
    if sim.force_type is not None:
        shocked_tx.type = sim.force_type.upper()

    # ---- 3) SIMULATED ----
    sim_X = build_feature_row(shocked_tx)
    simulated_prob = float(model.predict(sim_X)[0])

    # ---- 4) DELTA ----
    delta = simulated_prob - baseline_prob

    return {
        "baseline_fraud_probability": baseline_prob,
        "simulated_fraud_probability": simulated_prob,
        "delta": delta
    }
