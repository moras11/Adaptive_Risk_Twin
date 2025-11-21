"""
Create Cleaned Modeling Dataset for PaySim Fraud Detection
This dataset is used for:
- Baseline ML models (LR, XGBoost, LGBM)
- Simulation engine
- FastAPI inference pipeline
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------
# 1. Load Extended EDA Dataset
# ---------------------------------------

INPUT_PATH = os.path.join("data", "processed", "paysim_eda_extended.csv")
OUTPUT_PATH = os.path.join("data", "processed", "paysim_cleaned_core_features.csv")

print(f"Loading extended dataset from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# ---------------------------------------
# 2. Select Final Modeling Columns
# ---------------------------------------

raw_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "step",
    "type",  # will be one-hot encoded
]

engineered_features = [
    "balance_drop_ratio_orig",
    "balance_jump_ratio_dest",
    "zero_to_zero_dest",
    "unchanged_orig_balance",
]

target = ["isFraud"]

final_cols = raw_features + engineered_features + target
df = df[final_cols]

print("Selected columns:", df.columns.tolist())

# ---------------------------------------
# 3. One-Hot Encode Transaction Type
# ---------------------------------------

df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=False)

print("\nOne-hot encoded Columns:")
print([col for col in df.columns if col.startswith("type_")])

# ---------------------------------------
# 4. Clean Impossible Values (Optional)
# ---------------------------------------

# Replace infinite values from ratios
df.replace([np.inf, -np.inf], 0, inplace=True)

# Fill any missing values
df.fillna(0, inplace=True)

# ---------------------------------------
# 5. Save Final Dataset
# ---------------------------------------

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")
print(f"Final dataset shape: {df.shape}")
