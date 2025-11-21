"""
Fast SHAP Explainability for LightGBM Fraud Model
This script:
- Loads the LightGBM model + scaler
- Samples 1000 rows (to avoid long compute)
- Computes SHAP values quickly
- Saves summary plot, bar plot, dependence plots
"""

import os
import pandas as pd
import numpy as np
import shap
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load dataset + SAMPLE for SHAP (IMPORTANT)
# -----------------------------------------------------

DATA_PATH = os.path.join("data", "processed", "paysim_cleaned_core_features.csv")
MODEL_PATH = "models/lgbm_fraud.txt"
SCALER_PATH = "models/scaler.pkl"

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ⭐ SAMPLE ONLY 1000 ROWS (SUPER FAST SHAP)
df_sample = df.sample(n=1000, random_state=42)

y = df_sample["isFraud"]
X = df_sample.drop("isFraud", axis=1)

# -----------------------------------------------------
# 2. Load model + scaler
# -----------------------------------------------------

print("Loading LightGBM model & scaler...")
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

model = lgb.Booster(model_file=MODEL_PATH)

# Create save directory
os.makedirs("reports/shap", exist_ok=True)

# -----------------------------------------------------
# 3. SHAP Explainer (FAST MODE)
# -----------------------------------------------------

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(
    model, feature_perturbation="tree_path_dependent"  # FAST mode
)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_scaled)

# -----------------------------------------------------
# 4. Summary Plot
# -----------------------------------------------------

print("Saving SHAP summary plot...")
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("reports/shap/summary_plot.png", dpi=300)
plt.close()

# -----------------------------------------------------
# 5. Bar Plot (Feature Importance)
# -----------------------------------------------------

print("Saving SHAP bar plot...")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("reports/shap/shap_importance_bar.png", dpi=300)
plt.close()

# -----------------------------------------------------
# 6. Dependence plots for top 3 features
# -----------------------------------------------------

print("Saving SHAP dependence plots...")

# Compute feature importance
mean_abs_values = np.abs(shap_values).mean(axis=0)
top_indices = mean_abs_values.argsort()[-3:][::-1]

top_features = [X.columns[i] for i in top_indices]

for feat in top_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X, show=False)
    plt.tight_layout()
    out_path = f"reports/shap/dependence_{feat}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")

print("\nSHAP explainability generated successfully!")
