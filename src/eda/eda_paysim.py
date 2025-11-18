"""
Complete EDA for PaySim Fraud Dataset
Adaptive Risk Twin – Fraud Module (Core EDA + Fraud Signatures)

Sections:
1. Load + Basic Summary
2. Transaction Type Analysis
3. Fraud vs Amount
4. Fraud vs Origin/Destination Balances
5. Fraud Signature Engineering (MOST IMPORTANT)
6. Deep Fraud Pipeline Analysis
7. Time-Based Fraud Patterns
8. Correlation Heatmap
9. Save Extended EDA Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# ----------------------------------------------------
# Helper: Save plots automatically to reports/eda/
# ----------------------------------------------------

def save_plot(fig, filename):
    os.makedirs(os.path.join("reports", "eda"), exist_ok=True)
    output_path = os.path.join("reports", "eda", filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


# ----------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------

DATA_PATH = os.path.join("data", "processed", "paysim.csv")
print(f"\nLoading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df['isFraud'] = df['isFraud'].astype(int)

print("\n--- DATA PREVIEW ---")
print(df.head())

print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())


# ----------------------------------------------------
# 2. Transaction Type Analysis
# ----------------------------------------------------

fig = plt.figure(figsize=(8,5))
sns.countplot(data=df, x="type", order=df["type"].value_counts().index)
plt.title("Transaction Type Distribution")
plt.xticks(rotation=45)
save_plot(fig, "transaction_type_distribution.png")

fig = plt.figure(figsize=(8,5))
sns.countplot(data=df, x="type", hue="isFraud")
plt.title("Fraud Cases by Type")
plt.xticks(rotation=45)
save_plot(fig, "fraud_by_transaction_type.png")

fraud_rate = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
print("\n--- FRAUD RATE BY TYPE ---")
print(fraud_rate)

fig = plt.figure(figsize=(8,5))
sns.barplot(x=fraud_rate.index, y=fraud_rate.values)
plt.title("Fraud Rate by Transaction Type (%)")
plt.xticks(rotation=45)
save_plot(fig, "fraud_rate_by_type.png")


# ----------------------------------------------------
# 3. Fraud vs Amount
# ----------------------------------------------------

fig = plt.figure(figsize=(8,5))
sns.boxplot(
    data=df[df['amount'] < df['amount'].quantile(0.99)],
    x="isFraud",
    y="amount"
)
plt.title("Amount vs Fraud (1% Trimmed)")
plt.xticks([0, 1], ["Not Fraud", "Fraud"])
save_plot(fig, "amount_vs_fraud.png")


# ----------------------------------------------------
# 4. Fraud vs Origin/Destination Balances
# ----------------------------------------------------

# Origin balance distributions
origin_melt = df.melt(
    id_vars='isFraud',
    value_vars=['oldbalanceOrg', 'newbalanceOrig'],
    var_name='balance_type',
    value_name='balance_value'
)

fig = plt.figure(figsize=(10,5))
sns.boxplot(
    data=origin_melt,
    x='balance_type',
    y='balance_value',
    hue='isFraud'
)
plt.yscale('log')
plt.title("Origin Balances by Fraud vs Non-Fraud")
save_plot(fig, "origin_balances_fraud.png")


# Destination balances
dest_melt = df.melt(
    id_vars='isFraud',
    value_vars=['oldbalanceDest', 'newbalanceDest'],
    var_name='balance_type',
    value_name='balance_value'
)

fig = plt.figure(figsize=(10,5))
sns.boxplot(
    data=dest_melt,
    x='balance_type',
    y='balance_value',
    hue='isFraud'
)
plt.yscale('log')
plt.title("Destination Balances by Fraud vs Non-Fraud")
save_plot(fig, "destination_balances_fraud.png")


# ----------------------------------------------------
# 5. Fraud Signature Engineering (BEST PART)
# ----------------------------------------------------

df['balance_drop_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_drop_ratio_orig'] = df['balance_drop_orig'] / (df['oldbalanceOrg'] + 1)

df['balance_jump_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['balance_jump_ratio_dest'] = df['balance_jump_dest'] / (df['oldbalanceDest'] + 1)

df['zero_to_zero_dest'] = (
    (df['oldbalanceDest'] == 0) &
    (df['newbalanceDest'] == 0) &
    (df['amount'] > 0)
).astype(int)

df['unchanged_orig_balance'] = (
    (df['oldbalanceOrg'] == df['newbalanceOrig']) &
    (df['amount'] > 0)
).astype(int)

fraud_patterns = df.groupby('isFraud')[
    [
        'balance_drop_ratio_orig',
        'balance_jump_ratio_dest',
        'zero_to_zero_dest',
        'unchanged_orig_balance'
    ]
].mean()

print("\n--- FRAUD SIGNATURE ANALYSIS ---")
print(fraud_patterns)

fig = plt.figure(figsize=(10,5))
sns.kdeplot(
    data=df[df['balance_jump_ratio_dest'] < 5000],
    x='balance_jump_ratio_dest',
    hue='isFraud',
    common_norm=False
)
plt.title("Destination Balance Jump Ratio (Clipped)")
save_plot(fig, "jump_ratio_kde.png")


# ----------------------------------------------------
# 6. Deep Fraud Pipeline Analysis (TRANSFER → CASH_OUT)
# ----------------------------------------------------

fraud_only = df[df['isFraud'] == 1]

fig = plt.figure(figsize=(8,5))
sns.countplot(data=fraud_only, x="type")
plt.title("Fraud Types (Only Fraud Rows)")
plt.xticks(rotation=45)
save_plot(fig, "fraud_only_types.png")

print("\n--- FRAUD TRANSACTION TYPE COUNTS ---")
print(fraud_only['type'].value_counts())


# ----------------------------------------------------
# 7. Time-Based Fraud Patterns
# ----------------------------------------------------

fig = plt.figure(figsize=(10,5))
sns.lineplot(
    data=df.groupby("step")["isFraud"].mean().reset_index(),
    x="step",
    y="isFraud"
)
plt.title("Fraud Rate Over Time (Step)")
plt.ylabel("Fraud Rate")
save_plot(fig, "fraud_over_time.png")


# ----------------------------------------------------
# 8. Correlation Heatmap
# ----------------------------------------------------

corr_cols = [
    'amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_drop_ratio_orig',
    'balance_jump_ratio_dest',
    'zero_to_zero_dest',
    'unchanged_orig_balance',
    'isFraud'
]

corr_matrix = df[corr_cols].corr()

fig = plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap – Fraud Features")
save_plot(fig, "correlation_heatmap.png")


# ----------------------------------------------------
# 9. Save Extended EDA Dataset
# ----------------------------------------------------

OUTPUT_PATH = os.path.join("data", "processed", "paysim_eda_extended.csv")
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nExtended dataset saved to: {OUTPUT_PATH}")
