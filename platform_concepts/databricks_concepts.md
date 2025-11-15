
# Databricks – Conceptual Notes for Adaptive Risk Twin

## 1. What Databricks Is
Databricks is a unified analytics and machine learning platform combining:
- Apache Spark for distributed compute  
- Delta Lake for reliable storage  
- MLflow for experiment tracking  
- Jobs & Pipelines for orchestration  

It is often used in financial institutions for fraud detection, risk modeling, liquidity simulations, and enterprise-scale data engineering.

## 2. Data Ingestion in Databricks

### Mount S3
```python
dbutils.fs.mount(
    source="s3a://adaptive-risk-twin-data",
    mount_point="/mnt/risk"
)
```

### Read CSVs
```python
paysim_df = spark.read.csv("/mnt/risk/raw/paysim.csv", header=True, inferSchema=True)
```

### Write as Delta
```python
paysim_df.write.format("delta").mode("overwrite").saveAsTable("risk.paysim")
```

## 3. Feature Engineering
Example:
```python
from pyspark.sql.functions import *

agg = paysim_df.groupBy("nameOrig").agg(
    count("*").alias("txn_count"),
    sum("amount").alias("total_amount"),
    sum("isFraud").alias("fraud_count")
)
```

## 4. MLflow
```python
import mlflow

mlflow.set_experiment("fraud_model")

with mlflow.start_run():
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("auc", 0.93)
```

## 5. Orchestration
Use Databricks Jobs to chain:
- ingestion notebook  
- feature engineering notebook  
- model training  
- simulation notebook  

## 6. Adaptive Risk Twin Mapping
- Delta = main storage  
- Spark = scalable ETL  
- MLflow = model lifecycle  
- Jobs = simulation pipeline  
