
# Snowflake + Snowpark – Conceptual Notes for Adaptive Risk Twin

## 1. What Snowflake Is
Snowflake is a cloud-native data warehouse with:
- Elastic compute (warehouses)
- Centralized storage
- Zero-copy cloning
- Snowpark (DataFrame API)

Used heavily in insurance, fintech, and banking.

## 2. Ingesting Data

### Create Stage
```sql
CREATE STAGE risk_stage;
```

### Upload
```sql
PUT file://paysim.csv @risk_stage;
```

### Load
```sql
COPY INTO risk_db.raw.paysim
FROM @risk_stage/paysim.csv
FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1);
```

## 3. Feature Engineering in Snowpark
```python
from snowflake.snowpark import Session
from snowflake.snowpark.functions import *

session = Session.builder.configs(connection_params).create()

df = session.table("risk_db.raw.paysim")

agg = df.group_by("nameOrig").agg(
    count("*").alias("txn_count"),
    sum(col("amount")).alias("total_amount")
)
```

## 4. Task Scheduling
```sql
CREATE OR REPLACE TASK paysim_task
WAREHOUSE=compute_wh
SCHEDULE='1 HOUR'
AS
CALL build_paysim_features();
```

## 5. Adaptive Risk Twin Mapping
- Raw → Snowflake tables  
- Features → Snowpark pipelines  
- Models → external ML or Snowpark ML  
- Simulations → stored as marts  
