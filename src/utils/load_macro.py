import boto3
import pandas as pd
import os
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

bucket_name = "adaptive-risk-twin-data"
file_key = "raw/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_130173.csv"

print("Downloading Macro CPI dataset from S3...")

obj = s3.get_object(Bucket=bucket_name, Key=file_key)
csv_bytes = obj["Body"].read().decode("utf-8")

# The first 4 rows in World Bank data are metadata
df = pd.read_csv(StringIO(csv_bytes), skiprows=4)

print(df.head())
print(f"\nLoaded {len(df)} macro rows successfully!")

output_path = "data/processed/macro_cpi.csv"
df.to_csv(output_path, index=False)

print(f"Saved processed macro dataset to {output_path}")
