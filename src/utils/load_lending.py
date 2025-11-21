import boto3
import pandas as pd
import os
from dotenv import load_dotenv

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
file_key = "raw/accepted_2007_to_2018Q4.csv"

print("Downloading LendingClub from S3 in streaming mode...")

# Local file path
local_file = "data/processed/lendingclub.csv"

# Stream download using boto3 and write locally
with open(local_file, "wb") as f:
    s3.download_fileobj(bucket_name, file_key, f)

print("File downloaded locally. Now loading with pandas (this may take 1–3 minutes)...")

df = pd.read_csv(local_file, low_memory=False)

print(df.head())
print(f"\nLoaded {len(df)} rows successfully!")
print(f"Saved processed LendingClub to {local_file}")
