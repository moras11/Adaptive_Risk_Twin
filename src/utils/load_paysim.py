import boto3
import pandas as pd
import os
from dotenv import load_dotenv
from io import StringIO

# Load environment variables from .env
load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Create S3 client
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

BUCKET = "adaptive-risk-twin-data"
KEY = "raw/PS_20174392719_1491204439457_log.csv"   # PaySim file

def load_paysim():
    print("Downloading PaySim from S3...")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    data = obj["Body"].read().decode("utf-8")

    df = pd.read_csv(StringIO(data))
    print("Loaded PaySim successfully!")
    print(df.head())

    # Save to processed folder
    output_path = "data/processed/paysim.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return df


if __name__ == "__main__":
    load_paysim()
