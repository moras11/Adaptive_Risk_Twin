## Day 1 — Project Setup Completed

### Achievements
- Created the main GitHub repository: `Adaptive_Risk_Twin`.
- Set up the full project folder structure including:
  - src/
  - data/raw
  - data/processed
  - models/
  - reports/
  - platform_concepts/
- Added initial files: README.md and .gitignore.
- Created a local Python virtual environment (.venv).
- Installed essential packages (pandas, boto3, python-dotenv, etc.).
- Created S3 bucket: `adaptive-risk-twin-data`.
- Added all folders inside S3: raw/, processed/, models/, reports/.
- Downloaded the three datasets required:
  - PaySim (fraud transactions)
  - LendingClub (credit risk)
  - World Bank CPI (macro data)
- Uploaded all datasets into: `s3://adaptive-risk-twin-data/raw/`.
- Set up AWS account, created Access Key, and stored credentials securely in `.env`.
- Added `.env` to `.gitignore`.
- Added Databricks / Snowflake / BigQuery concept files into `platform_concepts/`.

### Skills Learned
- How to design a clean data science project structure.
- How to use GitHub + GitHub Desktop.
- How to create and organize an S3 bucket.
- How to download and organize real-world financial datasets.
- How to securely configure AWS credentials in a Python project.
- Conceptual understanding of Databricks, Snowflake, and BigQuery ingestion flows.

### Notes
Day 1 was focused entirely on setup — environment, datasets, structure, and cloud configuration.  
A strong foundation for Day 2 and beyond.



## Day 2 — Data Ingestion Completed (S3 → Python → Local)

### Achievements
- Successfully connected Python environment to AWS S3 using boto3.
- Created secure `.env` file and validated AWS credentials.
- Loaded PaySim fraud dataset from S3 and saved to data/processed.
- Implemented chunked streaming download for LendingClub (1.6GB).
- Loaded LendingClub dataset successfully and saved to processed folder.
- Loaded World Bank CPI macro dataset and saved to processed folder.
- Added clean ingestion scripts inside src/utils:
    - load_paysim.py
    - load_lending.py
    - load_macro.py

### Skills Learned
- Reading large files from S3 with chunked streaming.
- Organizing ingestion pipelines in project structure.
- Using environment variables for secure credentials.
- Handling read timeouts and debugging AWS endpoints.

### Next Steps (Day 3)
- Perform initial EDA (head, shape, nulls, dtypes).
- Validate schemas for fraud, credit, macro data.
- Clean and normalize fields.
- Save cleaned versions to processed folder.
- Begin Databricks / Snowflake conceptual comparison for EDA.

