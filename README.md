Adaptive Risk Twin – Fraud Detection MVP (Phase 1)

Real-time fraud scoring · What-if simulation engine · Explainable ML · CI/CD · Containerized API · Dashboard-ready

This project is the first phase of a larger Adaptive Risk Twin — an intelligent digital twin that simulates financial risks across anomalies, credit, liquidity and fraud.
Phase 1 focuses on building a production-ready Fraud Detection & Simulation System.

Project Highlights (Business Impact First)
Business Impact

Detects high-risk transactions in real time → reduces potential fraud losses

Runs simulations (e.g., increased transaction amount, changed transaction type) → helps teams understand extreme-case scenarios

Provides explainability (SHAP) → improves trust, audit readiness, and governance

Automated CI/CD → increases reliability and speeds up deployment

Technical Highlights

S3-based data ingestion

EDA → Feature Engineering → LightGBM Model

SHAP Explainability

FastAPI microservice

What-if Simulation Engine

Dockerized API

GitHub CI/CD (tests + formatting)

Streamlit dashboard (coming in Phase 1.5)

Quick Visual Preview of the System

(Insert images later — placeholders included)

1. API + Model Serving Screenshot (placeholder)
[API Screenshot Placeholder]

2. SHAP Explainability Screenshot (placeholder)
[SHAP Screenshot Placeholder]

Architecture — Fraud Module MVP
flowchart TD

    subgraph Training[Training Layer - Model Development]
        A[S3 - Raw PaySim Data] --> B[EDA & Cleaning]
        B --> C[Feature Engineering<br/>Balance drop/jump ratios,<br/>zero-to-zero, unchanged balance]
        C --> D[LightGBM Fraud Model Training]
        D --> E[SHAP Explainability]
    end

    subgraph Serving[Serving Layer - Real-time API]
        F[FastAPI Fraud Scoring API]
        G[Docker Container]
        F --> G
    end

    subgraph Simulation[Simulation Layer]
        H[What-if Simulation Engine<br/>(Amount shock, type change)]
        F --> H
    end

    subgraph Dashboard[Streamlit Dashboard (Phase 1.5)]
        I[Scoring UI<br/>Simulation UI<br/>SHAP UI]
        F --> I
        H --> I
    end

    subgraph CI[CI/CD Layer]
        J[GitHub Actions<br/>Tests + Black Format]
        J --> F
    end

    D --> F
    E --> I

Features
1. Feature Engineering

Balance drop features

Balance jump features

Ratio-based fraud signatures

Zero-to-zero destination handling

Unchanged balance indicator

Transaction type one-hot encoding

Schema alignment to ensure API consistency

2. Model – LightGBM

Trained on engineered PaySim dataset

Handles heavy class imbalance

Extremely fast inference

High ROC-AUC

3. Explainability – SHAP

Global feature importance

Per-transaction local explanations

Helps compliance teams understand fraud risk

4. FastAPI Microservice

Endpoints included:

POST /predict_fraud
GET  /health
POST /simulate

5. Simulation Engine (What-if Analysis)

Apply amount shock (e.g., +20%)

Force transaction type change

View change in fraud probability

Identify high-risk scenarios without real exposure

6. Docker Containerization

Reproducible environment

No dependency issues

Deployable on any cloud

7. CI/CD – GitHub Actions

Installs dependencies

Runs Black formatter

Executes unit tests

Ensures code quality before merging

📁 Project Structure
Adaptive_Risk_Twin/
│
├── src/
│   ├── api/
│   │   └── paysim_api.py
│   ├── train/
│   │   └── paysim_train_fraud_model.py
│   └── dashboard/   ← Streamlit app (Phase 1.5)
│
├── data/
│   └── processed/
│       └── paysim_cleaned_core_features.csv
│
├── models/
│   └── lgbm_fraud.txt
│
├── tests/
│   ├── test_feature_engineering.py
│   └── test_predict_endpoint_mock.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── progress_log.md

How to Run Locally
1. Create venv
python -m venv .venv
.venv/Scripts/activate   # Windows

2. Install requirements
pip install -r requirements.txt

3. Run API
uvicorn src.api.paysim_api:app --reload

4. Run dashboard (Phase 1.5)
streamlit run src/dashboard/app.py

Run in Docker

Build image:

docker build -t risk-twin-api .


Run container:

docker run -p 8000:8000 risk-twin-api

CI/CD

GitHub Actions workflow automatically:

Installs Python

Installs dependencies

Runs Black formatting

Runs unit tests

Rejects merge if anything breaks

Status badges can be added after first green build.

Roadmap (Phase 1 → Phase 2)
Phase 1 (MVP – Fraud Module)

✔ EDA
✔ Feature Engineering
✔ LightGBM
✔ SHAP
✔ FastAPI
✔ Simulation
✔ Docker
✔ Unit Tests
✔ CI Pipeline
Streamlit Dashboard (in progress)

Phase 2 (Full Adaptive Risk Twin)

⬜ Add Credit Model
⬜ Add Liquidity Forecasting Model
⬜ Add Operational Risk Rules
⬜ Multi-Agent Coordination Layer
⬜ Risk Knowledge Graph
⬜ Real-time Streaming (Kafka)
⬜ Enterprise Dashboard (Plotly/Streamlit Pro)

Contact

Moras Kashyap
Machine Learning & Data Science
Ireland
🔗 LinkedIn: (https://www.linkedin.com/in/moras-kashyap/)
🔗 GitHub: https://github.com/moras11