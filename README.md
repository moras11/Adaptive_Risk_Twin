# Adaptive Risk Twin – Fraud Detection MVP (Phase 1)

Real-time fraud scoring • What-if simulation engine • Explainable ML • CI/CD • Containerized API • Dashboard-ready

This project is the first phase of a larger Adaptive Risk Twin — an intelligent digital twin designed to simulate and analyze financial risks across fraud, credit, liquidity, and operational domains.  
Phase 1 focuses on building a production-ready Fraud Detection & Simulation System.

---

## Project Highlights

### Business Impact
- Detects high-risk transactions in real time, reducing potential fraud losses.
- Runs controlled what-if simulations (e.g., increased amount, altered transaction type) to understand risk sensitivity and extreme-case scenarios.
- Provides model explainability for audit readiness, compliance, and governance.
- Automated CI/CD ensures reliability and fast deployment.

### Technical Highlights
- S3-based data ingestion  
- EDA → Feature Engineering → LightGBM Model  
- SHAP explainability  
- FastAPI microservice for real-time inference  
- What-if simulation engine  
- Dockerized application  
- GitHub Actions CI pipeline  
- Streamlit dashboard (Phase 1.5)

---

## Quick Visual Preview

These will be filled after generating screenshots:

- API response preview  
- SHAP feature importance preview  
- Simulation preview  

---

## System Architecture (Phase 1)

```mermaid
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
```

---

## Features

### 1. Feature Engineering
- Balance drop features  
- Balance jump features  
- Ratio-based fraud signatures  
- Zero-to-zero destination handling  
- Unchanged balance indicator  
- Transaction type one-hot encoding  
- Schema alignment to ensure API consistency  

### 2. Model – LightGBM
- Trained on engineered PaySim dataset  
- Handles extreme class imbalance  
- Fast inference suitable for real-time systems  
- High ROC-AUC  

### 3. Explainability – SHAP
- Global feature importance  
- Local explanations for individual predictions  
- Supports governance and model transparency  

### 4. FastAPI Microservice
Endpoints:

```
POST /predict_fraud
POST /simulate
GET  /health
```

### 5. What-if Simulation Engine
- Adjust transaction amount (percentage shock)
- Override transaction type
- Compare baseline vs simulated fraud risk
- Understand sensitivity and model behavior

### 6. Dockerized Deployment
- Reproducible environment  
- Easy to deploy on any cloud or Kubernetes  

### 7. CI/CD Pipeline
- Black formatting enforcement  
- Unit tests for feature engineering and API  
- Pull request quality checks  
- Automated build and test workflow  

---

## Project Structure

```
Adaptive_Risk_Twin/
│
├── src/
│   ├── api/
│   │   └── paysim_api.py
│   ├── train/
│   │   └── paysim_train_fraud_model.py
│   └── dashboard/
│       └── app.py   (Phase 1.5)
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
```

---

## How to Run Locally

### 1. Create virtual environment

```
python -m venv .venv
.venv/Scripts/activate   # Windows
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the API

```
uvicorn src.api.paysim_api:app --reload
```

### 4. Run the Streamlit Dashboard (Phase 1.5)

```
streamlit run src/dashboard/app.py
```

---

## Running with Docker

### Build image
```
docker build -t risk-twin-api .
```

### Run container
```
docker run -p 8000:8000 risk-twin-api
```

---

## CI/CD

The GitHub Actions workflow performs:

- Dependency installation  
- Black formatting validation  
- Unit tests  
- API import checks  
- Automatic failure on formatting or test errors  

This ensures safe, reliable, production-friendly commits.

---

## Roadmap

### Phase 1 (Fraud MVP)
- EDA  
- Feature Engineering  
- LightGBM Model  
- SHAP Explainability  
- FastAPI  
- Simulation Engine  
- Docker  
- Unit Tests  
- CI Pipeline  
- Streamlit Dashboard (in progress)

### Phase 2 (Full Adaptive Risk Twin)
- Add Credit Risk Model  
- Add Liquidity Forecasting Model  
- Add Operational Risk Rules  
- Multi-Agent Coordination Layer  
- Risk Knowledge Graph  
- Real-time Streaming (Kafka)  
- Enterprise Dashboard (Plotly/Streamlit Pro)

---

## Contact

**Moras Kashyap**  
Machine Learning & Data Science  
GitHub: https://github.com/moras11  
LinkedIn: https://www.linkedin.com/in/moras-kashyap/
