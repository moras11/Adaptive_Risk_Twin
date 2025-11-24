# Adaptive Risk Twin – Fraud Detection MVP (Phase 1)

Real-time fraud scoring · What‑if simulation engine · Explainable ML · CI/CD · Containerized API · Dashboard-ready

This project is the first phase of a larger Adaptive Risk Twin — an intelligent digital twin that simulates financial risks across anomalies, credit, liquidity and fraud.  
Phase 1 delivers a production-ready Fraud Detection & Simulation engine.

---

## System Architecture

Below is a high-level architecture overview of the training and serving layers.

![System Architecture](Adaptive_Risk_Twin_Flowchart.png)

---

## Key Features

### Business Impact
- Detects high‑risk transactions in real time → reduces fraud losses  
- Runs simulations (amount shock, type change) → supports extreme‑case risk assessment  
- Provides explainability (SHAP) → improves trust, audit readiness, governance  
- Automated CI/CD → reliable, consistent deployment  

### Technical Highlights
- S3-based ingestion  
- EDA → Feature Engineering → LightGBM Model  
- SHAP Explainability  
- FastAPI microservice  
- What‑if Simulation Engine  
- Dockerized API  
- GitHub Actions: Black formatting + unit tests  

---

## Feature Engineering
- Balance drop ratios  
- Balance jump ratios  
- Zero‑to‑zero destination handling  
- Unchanged balance indicator  
- Transaction type one‑hot encoding  
- Schema alignment for consistent API inputs  

---

## Model – LightGBM
- Trained on engineered PaySim dataset  
- Handles extreme class imbalance  
- Fast inference & high ROC-AUC  

---

## How to Run Locally

1. **Create venv**  
   ```bash
   python -m venv .venv
   .venv/Scripts/activate   # Windows
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run API**  
   ```bash
   uvicorn src.api.paysim_api:app --reload
   ```

4. **Run dashboard (Phase 1.5)**  
   ```bash
   streamlit run src/dashboard/app.py
   ```

---

## CI Pipeline (GitHub Actions)

- Installs dependencies  
- Runs Black formatting  
- Runs unit tests  
- Rejects merge if anything fails  

---

## Roadmap

### Phase 1 (Fraud Module)  
✔ EDA  
✔ Feature Engineering  
✔ LightGBM  
✔ SHAP  
✔ FastAPI  
✔ Simulation  
✔ Docker  
✔ Unit Tests  
✔ CI Pipeline  
⧗ Streamlit Dashboard (in progress)

### Phase 2 (Full Adaptive Risk Twin)
- Credit Model  
- Liquidity Forecasting  
- Operational Risk Rules  
- Multi‑Agent Coordination Layer  
- Risk Knowledge Graph  
- Real-time Streaming (Kafka)  
- Enterprise Dashboard (Plotly/Streamlit Pro)

---

## Contact

**Moras Kashyap — Machine Learning & Data Science**  
LinkedIn: https://www.linkedin.com/in/moras-kashyap/  
GitHub: https://github.com/moras11
