import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pdfkit
from scipy.special import expit  # sigmoid
import os

API_BASE_URL = "http://127.0.0.1:8000"

# ---- adjust this if wkhtmltopdf is installed elsewhere ----
WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"


# ---------------- HELPERS ----------------
def call_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}


def call_predict(payload):
    try:
        r = requests.post(f"{API_BASE_URL}/predict_fraud", json=payload, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def call_simulate(payload):
    try:
        r = requests.post(f"{API_BASE_URL}/simulate", json=payload, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def call_shap(payload):
    try:
        r = requests.post(f"{API_BASE_URL}/explain_shap", json=payload, timeout=8)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def build_gauge(prob: float) -> go.Figure:
    """Premium Fraud Risk Gauge with light green/yellow/red and golden accent."""
    pct = float(prob) * 100.0

    # Determine risk zone and bar color
    if pct < 30:
        zone = "Low"
        bar_color = "#7CFF7C"       # light premium green
    elif pct < 70:
        zone = "Medium"
        bar_color = "#FFD966"       # champagne / light gold-ish yellow
    else:
        zone = "High"
        bar_color = "#FF7A7A"       # soft premium red

    # Premium arc colors (light versions)
    green_band  = "rgba(124, 255, 124, 0.35)"   # light mint green
    yellow_band = "rgba(255, 217, 102, 0.35)"   # premium gold tint
    red_band    = "rgba(255, 122, 122, 0.35)"   # soft red

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pct,

        # Title
        title={
            "text": f"<b>Fraud Risk:</b> {zone}",
            "font": {"size": 28, "color": "white"},
        },

        # Number in center
        number={
            "font": {"size": 36, "color": "white"},
            "suffix": "%"
        },

        gauge={
            "shape": "angular",

            # ---------------- AXIS ----------------
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 20, 40, 60, 80, 100],
                "tickfont": {"size": 13, "color": "#cccccc"},
                "tickcolor": "#999999",
                "tickwidth": 1
            },

            # ---------------- COLOR BANDS ----------------
            "steps": [
                {"range": [0, 30], "color": green_band},
                {"range": [30, 70], "color": yellow_band},
                {"range": [70, 100], "color": red_band},
            ],

            # ---------------- PREMIUM BAR ----------------
            "bar": {
                "color": bar_color,
                "thickness": 0.15
            },

            # ---------------- GOLDEN WHITE NEEDLE ----------------
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": pct,
                "thickness": 0.9
            },

            "borderwidth": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    ))

    fig.update_layout(
        height=340,
        margin=dict(l=50, r=50, t=80, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    return fig



import plotly.graph_objects as go
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_shap_waterfall_probability(base_value, shap_values, feature_names):
    """
    Creates a correct probability-based SHAP waterfall using cumulative logits.
    """

    import plotly.graph_objects as go
    import numpy as np

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Running totals
    running_logit = base_value
    points = [sigmoid(running_logit)]  # start probability
    labels = ["Base Probability"]

    # Build cumulative steps
    for i, (name, shap_val) in enumerate(zip(feature_names, shap_values)):
        running_logit += shap_val
        prob = sigmoid(running_logit)
        points.append(prob)
        labels.append(name)

    # Final step
    labels.append("Final Probability")
    points.append(points[-1])

    # Convert to percent
    points_pct = [p * 100 for p in points]

    # Build waterfall bars (difference between steps)
    diffs = [points_pct[0]] + [points_pct[i] - points_pct[i - 1] for i in range(1, len(points_pct))]

    colors = [
        "#FFD700" if lbl in ["Base Probability", "Final Probability"]
        else ("#7AC77E" if d >= 0 else "#FF6A6A")
        for lbl, d in zip(labels, diffs)
    ]

    fig = go.Figure(go.Waterfall(
        name="SHAP Probabilities",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(points_pct) - 2) + ["absolute"],
        x=labels,
        y=diffs,
        connector={"line": {"color": "white"}},
        text=[f"{val:.1f}%" for val in points_pct],
        textposition="outside",
        increasing={"marker": {"color": "#7AC77E"}},
        decreasing={"marker": {"color": "#FF6A6A"}},
    ))

    fig.update_layout(
        title="SHAP Waterfall (Probability Scale)",
        showlegend=False,
        template="plotly_dark",
        height=600,
        yaxis_title="Impact on Fraud Probability (%)",
        xaxis_tickangle=-45
    )

    return fig



def generate_pdf_report(payload, prediction, shap_result) -> bytes:
    """
    Generate a financial-style HTML -> PDF fraud report using pdfkit.
    (text + tables; visuals stay in the app UI)
    """
    tx_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in payload.items()
    )

    top_features = shap_result.get("top_features", [])
    tf_rows = "".join(
        f"<tr><td>{feat}</td><td>{val:.4f}</td></tr>"
        for feat, val in top_features
    )

    prob = prediction["fraud_probability"]
    flag = "FRAUD" if prediction["is_fraud"] else "SAFE"
    threshold = prediction["threshold"]

    narrative = (
        "The model assigns a HIGH fraud risk to this transaction, driven by large "
        "origin balance drops and suspicious destination balance jumps."
        if prediction["is_fraud"]
        else "The model assigns a LOW fraud risk to this transaction; feature "
             "contributions do not indicate typical fraud patterns."
    )

    html = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 24px 40px;
                background-color: #ffffff;
                color: #222222;
            }}
            h1, h2, h3 {{
                color: #1a1a1a;
            }}
            .header {{
                border-bottom: 2px solid #444;
                margin-bottom: 24px;
                padding-bottom: 8px;
            }}
            .header-title {{
                font-size: 24px;
                font-weight: bold;
            }}
            .section {{
                margin-bottom: 24px;
                padding: 12px 16px;
                border-radius: 6px;
                border: 1px solid #d0d0d0;
                background-color: #fafafa;
            }}
            .section-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 8px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 6px 8px;
                font-size: 12px;
            }}
            th {{
                background-color: #f0f0f0;
            }}
            .tag {{
                display: inline-block;
                padding: 3px 10px;
                border-radius: 999px;
                font-size: 11px;
                margin-left: 6px;
            }}
            .tag-safe {{
                background-color: #e6f4ea;
                color: #137333;
            }}
            .tag-fraud {{
                background-color: #fce8e6;
                color: #c5221f;
            }}
            .smalltext {{
                font-size: 11px;
                color: #666666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-title">Adaptive Risk Twin – Transaction Risk Analysis Report</div>
            <div class="smalltext">Generated automatically by the Fraud Risk Twin prototype.</div>
        </div>

        <div class="section">
            <div class="section-title">1. Prediction Summary</div>
            <p><b>Fraud Probability:</b> {prob:.4f}</p>
            <p><b>Decision Threshold:</b> {threshold:.2f}</p>
            <p><b>Final Decision:</b>
                <span class="tag {'tag-fraud' if prediction['is_fraud'] else 'tag-safe'}">
                    {flag}
                </span>
            </p>
            <p><b>Analyst Narrative:</b> {narrative}</p>
        </div>

        <div class="section">
            <div class="section-title">2. Transaction Details</div>
            <table>
                <tr><th>Field</th><th>Value</th></tr>
                {tx_rows}
            </table>
        </div>

        <div class="section">
            <div class="section-title">3. Top SHAP Feature Contributions</div>
            <table>
                <tr><th>Feature</th><th>SHAP Value</th></tr>
                {tf_rows}
            </table>
            <p class="smalltext">
                Positive SHAP values push the prediction toward FRAUD; negative values
                push it toward SAFE. Magnitude reflects contribution strength.
            </p>
        </div>

        <div class="section">
            <div class="section-title">4. Methodology Note</div>
            <p class="smalltext">
                This report is generated using a LightGBM fraud model trained on engineered
                transaction features (balance jumps, laundering signatures, and behaviour
                patterns). SHAP (Shapley Additive Explanations) is used to attribute model
                predictions to individual input features, providing regulator-friendly
                transparency for fraud risk decisions.
            </p>
        </div>
    </body>
    </html>
    """

    config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
    pdf_bytes = pdfkit.from_string(html, False, configuration=config)
    return pdf_bytes


# ---------------- PAGE CONFIG --------------
st.set_page_config(page_title="Adaptive Risk Twin", layout="wide")

st.title(" Adaptive Risk Twin – Fraud Risk Dashboard")
st.caption("Real-time scoring • Simulation • Explainability • EDA • Reports")

# init session state
if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_shap" not in st.session_state:
    st.session_state["last_shap"] = None

# ---------------- HEALTH CHECK --------------
health = call_health()
if health.get("status") != "ok":
    st.error("❌ Backend offline. Start FastAPI first.")
    st.json(health)
    st.stop()
else:
    st.success("Backend Connected  | Model Loaded")

# ---------------- TABS ---------------------
tab_predict, tab_sim, tab_shap, tab_eda, tab_report = st.tabs(
    ["Predict Fraud", "What-If Simulation", "Explainability (SHAP)", "EDA", "Reports"]
)

# =====================================================================
# TAB 1 — FRAUD PREDICTION
# =====================================================================
with tab_predict:
    st.header(" Real-Time Fraud Prediction")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount", 0.0, 1e9, 5000.0)
        tx_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
        )
        step = st.number_input("Step (1–744)", 1, 744, 250)

    with col2:
        oldbalanceOrg = st.number_input("Old Balance (Origin)", 0.0, 1e9, 90000.0)
        newbalanceOrig = st.number_input("New Balance (Origin)", 0.0, 1e9, 50000.0)
        oldbalanceDest = st.number_input("Old Balance (Destination)", 0.0, 1e9, 0.0)
        newbalanceDest = st.number_input("New Balance (Destination)", 0.0, 1e9, 50000.0)

    payload = {
        "amount": amount,
        "type": tx_type,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "step": int(step),
    }
    st.session_state["last_payload"] = payload

    if st.button("Predict Fraud"):
        with st.spinner("Calling Fraud Model..."):
            result, err = call_predict(payload)

        if err:
            st.error("Error: " + err)
        else:
            st.session_state["last_prediction"] = result

            st.markdown("### Prediction Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
            c2.metric("Fraud Flag", "FRAUD" if result["is_fraud"] else "SAFE")
            c3.metric("Threshold", result["threshold"])

            gauge_fig = build_gauge(result["fraud_probability"])
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.subheader("Raw Model Response")
            st.json(result)

# =====================================================================
# TAB 2 — WHAT-IF SIMULATION
# =====================================================================
with tab_sim:
    st.header(" What-If Simulation Engine")

    st.info("Modify transaction values and observe how fraud probability changes.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        amount_shock = st.slider("Amount Shock (%)", -50, 200, 20)
        force_type = st.selectbox(
            "Force Transaction Type",
            [None, "CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
        )

    sim_payload = {
        "transaction": payload,
        "amount_shock_percent": amount_shock,
        "force_type": force_type,
    }

    if st.button(" Run Simulation"):
        with st.spinner("Running Simulation..."):
            sim_result, err = call_simulate(sim_payload)

        if err:
            st.error("Error: " + err)
        else:
            base = sim_result["baseline_fraud_probability"]
            shock = sim_result["simulated_fraud_probability"]

            st.metric("Baseline", f"{base:.4f}")
            st.metric("After Shock", f"{shock:.4f}")
            st.metric("Change", f"{shock - base:.4f}")

            st.subheader("Simulation Output")
            st.json(sim_result)

# =====================================================================
# TAB 3 — SHAP EXPLAINABILITY  
# =====================================================================

with tab_shap:

    st.header(" SHAP Explainability")
    st.info("Visual explanation of why the model flagged (or did not flag) fraud.")

    if st.button("🔍 Explain this Transaction"):
        with st.spinner("Computing SHAP values..."):
            shap_result, err = call_shap(payload)

        if err:
            st.error("Error: " + err)
            st.stop()

        # Store for session use
        st.session_state["last_shap"] = shap_result

        shap_values = np.array(shap_result["shap_values"])
        feature_names = shap_result["feature_names"]
        prob = float(shap_result["fraud_probability"])
        base_value = shap_result.get("base_value", 0.0)

        # -------------------------------
        # 1) FRAUD RISK SUMMARY
        # -------------------------------
        st.subheader("Fraud Summary")
        st.metric("Predicted Fraud Probability", f"{prob*100:.2f}%")

        # -------------------------------
        # 2) TOP FEATURE CONTRIBUTIONS TABLE
        # -------------------------------
        st.subheader("Top Feature Contributions")

        abs_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value (logit)": shap_values
        }).sort_values("SHAP Value (logit)", key=np.abs, ascending=False)

        st.dataframe(abs_df.head(10), use_container_width=True)

        # -------------------------------
        # 3) BAR CHART – ABSOLUTE IMPORTANCE
        # -------------------------------
        st.subheader("Top 10 SHAP Feature Importance (Absolute Value)")

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(shap_values)
        }).sort_values("importance", ascending=False)

        fig_bar = px.bar(
            importance_df.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Contributing Features",
            color="importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # =============================================================================
        # 4) PREMIUM ENTERPRISE WATERFALL (Probability-Based)
        # =============================================================================
        st.subheader("Enterprise SHAP Waterfall (Probability-Based)")

        fig_wf = plot_shap_waterfall_probability(
            base_value=base_value,
            shap_values=shap_values,
            feature_names=feature_names
        )


        st.plotly_chart(fig_wf, use_container_width=True)

        # -------------------------------
        # 5) RAW OUTPUT (for debugging)
        # -------------------------------
        with st.expander("Raw SHAP Output"):
            st.json(shap_result)

# =====================================================================
# TAB 4 — EDA PAGE
# =====================================================================
with tab_eda:
    st.header("EDA & Risk Insights")

    st.info("Upload your dataset to explore distributions & correlations.")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except pd.errors.ParserError:
            df = pd.read_csv(uploaded, skiprows=4)
        except Exception:
            st.error("Could not parse this CSV automatically.")
            st.stop()

        st.success("File loaded successfully!")
        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for plotting.")
        else:
            st.subheader("Distribution Plot")
            colsel = st.selectbox("Numeric Column", numeric_cols)
            fig = px.histogram(df, x=colsel, nbins=40)
            st.plotly_chart(fig)

            st.subheader("Correlation Heatmap")
            fig = px.imshow(df[numeric_cols].corr(), text_auto=True, aspect="auto")
            st.plotly_chart(fig)

# =====================================================================
# TAB 5 — REPORTS
# =====================================================================
with tab_report:
    st.header("Transaction Risk Analysis Report")

    if (
        st.session_state["last_payload"] is None
        or st.session_state["last_prediction"] is None
        or st.session_state["last_shap"] is None
    ):
        st.warning(
            "Run a prediction and SHAP explanation first (tabs 1 & 3) "
            "before generating the report."
        )
    else:
        st.success("Prediction + SHAP available – ready to generate report.")
        if st.button("📄 Generate PDF Report"):
            pdf_bytes = generate_pdf_report(
                st.session_state["last_payload"],
                st.session_state["last_prediction"],
                st.session_state["last_shap"],
            )
            st.download_button(
                label="⬇ Download Transaction Risk Analysis Report",
                data=pdf_bytes,
                file_name="Transaction_Risk_Analysis_Report.pdf",
                mime="application/pdf",
            )
