from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
PROC_DIR  = BASE_DIR / "output" / "data"
MODEL_DIR = BASE_DIR / "output" / "models"
PLOT_DIR  = BASE_DIR / "output" / "plots"

# ---------- Page config ----------
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

# ---------- Cached resource loaders ----------
@st.cache_resource
def load_model():
    with open(MODEL_DIR / "churn_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_threshold():
    with open(MODEL_DIR / "threshold.txt") as f:
        return float(f.read().strip())

@st.cache_data
def load_raw():
    return pd.read_csv(DATA_DIR / "Telco-Customer-Churn.csv")

@st.cache_data
def load_train_columns():
    return pd.read_csv(PROC_DIR / "X_train.csv", nrows=1).columns.tolist()

# ---------- Header ----------
st.title("📊 Telco Customer Churn Intelligence")
st.markdown(
    "End-to-end churn analytics: from raw data exploration, through an "
    "XGBoost model with a **business-cost-optimal threshold**, to per-customer "
    "explanations powered by SHAP."
)

# ===========================================================================
# TAB 1 — BUSINESS OVERVIEW
# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Business Overview", "🔍 EDA Insights",
    "🤖 Model Performance", "🎯 Customer Risk Predictor"
])

with tab1:
    st.header("Business Overview")
    raw = load_raw()
    total_customers = len(raw)
    churners = (raw["Churn"] == "Yes").sum()
    churn_rate = churners / total_customers

    raw["MonthlyCharges"] = pd.to_numeric(raw["MonthlyCharges"], errors="coerce")
    avg_monthly = raw["MonthlyCharges"].mean()
    revenue_at_risk = churners * avg_monthly * 12   # annualized

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churn Rate", f"{churn_rate*100:.1f}%",
              delta=f"-{churners:,} customers", delta_color="inverse")
    c3.metric("Annualized Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.markdown("---")
    st.subheader("Why this matters")
    st.write(
        f"With an average monthly bill of **${avg_monthly:.2f} and "
        f"**{churners:,}** customers churning, the company is exposed to "
        f"approximately **${revenue_at_risk/1e6:.2f} M** in lost annual revenue. "
        f"Targeting even a fraction of these customers with retention offers "
        f"could recover millions."
    )

# ===========================================================================
# TAB 2 — EDA INSIGHTS
# ===========================================================================
with tab2:
    st.header("Exploratory Data Analysis")
    eda_files = [
        "churn_distribution.png",
        "churn_by_contract.png",
        "tenure_distribution.png",
        "monthly_charges_boxplot.png",
    ]
    captions = [
        "Overall churn distribution",
        "Churn rate by contract type",
        "Tenure distribution by churn status",
        "Monthly charges vs churn",
    ]
    for fname, cap in zip(eda_files, captions):
        p = PLOT_DIR / fname
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)
        else:
            st.warning(f"Missing plot: {fname} — run 1_eda.py first.")

# ===========================================================================
# TAB 3 — MODEL PERFORMANCE
# ===========================================================================
with tab3:
    st.header("Model Performance")
    model = load_model()
    thresh = load_threshold()

    X_test = pd.read_csv(PROC_DIR / "X_test.csv")
    X_test = X_test.select_dtypes(exclude=['object'])
    y_test = pd.read_csv(PROC_DIR / "y_test.csv").squeeze()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_proba >= thresh).astype(int)

    c1, c2 = st.columns(2)
    c1.metric("Optimal Threshold", f"{thresh:.2f}")
    from sklearn.metrics import roc_auc_score
    c2.metric("ROC-AUC (test)", f"{roc_auc_score(y_test, y_proba):.4f}")

    st.subheader("Saved evaluation plots")
    for fname, cap in [
        ("roc_curve.png", "ROC Curve"),
        ("pr_curve.png", "Precision-Recall Curve"),
        ("confusion_matrix_05.png", "Confusion Matrix @ 0.5"),
        ("threshold_cost_curve.png", "Threshold vs Total Cost"),
    ]:
        p = PLOT_DIR / fname
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)

    # ----- Confusion matrix at the OPTIMAL threshold -----
    st.subheader(f"Confusion Matrix at Optimal Threshold ({thresh:.2f})")
    from sklearn.metrics import confusion_matrix
    cm_opt = confusion_matrix(y_test, y_pred_opt, labels=[0, 1])
    fig, ax = plt.subplots()
    import seaborn as sns
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

# ===========================================================================
# TAB 4 — CUSTOMER RISK PREDICTOR
# ===========================================================================
with tab4:
    st.header("Customer Risk Predictor")
    st.write(
        "Enter the customer's details below. The model will return the "
        "churn probability, a risk level, and a SHAP explanation of *why*."
    )

    
    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)

        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0,
            value=float(tenure * monthly_charges), step=10.0)

        submitted = st.form_submit_button("Predict Churn Risk")

    # ---------- Preprocess + Predict ----------
    if submitted:
        raw_input = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "PhoneService": 1 if phone_service == "Yes" else 0,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": 1 if paperless == "Yes" else 0,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([raw_input])

        ohe_cols = [
            "InternetService", "Contract", "PaymentMethod",
            "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        input_enc = pd.get_dummies(input_df, columns=ohe_cols, drop_first=False)
        bool_cols = input_enc.select_dtypes(include="bool").columns
        input_enc[bool_cols] = input_enc[bool_cols].astype(int)

        # missing cols -> 0
        train_cols = load_train_columns()
        train_cols_numeric = [c for c in train_cols if c != 'gender']
        input_enc = input_enc.reindex(columns=train_cols_numeric, fill_value=0)

        proba = model.predict_proba(input_enc)[0, 1]
        thresh = load_threshold()

        # ---------- Risk bucket ----------
        if   proba < 0.30: level, color = "Low",    "#2ca02c"
        elif proba < thresh: level, color = "Medium", "#ff7f0e"
        else:                level, color = "High",   "#d62728"

        # ---------- Gauge ----------
        st.subheader("Churn Probability")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30],  "color": "#d4f4dd"},
                    {"range": [30, 50], "color": "#fff3cd"},
                    {"range": [50, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": thresh * 100,
                },
            },
        ))
        gauge.update_layout(height=280)
        st.plotly_chart(gauge, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{proba*100:.1f}%")
        m2.metric("Risk Level", level)
        m3.metric("Optimal Threshold", f"{thresh:.2f}")

        # ---------- SHAP waterfall for this single prediction ----------
        st.subheader("Why this prediction? (SHAP Waterfall)")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_enc)
        if isinstance(sv, list): sv = sv[1]
        ev = (explainer.expected_value[1]
              if isinstance(explainer.expected_value, (list, np.ndarray))
              else explainer.expected_value)

        expl = shap.Explanation(
            values=sv[0], base_values=ev, data=input_enc.iloc[0],
            feature_names=list(input_enc.columns),
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(expl, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ---------- Recommended action ----------
        st.subheader("Recommended Action")
        if level == "High":
            st.error(
                "🚨 **High Risk — Intervene now.**  Offer a loyalty discount or "
                "contract upgrade, assign a retention specialist, and prioritize "
                "this customer in the next win-back campaign."
            )
        elif level == "Medium":
            st.warning(
                "⚠️ **Medium Risk — Engage proactively.**  Send a personalized "
                "usage report, offer a feature tutorial, and consider a small "
                "loyalty perk to reduce churn probability."
            )
        else:
            st.success(
                "✅ **Low Risk — Maintain relationship.**  Continue standard "
                "engagement; focus retention budget on higher-risk segments."
            )