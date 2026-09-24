import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1f2328; margin-bottom:0; }
    .sub-title   { font-size:1rem;   color:#57606a;   margin-top:0; }
    .metric-card { background:#f7f8fa; border:1px solid #e5e7eb; border-radius:8px;
                   padding:16px 20px; text-align:center; }
    .risk-high   { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5;
                   border-radius:8px; padding:16px; font-size:1.2rem; font-weight:700;
                   text-align:center; }
    .risk-low    { background:#f0fdf4; color:#16a34a; border:1px solid #86efac;
                   border-radius:8px; padding:16px; font-size:1.2rem; font-weight:700;
                   text-align:center; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model artifacts
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, encoders, metrics

def artifacts_ready():
    return (
        os.path.exists("model.pkl") and
        os.path.exists("label_encoders.pkl") and
        os.path.exists("metrics.json")
    )

# ──────────────────────────────────────────────
# Preprocessing helper
# ──────────────────────────────────────────────
OCCUPATION_TYPES = [
    "Unknown", "Laborers", "Core staff", "Accountants",
    "High skill tech staff", "Sales staff", "Managers", "Drivers",
    "Medicine staff", "Cleaning staff", "HR staff", "Security staff",
    "Cooking staff", "Waiters/barmen staff", "Low-skill Laborers",
    "Private service staff", "Secretaries", "Realty agents", "IT staff",
]

FEATURE_NAMES = [
    "age", "gender", "owns_car", "owns_house", "no_of_children",
    "net_yearly_income", "no_of_days_employed", "occupation_type",
    "total_family_members", "migrant_worker", "yearly_debt_payments",
    "credit_limit", "credit_limit_used(%)", "credit_score",
    "prev_defaults", "default_in_last_6months",
]

def preprocess_input(data: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df["gender"]        = encoders["gender"].transform([data["gender"]])[0]
    df["owns_car"]      = encoders["owns_car"].transform([data["owns_car"]])[0]
    df["owns_house"]    = encoders["owns_house"].transform([data["owns_house"]])[0]
    df["occupation_type"] = encoders["occupation_type"].transform([data["occupation_type"]])[0]
    return df[FEATURE_NAMES]

def preprocess_batch(df_raw: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df_raw.copy()
    df.drop(columns=["customer_id", "name"], errors="ignore", inplace=True)
    if "credit_card_default" in df.columns:
        df.drop(columns=["credit_card_default"], inplace=True)
    df["owns_car"]   = df["owns_car"].replace("", "N").fillna("N")
    df["migrant_worker"] = df["migrant_worker"].replace("", "0.0").fillna(0.0).astype(float)
    df["gender"] = df["gender"].replace("XNA", "F").fillna("F")
    for col in ["gender", "owns_car", "owns_house", "occupation_type"]:
        le = encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda v: le.transform([v])[0] if v in le.classes_ else le.transform([le.classes_[0]])[0]
        )
    return df[FEATURE_NAMES]

# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────
st.sidebar.markdown("## 💳 Credit Risk Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Performance", "📈 Data Explorer"],
)
st.sidebar.markdown("---")

# ══════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 💳 Credit Risk Prediction System")
    st.markdown(
        "An end-to-end machine learning application that analyses a customer's "
        "demographic and financial profile to predict the likelihood of **credit card default**. "
        "Built on a Random Forest classifier trained on **45,528 real customer records**."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    if artifacts_ready():
        _, _, metrics = load_artifacts()
        acc = f"{metrics['accuracy']*100:.1f}%"
        roc = str(metrics["roc_auc"])
    else:
        acc = "—"
        roc = "—"
    col1.metric("Training Records", "45,528")
    col2.metric("Input Features", "16")
    col3.metric("Model Accuracy", acc)
    col4.metric("ROC-AUC Score", roc)

    st.markdown("---")
    st.markdown("### How to use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🔍 Single Prediction**\n\nEnter one customer's details manually and get an instant default prediction with probability score.")
    with c2:
        st.info("**📂 Batch Prediction**\n\nUpload a customer dataset CSV to predict credit risk for all customers at once and download results.")
    with c3:
        st.info("**📊 Model Performance**\n\nView accuracy, ROC-AUC, confusion matrix and feature importance charts.")

    if not artifacts_ready():
        st.warning("⚠️ Model not trained yet. Run `python train_model.py` first, then refresh this page.")
    else:
        st.success("✅ Model is ready. Use the sidebar to navigate.")

# ══════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown("Fill in the customer details below to predict credit default risk.")

    if not artifacts_ready():
        st.error("Model not found. Please run `python train_model.py` first.")
        st.stop()

    model, encoders, metrics = load_artifacts()

    with st.form("predict_form"):
        st.markdown("### 👤 Personal Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
        with c2:
            gender = st.selectbox("Gender", ["F", "M"])
        with c3:
            owns_car = st.selectbox("Owns Car", ["N", "Y"])
        with c4:
            owns_house = st.selectbox("Owns House", ["N", "Y"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            no_of_children = st.number_input("No. of Children", min_value=0, max_value=10, value=0)
        with c2:
            total_family_members = st.number_input("Total Family Members", min_value=1, max_value=15, value=2)
        with c3:
            migrant_worker = st.selectbox("Migrant Worker", [0, 1])
        with c4:
            occupation_type = st.selectbox("Occupation Type", OCCUPATION_TYPES)

        st.markdown("### 💰 Financial Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            net_yearly_income = st.number_input("Net Yearly Income (₹)", min_value=0.0, value=150000.0, step=1000.0)
        with c2:
            no_of_days_employed = st.number_input("Days Employed", min_value=0, value=1000)
        with c3:
            yearly_debt_payments = st.number_input("Yearly Debt Payments (₹)", min_value=0.0, value=20000.0, step=500.0)
        with c4:
            credit_limit = st.number_input("Credit Limit (₹)", min_value=0.0, value=30000.0, step=500.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            credit_limit_used = st.slider("Credit Limit Used (%)", min_value=0, max_value=100, value=40)
        with c2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        with c3:
            prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)

        default_in_last_6months = st.selectbox("Default in Last 6 Months", [0, 1])

        submitted = st.form_submit_button("🔮 Predict Credit Risk", use_container_width=True)

    if submitted:
        input_data = {
            "age": age,
            "gender": gender,
            "owns_car": owns_car,
            "owns_house": owns_house,
            "no_of_children": float(no_of_children),
            "net_yearly_income": net_yearly_income,
            "no_of_days_employed": float(no_of_days_employed),
            "occupation_type": occupation_type,
            "total_family_members": float(total_family_members),
            "migrant_worker": float(migrant_worker),
            "yearly_debt_payments": yearly_debt_payments,
            "credit_limit": credit_limit,
            "credit_limit_used(%)": credit_limit_used,
            "credit_score": float(credit_score),
            "prev_defaults": prev_defaults,
            "default_in_last_6months": default_in_last_6months,
        }

        X_input = preprocess_input(input_data, encoders)
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        st.markdown("---")
        st.markdown("### 📋 Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.markdown(f'<div class="risk-high">⚠️ HIGH RISK — Default Predicted<br><small>Probability: {probability*100:.1f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ LOW RISK — No Default Predicted<br><small>Probability of default: {probability*100:.1f}%</small></div>', unsafe_allow_html=True)

        with col2:
            st.metric("Default Probability", f"{probability*100:.1f}%")
            st.metric("Safe Probability", f"{(1-probability)*100:.1f}%")
            risk_level = "High" if probability > 0.6 else "Medium" if probability > 0.3 else "Low"
            st.metric("Risk Level", risk_level)

        # Risk gauge bar
        st.markdown("**Risk Probability Gauge**")
        st.progress(float(probability))
        st.caption(f"Default probability: {probability*100:.1f}% — {'🔴 High' if probability > 0.6 else '🟡 Medium' if probability > 0.3 else '🟢 Low'}")

# ══════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("## 📂 Batch Prediction")
    st.markdown("Upload a customer dataset CSV to predict credit risk for multiple customers at once.")

    if not artifacts_ready():
        st.error("Model not found. Please run `python train_model.py` first.")
        st.stop()

    model, encoders, _ = load_artifacts()

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    # Auto-load test.csv if present
    use_test = False
    if uploaded is None and os.path.exists("test.csv"):
        use_test = st.checkbox("Use the included customer dataset", value=True)

    df_input = None
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
    elif use_test:
        df_input = pd.read_csv("test.csv")

    if df_input is not None:
        st.markdown(f"**{len(df_input)} records loaded**")
        st.dataframe(df_input.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner("Predicting…"):
                X_batch = preprocess_batch(df_input, encoders)
                preds = model.predict(X_batch)
                probas = model.predict_proba(X_batch)[:, 1]

            results = df_input[["customer_id", "name"]].copy() if "customer_id" in df_input.columns else df_input.copy()
            results["default_prediction"] = preds
            results["default_probability_%"] = (probas * 100).round(2)
            results["risk_level"] = pd.cut(
                probas,
                bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )

            st.markdown("### Results")
            st.dataframe(
                results.style.applymap(
                    lambda v: "background-color:#fef2f2" if v == "High"
                    else "background-color:#fefce8" if v == "Medium"
                    else "background-color:#f0fdf4",
                    subset=["risk_level"],
                ),
                use_container_width=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(results))
            col2.metric("Predicted Defaults", int(preds.sum()))
            col3.metric("Default Rate", f"{preds.mean()*100:.1f}%")

            # Risk breakdown chart
            risk_counts = results["risk_level"].value_counts()
            st.markdown("### Risk Level Distribution")
            chart_data = pd.DataFrame({
                "Risk Level": risk_counts.index.tolist(),
                "Count": risk_counts.values.tolist(),
            })
            st.bar_chart(chart_data.set_index("Risk Level"))

            # Download
            csv_out = results.to_csv(index=False)
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ══════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")

    if not artifacts_ready():
        st.error("Model not found. Please run `python train_model.py` first.")
        st.stop()

    _, _, metrics = load_artifacts()

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
    report = metrics["classification_report"]
    col3.metric("F1-Score (Default class)", f"{report['1']['f1-score']:.4f}")

    st.markdown("---")

    # Classification report table
    st.markdown("### Classification Report")
    report_df = pd.DataFrame({
        "Class": ["No Default (0)", "Default (1)", "Macro Avg", "Weighted Avg"],
        "Precision": [
            report["0"]["precision"], report["1"]["precision"],
            report["macro avg"]["precision"], report["weighted avg"]["precision"],
        ],
        "Recall": [
            report["0"]["recall"], report["1"]["recall"],
            report["macro avg"]["recall"], report["weighted avg"]["recall"],
        ],
        "F1-Score": [
            report["0"]["f1-score"], report["1"]["f1-score"],
            report["macro avg"]["f1-score"], report["weighted avg"]["f1-score"],
        ],
        "Support": [
            int(report["0"]["support"]), int(report["1"]["support"]),
            int(report["macro avg"]["support"]), int(report["weighted avg"]["support"]),
        ],
    }).set_index("Class")
    st.dataframe(report_df.style.format("{:.4f}", subset=["Precision", "Recall", "F1-Score"]), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown("### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Default", "Actual: Default"],
            columns=["Predicted: No Default", "Predicted: Default"],
        )
        st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Feature Importance
    with col2:
        st.markdown("### Top Feature Importances")
        fi = pd.DataFrame(metrics["feature_importances"]).head(12)
        fi_chart = fi.set_index("feature")["importance"].sort_values()
        st.bar_chart(fi_chart, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.markdown("## 📈 Data Explorer")

    if not os.path.exists("train.csv"):
        st.error("`train.csv` not found.")
        st.stop()

    df = pd.read_csv("train.csv")
    st.markdown(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Default Rate")
        default_counts = df["credit_card_default"].value_counts().rename({0: "No Default", 1: "Default"})
        st.bar_chart(default_counts, use_container_width=True)
        total = len(df)
        defaults = df["credit_card_default"].sum()
        st.caption(f"Default rate: {defaults/total*100:.1f}%  ({defaults:,} / {total:,})")

    with col2:
        st.markdown("### Gender Distribution")
        gender_counts = df["gender"].replace("XNA", "F").value_counts()
        st.bar_chart(gender_counts, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Credit Score Distribution by Default")
        cs_default = df[df["credit_card_default"] == 1]["credit_score"].dropna()
        cs_no_default = df[df["credit_card_default"] == 0]["credit_score"].dropna()
        chart_data = pd.DataFrame({
            "Default": pd.cut(cs_default, bins=10).value_counts().sort_index().values,
            "No Default": pd.cut(cs_no_default, bins=10).value_counts().sort_index().values,
        })
        st.bar_chart(chart_data, use_container_width=True)

    with col2:
        st.markdown("### Occupation Type Counts")
        occ_counts = df["occupation_type"].value_counts().head(10)
        st.bar_chart(occ_counts, use_container_width=True)

    st.markdown("---")
    st.markdown("### Summary Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
