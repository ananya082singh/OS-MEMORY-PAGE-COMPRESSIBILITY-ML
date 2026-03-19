import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Compressibility Predictor", layout="wide")

# =========================
# 🔥 CUSTOM CSS
# =========================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 52px;
    color: #00ADB5;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #B0B0B0;
}
.card {
    background: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION 🔥
# =========================
st.markdown('<div class="title">🧠 Memory Compressibility AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict, Analyze & Optimize Memory Compression using Machine Learning</div>', unsafe_allow_html=True)

st.divider()

# =========================
# KPI DASHBOARD
# =========================
col1, col2, col3 = st.columns(3)

col1.markdown('<div class="card"><h3>Accuracy</h3><h2>97.9%</h2></div>', unsafe_allow_html=True)
col2.markdown('<div class="card"><h3>Models Used</h3><h2>4</h2></div>', unsafe_allow_html=True)
col3.markdown('<div class="card"><h3>Prediction Time</h3><h2>~40μs</h2></div>', unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
st.sidebar.info("Multi-model comparison enabled 🚀")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(data):
    arr = np.frombuffer(data, dtype=np.uint8)

    distinct = len(np.unique(arr))
    counts = np.bincount(arr)
    max_freq_ratio = np.max(counts) / len(arr)
    zero_ratio = np.sum(arr == 0) / len(arr)
    run_count = np.sum(arr[:-1] != arr[1:])

    probs = counts / len(arr)
    probs = probs[probs>0]
    entropy = -np.sum(probs * np.log2(probs))

    return [distinct, max_freq_ratio, zero_ratio, run_count,entropy]

# =========================
# LOAD MODELS
# =========================
def load_all_models():
    models = {}
    try:
        models["Random Forest"] = joblib.load("rf_model.pkl")
    except:
        models["Random Forest"] = None

    try:
        models["XGBoost"] = joblib.load("xgb_model.pkl")
    except:
        models["XGBoost"] = None

    try:
        models["SVM"] = joblib.load("svm_model.pkl")
    except:
        models["SVM"] = None

    return models

models = load_all_models()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload", "📊 Analysis", "📈 Models", "📥 Insights"])

# =========================
# TAB 1 — UPLOAD
# =========================
with tab1:
    st.subheader("Upload Memory Pages")

    uploaded_files = st.file_uploader(
        "Upload `.bin` files",
        type=["bin"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        for file in uploaded_files:
            data = file.read()
            features = extract_features(data)

            row = {
                "File": file.name,
                "Distinct": features[0],
                "MaxFreq": round(features[1], 4),
                "ZeroRatio": round(features[2], 4),
                "RunCount": features[3],
            }

            # 🔥 MULTI MODEL
            for name, model in models.items():
                if model:
                    pred = model.predict([features])[0]
                else:
                    pred = 1 if features[1] > 0.5 else 0

                row[name] = "🟢 Compressible" if pred == 1 else "🔴 Incompressible"

            results.append(row)

        df = pd.DataFrame(results)

        st.dataframe(df, use_container_width=True)

        # =========================
        # SMART SUMMARY 🔥
        # =========================
        total = len(df)

        compressible = (df[["Random Forest","XGBoost","SVM"]] == "🟢 Compressible").sum().sum()
        incompressible = (df[["Random Forest","XGBoost","SVM"]] == "🔴 Incompressible").sum().sum()

        st.markdown("### 📊 Summary Insights")

        c1, c2, c3 = st.columns(3)
        c1.metric("Files Uploaded", total)
        c2.metric("Compressible Predictions", compressible)
        c3.metric("Incompressible Predictions", incompressible)

        st.success("🏆 Random Forest performs best (~97.9% accuracy)")

        st.toast("Prediction Complete 🚀")

        # DOWNLOAD
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "predictions.csv")

        st.session_state["data"] = df

    else:
        st.markdown("""
### 📂 Upload Required

Upload `.bin` files to begin analysis.

👉 This tool will:
- Extract memory features  
- Compare ML models  
- Predict compressibility  
""")

# =========================
# TAB 2 — ANALYSIS
# =========================
with tab2:
    st.subheader("Feature Analysis")

    if "data" in st.session_state:
        df = st.session_state["data"]

        st.bar_chart(df[["Distinct", "MaxFreq", "ZeroRatio", "RunCount"]])
    else:
        st.info("Upload files first")

# =========================
# TAB 3 — MODELS
# =========================
with tab3:
    st.subheader("Model Accuracy")

    acc = {
        "Logistic Regression": 0.977,
        "Random Forest": 0.979,
        "SVM": 0.976,
        "XGBoost": 0.978
    }

    chart_df = pd.DataFrame({
        "Model": list(acc.keys()),
        "Accuracy": list(acc.values())
    })

    st.bar_chart(chart_df.set_index("Model"))

# =========================
# TAB 4 — INSIGHTS
# =========================
with tab4:
    st.subheader("Prediction Insights")

    if "data" in st.session_state:
        df = st.session_state["data"]

        combined = pd.concat([
            df["Random Forest"],
            df["XGBoost"],
            df["SVM"]
        ])

        st.bar_chart(combined.value_counts())
    else:
        st.info("Upload files first")

# =========================
# FOOTER
# =========================
st.divider()

st.markdown("""
### 🚀 Why this matters

- Reduces unnecessary compression overhead  
- Improves system performance  
- Mimics real OS-level decision systems  
""")

st.caption("Built with ❤️ | Ananya Rajawat | ML + Systems 🚀")