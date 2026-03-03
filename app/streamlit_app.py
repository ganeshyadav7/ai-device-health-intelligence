import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="ADHIS - Privacy-First AI Device Intelligence",
    page_icon="🔋",
    layout="wide",
)

# -----------------------------
# Paths (works locally + Streamlit Cloud)
# app/streamlit_app.py  -> project root is parent of app/
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

DATA_RAW = ROOT_DIR / "data" / "raw" / "device_telemetry.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "telemetry_with_explanations.csv"
METRICS_PATH = ROOT_DIR / "models" / "battery_model_metrics.json"

# -----------------------------
# Styling (simple dark look)
# -----------------------------
st.markdown(
    """
    <style>
      .small-subtitle {opacity: 0.75; margin-top: -10px;}
      .kpi-card {padding: 14px; border-radius: 14px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);}
      .muted {opacity:0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_processed_data():
    if DATA_PROCESSED.exists():
        return pd.read_csv(DATA_PROCESSED)
    # fallback to raw if processed is missing
    if DATA_RAW.exists():
        df = pd.read_csv(DATA_RAW)
        # minimal columns so app still works
        if "cluster_name" not in df.columns:
            df["cluster_name"] = "Unknown"
        if "anomaly_flag" not in df.columns:
            df["anomaly_flag"] = 1
        if "genai_explanation" not in df.columns:
            df["genai_explanation"] = "No explanation file found yet. Run Notebook 06."
        df["predicted_battery_health"] = df.get("battery_health", np.nan)
        return df
    return None


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def risk_flags(row, overheating_threshold=42.0, low_batt_threshold=70.0):
    # anomaly_flag: 1=normal, -1=anomaly (as per your notebook)
    overheating_risk = False
    low_battery_risk = False

    avg_temp = float(row.get("avg_temp", 0))
    pred = row.get("predicted_battery_health", row.get("battery_health", np.nan))
    anomaly_flag = int(row.get("anomaly_flag", 1))

    if anomaly_flag == -1 and avg_temp > overheating_threshold:
        overheating_risk = True

    if pd.notna(pred) and float(pred) < low_batt_threshold:
        low_battery_risk = True

    return overheating_risk, low_battery_risk


# -----------------------------
# Load data
# -----------------------------
df = load_processed_data()
metrics = load_metrics()

st.title("Privacy-First AI Device Intelligence Platform")
st.markdown(
    '<div class="small-subtitle">Battery prediction • Anomaly detection • Usage segmentation • GenAI-style explanations</div>',
    unsafe_allow_html=True,
)

if df is None:
    st.error("No dataset found. Please add: data/raw/device_telemetry.csv (and/or processed files).")
    st.stop()

# Ensure columns exist for consistent UI
if "predicted_battery_health" not in df.columns:
    # If you didn't store predictions in processed file, we fallback to battery_health
    df["predicted_battery_health"] = df.get("battery_health", np.nan)

if "cluster_name" not in df.columns:
    df["cluster_name"] = "Unknown"

if "anomaly_flag" not in df.columns:
    df["anomaly_flag"] = 1

if "genai_explanation" not in df.columns:
    df["genai_explanation"] = "Explanation not found. Run Notebook 06 to generate telemetry_with_explanations.csv"

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

device_ids = sorted(df["device_id"].unique().tolist()) if "device_id" in df.columns else list(range(1, len(df) + 1))
default_device = device_ids[0] if device_ids else 1

selected_device = st.sidebar.selectbox("Select device_id", device_ids, index=0)

overheating_threshold = st.sidebar.slider("Overheating threshold (°C)", 35, 55, 42)
low_battery_threshold = st.sidebar.slider("Low battery health threshold", 50, 90, 70)

# -----------------------------
# Selected row
# -----------------------------
row = df[df["device_id"] == selected_device].iloc[0] if "device_id" in df.columns else df.iloc[0]
overheat_risk, low_batt_risk = risk_flags(row, overheating_threshold, low_battery_threshold)

# -----------------------------
# System overview KPIs
# -----------------------------
total_devices = int(df["device_id"].nunique()) if "device_id" in df.columns else int(df.shape[0])
anomalies_detected = int((df["anomaly_flag"] == -1).sum()) if "anomaly_flag" in df.columns else 0
avg_battery = float(df["predicted_battery_health"].mean()) if "predicted_battery_health" in df.columns else float(df["battery_health"].mean())

st.subheader("System Overview")
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown('<div class="kpi-card">Total Devices<br><h2 style="margin:0;">{}</h2></div>'.format(total_devices), unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card">Anomalies Detected<br><h2 style="margin:0;">{}</h2></div>'.format(anomalies_detected), unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi-card">Average Battery Health<br><h2 style="margin:0;">{:.2f}</h2></div>'.format(avg_battery), unsafe_allow_html=True)

st.divider()

# -----------------------------
# ✅ NEW: Model Performance (RMSE + R²)
# -----------------------------
st.subheader("Model Performance (Regression)")

if metrics is None:
    st.warning("Metrics file not found: models/battery_model_metrics.json\n\nRun Notebook 03 and make sure the file is saved & pushed to GitHub.")
else:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
    with m2:
        st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    with m3:
        st.metric("Records used", f"{metrics.get('n_records', 'NA')}")
    with m4:
        st.metric("Features used", f"{metrics.get('n_features', 'NA')}")

    st.caption(
        "RMSE lower is better. R² closer to 1.0 is better. "
        "These metrics are saved from Notebook 03 and displayed here for recruiters."
    )

st.divider()

# -----------------------------
# Main KPI area: Prediction / Segment / Risk
# -----------------------------
c1, c2, c3 = st.columns(3)

pred_val = float(row.get("predicted_battery_health", row.get("battery_health", np.nan)))
cluster_val = str(row.get("cluster_name", "Unknown"))
anomaly_flag = int(row.get("anomaly_flag", 1))

with c1:
    st.markdown("### Predicted Battery Health")
    st.metric("Prediction", f"{pred_val:.2f}")

with c2:
    st.markdown("### Segment")
    st.metric("Cluster", cluster_val)

with c3:
    st.markdown("### Risk Status")
    st.write(f"**Anomaly flag:** {anomaly_flag} (1 = normal, -1 = anomaly)")
    st.write(f"**Overheating risk:** {'✅ Yes' if overheat_risk else '❌ No'}")
    st.write(f"**Low battery risk:** {'✅ Yes' if low_batt_risk else '❌ No'}")

st.divider()

# -----------------------------
# Telemetry Snapshot
# -----------------------------
st.subheader("Telemetry Snapshot")
snap_cols = ["battery_cycles", "avg_temp", "screen_on_time", "fast_charging_count", "cpu_usage", "battery_health"]
snap_cols = [c for c in snap_cols if c in df.columns]
st.dataframe(pd.DataFrame([row[snap_cols].to_dict()]), use_container_width=True)

# -----------------------------
# GenAI Explanation
# -----------------------------
st.subheader("GenAI Explanation")
st.info(str(row.get("genai_explanation", "No explanation available.")))

st.divider()

# -----------------------------
# Quick View: At-Risk Devices
# -----------------------------
st.subheader("Quick View: At-Risk Devices (sample)")

tmp = df.copy()
tmp["overheating_risk"] = tmp.apply(lambda r: risk_flags(r, overheating_threshold, low_battery_threshold)[0], axis=1)
tmp["low_battery_risk"] = tmp.apply(lambda r: risk_flags(r, overheating_threshold, low_battery_threshold)[1], axis=1)

at_risk = tmp[(tmp["overheating_risk"] == True) | (tmp["low_battery_risk"] == True)]
cols_show = ["device_id", "cluster_name", "avg_temp", "cpu_usage", "predicted_battery_health", "overheating_risk", "low_battery_risk"]
cols_show = [c for c in cols_show if c in at_risk.columns]

st.dataframe(at_risk[cols_show].head(15), use_container_width=True)

st.divider()

# -----------------------------
# Cluster distribution (simple)
# -----------------------------
st.subheader("Cluster Distribution")

cluster_counts = df["cluster_name"].value_counts(dropna=False)
st.bar_chart(cluster_counts)

st.caption("✅ If you want, next we can add: retrain button, PDF report, Docker, CI/CD badge, architecture diagram.")
