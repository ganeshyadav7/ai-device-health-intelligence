import pandas as pd
import joblib
from pathlib import Path
import streamlit as st

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="ADHIS - Privacy-First AI Device Intelligence",
    layout="wide",
)

# ---------------------------
# Safe Path Handling (FIXED)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # ai-device-health/
DATA_PATH = BASE_DIR / "data" / "processed" / "telemetry_with_explanations.csv"
MODEL_PATH = BASE_DIR / "models" / "battery_model.pkl"

# ---------------------------
# App Header
# ---------------------------
st.title("Privacy-First AI Device Intelligence Platform")
st.caption("Battery prediction • Anomaly detection • Usage segmentation • GenAI-style explanations")

# ---------------------------
# Validate Files
# ---------------------------
if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

# ---------------------------
# Load Data & Model
# ---------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Controls")

device_id = st.sidebar.number_input(
    "Select device_id",
    min_value=int(df["device_id"].min()),
    max_value=int(df["device_id"].max()),
    value=int(df["device_id"].min()),
)

temp_threshold = st.sidebar.slider(
    "Overheating threshold (°C)",
    35, 55, 42
)

battery_threshold = st.sidebar.slider(
    "Low battery health threshold",
    50, 90, 70
)

# ---------------------------
# KPI Overview
# ---------------------------
st.subheader("System Overview")

total_devices = len(df)
total_anomalies = int((df["anomaly_flag"] == -1).sum()) if "anomaly_flag" in df.columns else 0
avg_health = df["battery_health"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("Total Devices", total_devices)
k2.metric("Anomalies Detected", total_anomalies)
k3.metric("Average Battery Health", f"{avg_health:.2f}")

st.divider()

# ---------------------------
# Selected Device
# ---------------------------
row = df[df["device_id"] == device_id].iloc[0]

# Prepare prediction input
X_pred = pd.DataFrame([{
    "battery_cycles": row["battery_cycles"],
    "avg_temp": row["avg_temp"],
    "screen_on_time": row["screen_on_time"],
    "fast_charging_count": row["fast_charging_count"],
    "cpu_usage": row["cpu_usage"],
}])

predicted_health = float(model.predict(X_pred)[0])

# Risk flags
anomaly_flag = row.get("anomaly_flag", 1)
is_overheating = (anomaly_flag == -1) and (row["avg_temp"] > temp_threshold)
is_low_battery = predicted_health < battery_threshold
cluster_name = row.get("cluster_name", "Unknown")
explanation = row.get("genai_explanation", "No explanation available.")

# ---------------------------
# Main Display Row
# ---------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Predicted Battery Health")
    st.metric("Prediction", f"{predicted_health:.2f}")

with c2:
    st.subheader("Segment")
    st.metric("Cluster", cluster_name)

with c3:
    st.subheader("Risk Status")
    st.write(f"Anomaly flag: {anomaly_flag} (1 = normal, -1 = anomaly)")
    st.write(f"Overheating risk: {'✅ Yes' if is_overheating else '❌ No'}")
    st.write(f"Low battery risk: {'✅ Yes' if is_low_battery else '❌ No'}")

st.divider()

# ---------------------------
# Telemetry Snapshot
# ---------------------------
st.subheader("Telemetry Snapshot")

snapshot_cols = [
    "battery_cycles",
    "avg_temp",
    "screen_on_time",
    "fast_charging_count",
    "cpu_usage",
    "battery_health",
]

st.dataframe(pd.DataFrame([row[snapshot_cols]]), use_container_width=True)

# ---------------------------
# GenAI Explanation
# ---------------------------
st.subheader("GenAI Explanation")
st.info(explanation)

st.divider()

# ---------------------------
# Cluster Distribution Chart
# ---------------------------
st.subheader("Cluster Distribution")

if "cluster_name" in df.columns:
    st.bar_chart(df["cluster_name"].value_counts())

st.divider()

# ---------------------------
# Quick View: At-Risk Devices
# ---------------------------
st.subheader("Quick View: At-Risk Devices (sample)")

risk_df = df.copy()

# Predict for all rows (demo approach)
def predict_row(r):
    Xr = pd.DataFrame([{
        "battery_cycles": r["battery_cycles"],
        "avg_temp": r["avg_temp"],
        "screen_on_time": r["screen_on_time"],
        "fast_charging_count": r["fast_charging_count"],
        "cpu_usage": r["cpu_usage"],
    }])
    return float(model.predict(Xr)[0])

risk_df["predicted_battery_health"] = risk_df.apply(predict_row, axis=1)

risk_df["overheating_risk"] = (
    (risk_df.get("anomaly_flag", 1) == -1)
    & (risk_df["avg_temp"] > temp_threshold)
)

risk_df["low_battery_risk"] = (
    risk_df["predicted_battery_health"] < battery_threshold
)

top_risk = risk_df[
    (risk_df["overheating_risk"]) | (risk_df["low_battery_risk"])
][[
    "device_id",
    "cluster_name",
    "avg_temp",
    "cpu_usage",
    "predicted_battery_health",
    "overheating_risk",
    "low_battery_risk"
]].head(20)

st.dataframe(top_risk, use_container_width=True)

st.caption("Adjust thresholds in sidebar to see real-time risk changes.")