import pandas as pd
import joblib
from pathlib import Path
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="ADHIS - Privacy-First AI Device Intelligence",
    layout="wide",
)

# ---------------------------
# Safe Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # ai-device-health/
DATA_PATH = BASE_DIR / "data" / "processed" / "telemetry_with_explanations.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "battery_model.pkl"

# ---------------------------
# Header
# ---------------------------
st.title("Privacy-First AI Device Intelligence Platform")
st.caption("Battery prediction • Anomaly detection • Usage segmentation • GenAI-style explanations")

# ---------------------------
# Validate Data
# ---------------------------
if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# ---------------------------
# Train or Load Model (FIXED)
# ---------------------------
FEATURES = ["battery_cycles", "avg_temp", "screen_on_time", "fast_charging_count", "cpu_usage"]
TARGET = "battery_health"

@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    # train/validation split (simple)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def load_or_train_model():
    # If model exists locally, load it
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "Loaded saved model"

    # Otherwise train a fresh model (works on Streamlit Cloud too)
    model = train_model(df)

    # Try saving locally (works on your laptop; cloud may not persist)
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        status = "Trained new model and saved locally"
    except Exception:
        status = "Trained new model (not saved in this environment)"

    return model, status

model, model_status = load_or_train_model()
st.sidebar.success(model_status)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Controls")

device_id = st.sidebar.number_input(
    "Select device_id",
    min_value=int(df["device_id"].min()),
    max_value=int(df["device_id"].max()),
    value=int(df["device_id"].min()),
    step=1
)

temp_threshold = st.sidebar.slider("Overheating threshold (°C)", 35, 55, 42)
battery_threshold = st.sidebar.slider("Low battery health threshold", 50, 90, 70)

# ---------------------------
# System Overview
# ---------------------------
st.subheader("System Overview")

total_devices = len(df)
total_anomalies = int((df["anomaly_flag"] == -1).sum()) if "anomaly_flag" in df.columns else 0
avg_health = float(df["battery_health"].mean())

k1, k2, k3 = st.columns(3)
k1.metric("Total Devices", total_devices)
k2.metric("Anomalies Detected", total_anomalies)
k3.metric("Average Battery Health", f"{avg_health:.2f}")

st.divider()

# ---------------------------
# Device Lookup + Prediction
# ---------------------------
row = df[df["device_id"] == device_id].iloc[0]

X_pred = pd.DataFrame([{
    "battery_cycles": row["battery_cycles"],
    "avg_temp": row["avg_temp"],
    "screen_on_time": row["screen_on_time"],
    "fast_charging_count": row["fast_charging_count"],
    "cpu_usage": row["cpu_usage"],
}])

predicted_health = float(model.predict(X_pred)[0])

anomaly_flag = int(row.get("anomaly_flag", 1))
is_overheating = (anomaly_flag == -1) and (row["avg_temp"] > temp_threshold)
is_low_battery = predicted_health < battery_threshold

cluster_name = row.get("cluster_name", "Unknown")
explanation = row.get("genai_explanation", "No explanation available.")

# ---------------------------
# Main Cards
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
    st.write(f"**Anomaly flag:** {anomaly_flag} (1 = normal, -1 = anomaly)")
    st.write(f"**Overheating risk:** {'✅ Yes' if is_overheating else '❌ No'}")
    st.write(f"**Low battery risk:** {'✅ Yes' if is_low_battery else '❌ No'}")

st.divider()

# ---------------------------
# Telemetry Snapshot
# ---------------------------
st.subheader("Telemetry Snapshot")
snapshot_cols = ["battery_cycles", "avg_temp", "screen_on_time", "fast_charging_count", "cpu_usage", "battery_health"]
st.dataframe(pd.DataFrame([row[snapshot_cols]]), use_container_width=True)

# ---------------------------
# GenAI Explanation
# ---------------------------
st.subheader("GenAI Explanation")
st.info(explanation)

st.divider()

# ---------------------------
# Cluster Distribution
# ---------------------------
st.subheader("Cluster Distribution")
if "cluster_name" in df.columns:
    st.bar_chart(df["cluster_name"].value_counts())
else:
    st.warning("cluster_name not found in dataset.")

st.divider()

# ---------------------------
# Quick View: At-Risk Devices (sample)
# ---------------------------
st.subheader("Quick View: At-Risk Devices (sample)")

risk_df = df.copy()
risk_df["predicted_battery_health"] = model.predict(risk_df[FEATURES])

risk_df["overheating_risk"] = (
    (risk_df.get("anomaly_flag", 1) == -1)
    & (risk_df["avg_temp"] > temp_threshold)
)
risk_df["low_battery_risk"] = risk_df["predicted_battery_health"] < battery_threshold

top_risk = risk_df[(risk_df["overheating_risk"]) | (risk_df["low_battery_risk"])][
    ["device_id", "cluster_name", "avg_temp", "cpu_usage", "predicted_battery_health", "overheating_risk", "low_battery_risk"]
].head(20)

st.dataframe(top_risk, use_container_width=True)
st.caption("Adjust thresholds in sidebar to see real-time risk updates.")
