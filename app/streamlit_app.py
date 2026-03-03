import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ----------------------------
# Paths (repo-relative)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent           # .../app
REPO_DIR = APP_DIR.parent                          # repo root
DATA_RAW = REPO_DIR / "data" / "raw" / "device_telemetry.csv"
DATA_EXPLAIN = REPO_DIR / "data" / "processed" / "telemetry_with_explanations.csv"

MODELS_DIR = REPO_DIR / "models"
MODEL_PATH = MODELS_DIR / "battery_model.pkl"
METRICS_PATH = MODELS_DIR / "battery_model_metrics.json"

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(
    page_title="ADHIS - Privacy-First AI Device Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def load_explanations_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_raw_telemetry(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))

def train_and_save_model(raw_df: pd.DataFrame) -> dict:
    """
    Train RF model and save it locally. Also return + save metrics.
    """
    target = "battery_health"
    drop_cols = ["device_id", target]

    X = raw_df.drop(columns=drop_cols)
    y = raw_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    # Save model locally (DO NOT upload this to GitHub)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "n_records": int(raw_df.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "feature_importance": {k: float(v) for k, v in importances.to_dict().items()},
        "model_file": str(MODEL_PATH.as_posix()),
    }
    save_metrics(METRICS_PATH, metrics)
    st.sidebar.write("Metrics saved at:", str(METRICS_PATH))
st.sidebar.write("Exists now?", METRICS_PATH.exists())
    return metrics

def get_prediction_from_model(model, row: pd.Series) -> float:
    feature_cols = ["battery_cycles", "avg_temp", "screen_on_time", "fast_charging_count", "cpu_usage"]
    X_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    return float(model.predict(X_row)[0])

# ----------------------------
# UI Header
# ----------------------------
st.title("Privacy-First AI Device Intelligence Platform")
st.caption("Battery prediction • Anomaly detection • Usage segmentation • GenAI-style explanations")

# ----------------------------
# Load data (processed explanations)
# ----------------------------
if not DATA_EXPLAIN.exists():
    st.error(f"Missing data file: {DATA_EXPLAIN}")
    st.stop()

df = load_explanations_csv(DATA_EXPLAIN)

# Basic validation
required_cols = {"device_id", "battery_health", "avg_temp", "cluster_name", "genai_explanation"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns in {DATA_EXPLAIN}: {missing}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

device_ids = df["device_id"].astype(int).sort_values().unique().tolist()
default_device = device_ids[0] if device_ids else 1

selected_device = st.sidebar.selectbox("Select device_id", device_ids, index=0)

overheat_threshold = st.sidebar.slider("Overheating threshold (°C)", 35, 55, 42)
low_battery_threshold = st.sidebar.slider("Low battery health threshold", 50, 90, 70)

st.sidebar.divider()

# Retrain button (manual)
st.sidebar.subheader("Model Management")
retrain_clicked = st.sidebar.button("🔁 Retrain Model (Random Forest)")

# ----------------------------
# Retrain model logic
# ----------------------------
if retrain_clicked:
    if not DATA_RAW.exists():
        st.sidebar.error(f"Missing raw dataset: {DATA_RAW}")
    else:
        raw_df = load_raw_telemetry(DATA_RAW)
        metrics = train_and_save_model(raw_df)
        st.sidebar.success("✅ Trained new model and saved locally")
        st.sidebar.write(f"RMSE: **{metrics['rmse']:.3f}**")
        st.sidebar.write(f"R²: **{metrics['r2']:.3f}**")

# ----------------------------
# Load metrics (if exists)
# ----------------------------
metrics = load_metrics(METRICS_PATH)

# Try loading model if exists
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

# ----------------------------
# System overview KPIs
# ----------------------------
total_devices = int(df["device_id"].nunique())
avg_battery = float(df["battery_health"].mean())

# Some anomaly logic (if anomaly_flag exists else derive from temp)
if "anomaly_flag" in df.columns:
    anomalies_detected = int((df["anomaly_flag"] == -1).sum())
else:
    anomalies_detected = int((df["avg_temp"] > overheat_threshold).sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("System Overview")
    st.metric("Total Devices", total_devices)
with c2:
    st.subheader(" ")
    st.metric("Anomalies Detected", anomalies_detected)
with c3:
    st.subheader(" ")
    st.metric("Average Battery Health", f"{avg_battery:.2f}")

st.divider()

# ----------------------------
# Selected device view
# ----------------------------
row = df.loc[df["device_id"] == selected_device].iloc[0]

left, mid, right = st.columns(3)

# Prediction card
with left:
    st.subheader("Predicted Battery Health")
    if model is not None:
        pred = get_prediction_from_model(model, row)
        st.metric("Prediction", f"{pred:.2f}")
    else:
        st.metric("Prediction", "Not trained")
        st.caption("Click **Retrain Model** in the sidebar to generate predictions.")

# Segment card
with mid:
    st.subheader("Segment")
    st.write("Cluster")
    st.markdown(f"### {row.get('cluster_name', 'Unknown')}")

# Risk status card
with right:
    st.subheader("Risk Status")

    # anomaly
    if "anomaly_flag" in df.columns:
        anomaly_ok = int(row["anomaly_flag"]) == 1
        anomaly_text = "✅ Normal" if anomaly_ok else "⚠️ Anomaly"
    else:
        anomaly_ok = float(row["avg_temp"]) <= overheat_threshold
        anomaly_text = "✅ Normal" if anomaly_ok else "⚠️ Overheating"

    overheat_risk = float(row["avg_temp"]) > overheat_threshold
    low_battery_risk = float(row["battery_health"]) < low_battery_threshold

    st.write(f"Anomaly: **{anomaly_text}**")
    st.write(f"Overheating risk: {'❌ Yes' if overheat_risk else '✅ No'}")
    st.write(f"Low battery risk: {'❌ Yes' if low_battery_risk else '✅ No'}")

st.divider()

# Telemetry snapshot
st.subheader("Telemetry Snapshot")
show_cols = [c for c in ["battery_cycles","avg_temp","screen_on_time","fast_charging_count","cpu_usage","battery_health"] if c in df.columns]
st.dataframe(pd.DataFrame([row[show_cols].to_dict()]))

# GenAI explanation
st.subheader("GenAI Explanation")
st.info(str(row.get("genai_explanation", "No explanation available.")))

st.divider()

# ----------------------------
# Model Performance Metrics Section
# ----------------------------
st.subheader("Model Performance (Recruiter-ready)")
if metrics is None:
    st.warning("No saved metrics found yet. Click **Retrain Model** to generate RMSE + R².")
else:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("RMSE", f"{metrics['rmse']:.3f}")
    with m2:
        st.metric("R²", f"{metrics['r2']:.3f}")
    with m3:
        st.metric("Training Records", metrics.get("n_records", "N/A"))

    st.write("**Feature Importance**")
    fi = metrics.get("feature_importance", {})
    if fi:
        fi_series = pd.Series(fi).sort_values(ascending=False)
        st.bar_chart(fi_series)
    else:
        st.caption("No feature importance saved.")

st.divider()

# ----------------------------
# Quick view table (at-risk sample)
# ----------------------------
st.subheader("Quick View: At-Risk Devices (sample)")

tmp = df.copy()
tmp["overheating_risk"] = tmp["avg_temp"] > overheat_threshold
tmp["low_battery_risk"] = tmp["battery_health"] < low_battery_threshold

# Optional predicted col if model exists
if model is not None:
    preds = []
    for _, r in tmp.iterrows():
        try:
            preds.append(get_prediction_from_model(model, r))
        except Exception:
            preds.append(np.nan)
    tmp["predicted_battery_health"] = preds

cols_to_show = ["device_id", "cluster_name", "avg_temp", "cpu_usage", "battery_health", "overheating_risk", "low_battery_risk"]
if "predicted_battery_health" in tmp.columns:
    cols_to_show.insert(5, "predicted_battery_health")

st.dataframe(
    tmp.loc[(tmp["overheating_risk"] | tmp["low_battery_risk"]), cols_to_show]
      .head(20)
      .reset_index(drop=True)
)
