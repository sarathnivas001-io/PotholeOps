"""
dashboard/app.py

Streamlit dashboard with two tabs:
  1. Live Inference — upload a pothole image, call the FastAPI /predict endpoint
  2. Drift Monitoring — show the latest Evidently drift report + summary

Run: streamlit run dashboard/app.py
Expects the FastAPI service running at API_URL (default http://localhost:8000).
"""
import json
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://localhost:8000"
REPORTS_DIR = Path("reports")

st.set_page_config(page_title="PotholeOps Dashboard", layout="wide")
st.title("🕳️ PotholeOps — Pothole Severity Monitoring")

tab_infer, tab_drift = st.tabs(["Live Inference", "Drift Monitoring"])

with tab_infer:
    st.subheader("Upload a road image for severity classification")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Calling inference API..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=15,
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.metric("Predicted severity", result["prediction"].upper())
                    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                    st.bar_chart(result["class_probabilities"])
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not reach the API at {API_URL}. "
                             f"Start it with: uvicorn src.api.main:app --port 8000")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

with tab_drift:
    st.subheader("Prediction drift vs. validation baseline")
    st.caption("Generate a fresh report with: `python src/monitoring/drift_report.py`")

    summary_path = REPORTS_DIR / "drift_summary.json"
    report_path = REPORTS_DIR / "drift_report.html"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        drift_flag = summary["drift_detected"]
        st.error("⚠️ Drift detected") if drift_flag else st.success("✅ No significant drift")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Baseline (validation set) distribution**")
            st.bar_chart(summary["reference_counts"])
        with col2:
            st.write("**Current (live) prediction distribution**")
            st.bar_chart(summary["current_counts"])
    else:
        st.info("No drift report yet. Run the drift monitoring script first.")

    if report_path.exists():
        with st.expander("Full Evidently AI report"):
            st.components.v1.html(report_path.read_text(), height=800, scrolling=True)
