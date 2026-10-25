"""Inspection dashboard. Talks to the API; holds no model itself.

Run: streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
SAMPLES_DIR = os.getenv("SAMPLES_DIR", "")
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]


class SampleImage(io.BytesIO):
    """A bundled sample image that behaves like a Streamlit upload."""

    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name
        self.type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def sample_images() -> list[Path]:
    """Sample images shipped with the demo (SAMPLES_DIR), if any."""
    if not SAMPLES_DIR or not Path(SAMPLES_DIR).is_dir():
        return []
    return sorted(p for p in Path(SAMPLES_DIR).iterdir() if p.suffix.lower().lstrip(".") in IMAGE_TYPES)

st.set_page_config(page_title="Visual inspection", layout="wide")
st.title("Visual inspection")


def api_status() -> dict | None:
    try:
        return requests.get(f"{API}/health", timeout=3).json()
    except requests.RequestException:
        return None


status = api_status()
with st.sidebar:
    if status is None:
        st.error(f"Can't reach the API at {API}. Start it with `uvicorn app.api:app`.")
    elif not status["model_loaded"]:
        st.warning("API is running but no model is loaded. Train one, then set MODEL_DIR.")
    else:
        st.success(f"Model: {status['model_version']}")
    st.caption("Scores are relative to a threshold set on held-out normal images. "
               "They are not probabilities and say nothing about defect severity.")

inspect_tab, history_tab = st.tabs(["Inspect an image", "History"])

with inspect_tab:
    upload = st.file_uploader("Image of a part", type=IMAGE_TYPES)
    samples = sample_images()
    if samples and upload is None:
        choice = st.selectbox("Or try a sample image", ["Choose a sample"] + [p.name for p in samples])
        if choice != "Choose a sample":
            upload = SampleImage(Path(SAMPLES_DIR) / choice)
    if upload and st.button("Inspect image", type="primary"):
        with st.spinner("Inspecting"):
            try:
                r = requests.post(f"{API}/inspect",
                                  files={"file": (upload.name, upload.getvalue(), upload.type)},
                                  timeout=60)
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                r = None
        if r is not None and r.ok:
            res = r.json()
            left, right = st.columns(2)
            left.image(upload, caption="Uploaded", use_container_width=True)
            if res.get("heatmap_png_base64"):
                right.image(base64.b64decode(res["heatmap_png_base64"]),
                            caption="Anomaly heatmap (red = most unusual)",
                            use_container_width=True)
            verdict = "Flagged for review" if res["is_anomalous"] else "No anomaly found"
            (st.error if res["is_anomalous"] else st.success)(verdict)
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly score", f"{res['anomaly_score']:.3f}")
            c2.metric("Threshold", f"{res['threshold']:.3f}")
            c3.metric("Latency", f"{res['latency_ms']:.0f} ms")
        elif r is not None:
            st.error(f"API returned {r.status_code}: {r.json().get('detail', r.text)}")

with history_tab:
    only_flagged = st.toggle("Only flagged images")
    try:
        rows = requests.get(f"{API}/history",
                            params={"limit": 200, "only_anomalous": only_flagged}, timeout=5).json()
    except requests.RequestException:
        rows = []
    if not rows:
        st.info("No inspections yet. Upload an image on the first tab.")
    else:
        df = pd.DataFrame(rows)
        st.metric("Flagged rate", f"{df['is_anomalous'].mean():.0%}", help=f"{len(df)} inspections")
        st.dataframe(df[["id", "created_at", "filename", "category", "anomaly_score",
                         "threshold", "is_anomalous", "latency_ms"]],
                     hide_index=True, use_container_width=True)
