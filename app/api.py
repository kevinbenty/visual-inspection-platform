"""FastAPI service.

Run:  uvicorn app.api:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile

from app.db import InspectionStore
from app.schemas import Health, HistoryItem, InspectionResult
from src.preprocessing import InvalidImageError, load_image_bytes

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("vip.api")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 2**20
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/tiff"}

app = FastAPI(title="Visual Inspection API", version="0.1.0",
              description="Image-level anomaly scoring and localisation for MVTec AD categories.")


@lru_cache
def get_inspector():
    from app.inference import Inspector

    model_dir = os.getenv("MODEL_DIR", "models/patchcore/bottle")
    log.info("Loading model from %s", model_dir)
    return Inspector(model_dir, device=os.getenv("DEVICE", "cpu"))


@lru_cache
def get_store() -> InspectionStore:
    return InspectionStore(os.getenv("DB_PATH", "inspections.db"))


@app.get("/health", response_model=Health)
def health():
    try:
        insp = get_inspector()
        return Health(status="ok", model_loaded=True, model_version=insp.model_version)
    except Exception as exc:  # report, don't crash the probe
        log.warning("Model not loaded: %s", exc)
        return Health(status="degraded", model_loaded=False)


@app.post("/inspect", response_model=InspectionResult)
async def inspect(file: UploadFile = File(...), include_heatmap: bool = True,
                  inspector=Depends(get_inspector), store: InspectionStore = Depends(get_store)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported content type {file.content_type}.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // 2**20} MB.")
    try:
        img = load_image_bytes(data)
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc)) from exc

    result = inspector.predict(img, with_overlay=include_heatmap)
    row = store.add(file.filename or "upload", result)
    log.info("inspected %s score=%.3f anomalous=%s %.1fms", row["filename"],
             result["anomaly_score"], result["is_anomalous"], result["latency_ms"])
    return InspectionResult(**row, **result)


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = Query(50, ge=1, le=500), only_anomalous: bool = False,
            store: InspectionStore = Depends(get_store)):
    return store.list(limit, only_anomalous)
