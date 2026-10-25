from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PeakLocation(BaseModel):
    x: int
    y: int
    frame: str


class InspectionResult(BaseModel):
    id: int
    filename: str
    created_at: datetime
    category: str
    model_version: str
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    score_ratio: float | None = Field(None, description="score / threshold; not a probability")
    latency_ms: float
    peak_location: PeakLocation | None = None
    heatmap_png_base64: str | None = None


class HistoryItem(BaseModel):
    id: int
    filename: str
    created_at: datetime
    category: str
    model_version: str
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    latency_ms: float


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
