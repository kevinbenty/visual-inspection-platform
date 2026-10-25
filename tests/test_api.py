"""API tests use a fake inspector, so they run without model weights."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import api  # noqa: E402
from app.db import InspectionStore  # noqa: E402


class FakeInspector:
    model_version = "fake-bottle"

    def predict(self, img, with_overlay=True):
        return {"anomaly_score": 2.0, "threshold": 1.0, "is_anomalous": True,
                "score_ratio": 2.0, "category": "bottle", "model_version": self.model_version,
                "latency_ms": 1.0, "peak_location": None, "heatmap_png_base64": None}


@pytest.fixture
def client(tmp_path):
    store = InspectionStore(str(tmp_path / "t.db"))
    api.app.dependency_overrides[api.get_inspector] = lambda: FakeInspector()
    api.app.dependency_overrides[api.get_store] = lambda: store
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_inspect_and_history(client, png_bytes):
    r = client.post("/inspect", files={"file": ("a.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_anomalous"] is True and body["filename"] == "a.png"
    hist = client.get("/history").json()
    assert len(hist) == 1 and hist[0]["id"] == body["id"]
    assert client.get("/history", params={"only_anomalous": True}).json()


def test_rejects_wrong_content_type(client):
    r = client.post("/inspect", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/inspect", files={"file": ("a.png", b"garbage", "image/png")})
    assert r.status_code == 422


def test_history_limit_validation(client):
    assert client.get("/history", params={"limit": 0}).status_code == 422
