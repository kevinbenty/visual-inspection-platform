"""SQLite inspection history. Swap for PostgreSQL later by replacing this module."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    model_version TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    threshold REAL NOT NULL,
    is_anomalous INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
"""


class InspectionStore:
    def __init__(self, path: str):
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, filename: str, r: dict) -> dict:
        created = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO inspections (filename, created_at, category, model_version,"
                " anomaly_score, threshold, is_anomalous, latency_ms) VALUES (?,?,?,?,?,?,?,?)",
                (filename, created, r["category"], r["model_version"], r["anomaly_score"],
                 r["threshold"], int(r["is_anomalous"]), r["latency_ms"]))
            return {"id": cur.lastrowid, "filename": filename, "created_at": created}

    def list(self, limit: int = 50, only_anomalous: bool = False) -> list[dict]:
        q = "SELECT * FROM inspections"
        if only_anomalous:
            q += " WHERE is_anomalous = 1"
        q += " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, (limit,)).fetchall()
        return [{**dict(r), "is_anomalous": bool(r["is_anomalous"])} for r in rows]
