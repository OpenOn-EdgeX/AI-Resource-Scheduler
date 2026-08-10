"""
모델별 실행 이력 저장소 — SQLite (가벼운 시작; 복잡한 DB 는 추후).

역할 2: KETRIS 실측 결과(kernels_per_iter, resource_bound_type, mps_pct, mode,
GPU util/throughput 통계)를 model_id 키로 upsert 저장, 같은 모델 재유입 시 조회.
이력 없으면 404 → PPO 쪽이 lsu_est 로 폴백 (State ⑤그룹 설계).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_history (
    model_id            TEXT PRIMARY KEY,
    kernels_per_iter    INTEGER,
    resource_bound_type TEXT,
    mps_pct             REAL,
    mode                TEXT,
    avg_gpu_util        REAL,
    avg_throughput      REAL,
    run_count           INTEGER DEFAULT 0,
    updated_at          TEXT,
    raw_json            TEXT
);
"""

_KNOWN = ("kernels_per_iter", "resource_bound_type", "mps_pct", "mode")


class ModelHistoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, model_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM model_history WHERE model_id=?",
                            (model_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["raw"] = json.loads(d.pop("raw_json") or "{}")
        return d

    def upsert(self, model_id: str, data: dict) -> dict:
        """
        KETRIS 실측 결과 반영 (3단계 피드백 수신 지점).
        gpu_util/throughput 은 run_count 기반 이동평균으로 누적.
        """
        with self._lock, self._conn() as c:
            prev = c.execute("SELECT * FROM model_history WHERE model_id=?",
                             (model_id,)).fetchone()
            n = (prev["run_count"] if prev else 0) or 0

            def avg(prev_v, new_v):
                if new_v is None:
                    return prev_v
                if prev_v is None or n == 0:
                    return float(new_v)
                return (prev_v * n + float(new_v)) / (n + 1)

            fields = {k: data.get(k, prev[k] if prev else None) for k in _KNOWN}
            avg_util = avg(prev["avg_gpu_util"] if prev else None, data.get("gpu_util"))
            avg_thr = avg(prev["avg_throughput"] if prev else None, data.get("throughput"))
            raw = json.loads(prev["raw_json"]) if prev and prev["raw_json"] else {}
            raw.update({k: v for k, v in data.items() if k not in _KNOWN})

            c.execute(
                """INSERT INTO model_history
                   (model_id, kernels_per_iter, resource_bound_type, mps_pct, mode,
                    avg_gpu_util, avg_throughput, run_count, updated_at, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(model_id) DO UPDATE SET
                     kernels_per_iter=excluded.kernels_per_iter,
                     resource_bound_type=excluded.resource_bound_type,
                     mps_pct=excluded.mps_pct, mode=excluded.mode,
                     avg_gpu_util=excluded.avg_gpu_util,
                     avg_throughput=excluded.avg_throughput,
                     run_count=excluded.run_count, updated_at=excluded.updated_at,
                     raw_json=excluded.raw_json""",
                (model_id, fields["kernels_per_iter"], fields["resource_bound_type"],
                 fields["mps_pct"], fields["mode"], avg_util, avg_thr, n + 1,
                 datetime.now().isoformat(timespec="seconds"), json.dumps(raw)))
        return self.get(model_id)

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM model_history").fetchone()[0]
