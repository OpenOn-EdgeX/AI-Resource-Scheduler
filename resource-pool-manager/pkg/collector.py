"""
실시간 노드 자원 수집기 — 백그라운드 스레드, 기본 5초 주기.

수집: GPU util / 메모리 (nvidia-smi). physical_sm_total 은 기동 시 1회
torch 로 감지(폴백 env PHYSICAL_SM_TOTAL). PPO 요청 경로에는 최신 스냅샷만
제공하므로 nvidia-smi 지연이 응답 지연으로 전파되지 않는다.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


def detect_physical_sm() -> int:
    """물리 SM 수 감지: torch(정확) → env → 기본값 순."""
    env = os.environ.get("PHYSICAL_SM_TOTAL")
    if env:
        return int(env)
    try:
        import torch
        if torch.cuda.is_available():
            sm = torch.cuda.get_device_properties(0).multi_processor_count
            logger.info(f"physical_sm_total={sm} (torch 감지)")
            return int(sm)
    except Exception as e:
        logger.warning(f"torch SM 감지 실패: {e}")
    return 148  # 폴백 (RTX PRO 6000 Blackwell 급 근사치, env 로 교정 권장)


class NodeCollector:
    def __init__(self, interval_s: float = 5.0):
        self.interval = interval_s
        self._latest = {"gpu_util": 0.0, "mem_util": 0.0, "mem_used_mb": 0,
                        "mem_total_mb": 0, "collected_at": 0.0, "nvidia_smi_ok": False}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _collect_once(self) -> None:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                util, used, total = [float(x) for x in r.stdout.strip().split("\n")[0].split(",")]
                self._latest = {
                    "gpu_util": util / 100.0,
                    "mem_used_mb": int(used), "mem_total_mb": int(total),
                    "mem_util": used / total if total else 0.0,
                    "collected_at": time.time(), "nvidia_smi_ok": True,
                }
                return
        except Exception as e:
            logger.warning(f"nvidia-smi 수집 실패: {e}")
        self._latest = {**self._latest, "collected_at": time.time(), "nvidia_smi_ok": False}

    def _loop(self):
        while not self._stop.is_set():
            self._collect_once()
            self._stop.wait(self.interval)

    def start(self):
        self._collect_once()  # 기동 직후 1회 즉시
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def latest(self) -> dict:
        return dict(self._latest)
