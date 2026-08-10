"""
KETRIS 실측 → 모델 이력 피드백 루프 (3단계 ③).

KETRIS 모니터링 모듈(MetricsLogger)을 임포트해 shm 스냅샷을 주기 폴링,
활성 할당(pod_name→model_id 매핑)의 실행 통계를 이력 저장소에 upsert 한다.

- MetricsLogger 인스턴스를 유지해야 구간 delta(exec_rate_pct 등)가 정확함
  (--once 단발 실행은 이전 스냅샷이 없어 rate 가 0 으로 나옴).
- shm(=controller)이 없으면 조용히 대기 — controller 기동 후 자동 연결.
- gpu_util 은 exec_rate_pct(시간 점유율) 근사값으로 기록. DCGM 실측으로의
  정밀화는 백로그.
"""
from __future__ import annotations

import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

KETRIS_DIR = os.environ.get("KETRIS_DIR", "/opt/ketris")


class KetrisMetricsFeedback:
    def __init__(self, registry, history, group_id: str = "default",
                 interval_s: float = 10.0):
        self.registry = registry
        self.history = history
        self.group_id = group_id
        self.interval = interval_s
        self._logger_inst = None      # KETRIS MetricsLogger (lazy — shm 필요)
        self._stop = threading.Event()
        self._warned = False

    def _ensure_logger(self) -> bool:
        if self._logger_inst is not None:
            return True
        try:
            sys.path.insert(0, os.path.join(KETRIS_DIR, "monitor"))
            sys.path.insert(0, os.path.join(KETRIS_DIR, "shm"))
            from metrics_logger import MetricsLogger  # noqa: PLC0415
            self._logger_inst = MetricsLogger(self.group_id, stdout=False)
            logger.info(f"[feedback] KETRIS shm 연결됨 (group={self.group_id})")
            self._warned = False
            return True
        except Exception as e:
            if not self._warned:
                logger.warning(f"[feedback] KETRIS shm 미연결 — controller 기동 대기 ({e})")
                self._warned = True
            return False

    def _tick(self):
        if not self._ensure_logger():
            return
        try:
            rows = self._logger_inst.snapshot()
        except Exception as e:
            logger.warning(f"[feedback] snapshot 실패, 재연결 예정: {e}")
            self._logger_inst = None
            return

        # tenant_id(15자 절단) → model_id 매핑
        pod_to_model = {a["pod_name"][:15]: a["model_id"]
                        for a in self.registry.list_allocations()}
        for row in rows:
            model_id = pod_to_model.get(row["tenant"])
            if model_id is None:
                continue  # 자원풀 매니저가 모르는 테넌트 (외부 등록분)
            self.history.upsert(model_id, {
                # gpu_util ≈ 시간 점유율(exec_rate). DCGM 정밀화는 백로그.
                "gpu_util": row["exec_rate_pct"] / 100.0,
                "mode": row["mode"],
                # 확장 필드는 raw_json 으로
                "wait_rate_pct": row["wait_rate_pct"],
                "fairness": row["fairness"],
                "killer_rate": row["killer_rate"],
                "mem_mb": row["mem_mb"],
                "virtual_sm": row["virtual_sm"],
                "source": "ketris_shm",
            })

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.warning(f"[feedback] tick 오류(계속): {e}")
            self._stop.wait(self.interval)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()


__all__ = ["KetrisMetricsFeedback"]
