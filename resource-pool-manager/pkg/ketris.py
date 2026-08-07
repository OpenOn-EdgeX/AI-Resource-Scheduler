"""
KETRIS(ERA controller) 어댑터 — 3단계 연동.

경로: PPO lsu_amount → 자원풀 매니저(LSU→virtual_sm 환산) → KETRIS POST /register
     → KETRIS 가 mps_pct(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) env 반환 → 시공간 분할 적용.

원칙:
- best-effort: KETRIS(:8090)가 죽어 있어도 자원풀 매니저는 로컬 registry 만으로 동작.
- 기동 시 GET /tenants 동기화: 자원풀 매니저 재기동 시 KETRIS 를 진실원으로
  논리 할당 목록을 복원 ("메모리 초기화" 문제의 해소 경로).
- tenant_id 는 KETRIS registry.py 가 15자로 절단하므로 여기서 미리 맞춘다.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class KetrisClient:
    def __init__(self, base_url: str, timeout_s: float = 2.0,
                 breaker_cooldown_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.cooldown = breaker_cooldown_s
        self._breaker_until = 0.0
        self._healthy = True

    @staticmethod
    def tenant_id_for(pod_name: str) -> str:
        """KETRIS registry.py 는 tenant_id 를 15자로 절단 — 사전 정규화."""
        return pod_name[:15]

    def _call(self, method: str, path: str, body: dict | None = None) -> dict | None:
        if time.monotonic() < self._breaker_until:
            return None
        req = urllib.request.Request(
            self.base_url + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
            if not self._healthy:
                logger.warning(f"[ketris] KETRIS 복구됨 ({self.base_url})")
                self._healthy = True
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            self._trip(f"HTTP {e.code}")
            return None
        except Exception as e:
            self._trip(str(e))
            return None

    def _trip(self, reason: str):
        self._breaker_until = time.monotonic() + self.cooldown
        if self._healthy:
            logger.warning(f"[ketris] KETRIS 응답 없음 → 로컬 registry 단독 동작 "
                           f"(사유: {reason}, {self.cooldown}s 후 재시도)")
            self._healthy = False

    # ── 공개 API ─────────────────────────────────────────────────────────────
    def register(self, pod_name: str, virtual_sm: int,
                 virtual_mem_mb: int = 4096, weight: float = 1.0) -> dict | None:
        """
        KETRIS 테넌트 등록. 성공 시 {tenant_idx, env} 반환 —
        env 에 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE(공간분할 적용값) 포함.
        """
        return self._call("POST", "/register", {
            "tenant_id": self.tenant_id_for(pod_name),
            "virtual_sm": int(virtual_sm),
            "virtual_mem_mb": int(virtual_mem_mb),
            "weight": weight,
        })

    def deregister_by_pod(self, pod_name: str) -> dict | None:
        return self._call("POST", "/deregister_by_id",
                          {"tenant_id": self.tenant_id_for(pod_name)})

    def tenants(self) -> list[dict] | None:
        """활성 테넌트 목록 (기동 시 동기화용). 실패 시 None."""
        d = self._call("GET", "/tenants")
        return d.get("tenants") if d else None

    def status(self) -> dict | None:
        return self._call("GET", "/status")


__all__ = ["KetrisClient"]
