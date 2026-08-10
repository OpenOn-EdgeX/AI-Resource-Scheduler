"""
자원풀 매니저 클라이언트 — 폴백 안전장치 내장.

폴백 규약 (자원풀 매니저가 죽어도 PPO 는 절대 멈추지 않는다):
  1) 타임아웃 1초, 재시도 없음
  2) 실패 시 30초 내 성공 캐시(stale)가 있으면 그걸 사용
  3) 그것도 없으면 None 반환 → 호출측(api)이 기존 mock 로직으로 완전 폴백
  4) 서킷브레이커: 실패 후 10초간 호출 스킵(요청마다 1초 지연 방지), 이후 자동 재시도
  5) 다운그레이드/복구 로그는 상태 전환 시에만 (스팸 방지)

의존성: stdlib(urllib)만 사용 — requests 불필요.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class PoolClient:
    def __init__(self, base_url: str, timeout_s: float = 1.0,
                 cache_ttl_s: float = 30.0, breaker_cooldown_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.cache_ttl = cache_ttl_s
        self.cooldown = breaker_cooldown_s
        self._lock = threading.Lock()
        self._status_cache: tuple[float, dict] | None = None   # (ts, data)
        self._breaker_until = 0.0
        self._healthy = True   # 로그 상태 전환 추적용

    # ── 내부 HTTP ────────────────────────────────────────────────────────────
    def _call(self, method: str, path: str, body: dict | None = None) -> dict | None:
        now = time.monotonic()
        if now < self._breaker_until:
            return None  # 서킷 open — 쿨다운 중 호출 스킵
        req = urllib.request.Request(
            self.base_url + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
            if not self._healthy:
                logger.warning(f"[pool] 자원풀 매니저 복구됨 ({self.base_url})")
                self._healthy = True
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # 이력 없음 등 — 정상 케이스, 브레이커 열지 않음
            self._trip(f"HTTP {e.code}")
            return None
        except Exception as e:
            self._trip(str(e))
            return None

    def _trip(self, reason: str):
        self._breaker_until = time.monotonic() + self.cooldown
        if self._healthy:
            logger.warning(f"[pool] 자원풀 매니저 응답 없음 → mock 폴백으로 전환 "
                           f"(사유: {reason}, {self.cooldown}s 후 재시도)")
            self._healthy = False

    # ── 공개 API ─────────────────────────────────────────────────────────────
    def get_status(self) -> dict | None:
        """④그룹 조회. 실패 시 stale 캐시 → 그것도 없으면 None(mock 폴백)."""
        data = self._call("GET", "/pool/status")
        now = time.monotonic()
        if data is not None:
            with self._lock:
                self._status_cache = (now, data)
            return data
        with self._lock:
            if self._status_cache and now - self._status_cache[0] <= self.cache_ttl:
                return self._status_cache[1]   # stale 캐시 (30초 내)
        return None

    def get_history(self, model_id: str) -> dict | None:
        """⑤그룹 이력 조회. 404/실패 → None(lsu_est 폴백)."""
        if not model_id:
            return None
        return self._call("GET", f"/history/{model_id}")

    def register_allocation(self, model_id: str, pod_name: str,
                            lsu_amount: float) -> dict | None:
        """PPO /allocate 결정 직후 자동 등록 (best-effort — 실패해도 할당 응답은 나감)."""
        return self._call("POST", "/allocations",
                          {"model_id": model_id, "pod_name": pod_name,
                           "lsu_amount": lsu_amount})

    def release_by_pod(self, pod_name: str) -> dict | None:
        """PPO /feedback(종료) 시점 자동 해제 (best-effort)."""
        return self._call("POST", "/allocations/release_by_pod", {"pod_name": pod_name})

    def upsert_history(self, model_id: str, data: dict) -> dict | None:
        """실행 결과를 모델 이력에 기록 (best-effort). 3단계 KETRIS 실측 피드백도 이 경로."""
        if not model_id:
            return None
        return self._call("POST", f"/history/{model_id}", data)


__all__ = ["PoolClient"]
