"""
논리 할당 레지스트리 — virtual_sm 추적 + mode/overcommit 계산.

계산식은 KETRIS에서 검증된 기존 공식을 그대로 이식:
  - mps_pct  : KETRIS controller registry::_calc_mps_pct
  - mode     : KETRIS controller overcommit::check_and_update
LSU→SM 환산(lsu_to_virtual_sm)은 본 모듈 신규 (KETRIS /register 가 LSU 를 받지
않으므로, 엣지 과제에서 lsu_amount → virtual_sm 어댑터 역할).

KETRIS controller/shm 라이브가 없는 환경에서는, 활성 할당의 단일 진실은
본 레지스트리다 (3단계에서 KETRIS 와 동기화 예정).
"""
from __future__ import annotations

import itertools
import threading
import time

MODE_FREE = "FREE"
MODE_OVERCOMMIT = "OVERCOMMIT"


class PoolRegistry:
    def __init__(self, physical_sm_total: int, device_lsu_capacity: float):
        self.physical_sm_total = int(physical_sm_total)
        self.device_lsu_capacity = float(device_lsu_capacity)
        self._allocs: dict[int, dict] = {}       # allocation_id -> record
        self._lock = threading.Lock()
        self._next_id = itertools.count(1)

    # ── 환산/계산식 ──────────────────────────────────────────────────────────
    def lsu_to_virtual_sm(self, lsu_amount: float) -> int:
        """LSU → 물리 SM 개수 단위 환산 (lsu/capacity 비율 × 총 SM)."""
        cap = max(self.device_lsu_capacity, 1e-6)
        return max(1, round(lsu_amount / cap * self.physical_sm_total))

    def calc_mps_pct(self, virtual_sm: int) -> int:
        """KETRIS controller registry::_calc_mps_pct 이식."""
        if self.physical_sm_total <= 0:
            return 100
        return min(100, int(virtual_sm / self.physical_sm_total * 100))

    # ── 할당 등록/해제 ───────────────────────────────────────────────────────
    def register(self, model_id: str, pod_name: str, lsu_amount: float) -> dict:
        virtual_sm = self.lsu_to_virtual_sm(lsu_amount)
        rec = {
            "allocation_id": next(self._next_id),
            "model_id": model_id,
            "pod_name": pod_name,
            "lsu_amount": float(lsu_amount),
            "virtual_sm": virtual_sm,
            "mps_pct": self.calc_mps_pct(virtual_sm),
            "registered_at": time.time(),
        }
        with self._lock:
            self._allocs[rec["allocation_id"]] = rec
        return rec

    def release(self, allocation_id: int) -> bool:
        with self._lock:
            return self._allocs.pop(allocation_id, None) is not None

    def release_by_pod(self, pod_name: str) -> int:
        """pod_name 으로 전체 해제 (PPO /feedback 시점 호출 경로). 해제 수 반환."""
        with self._lock:
            ids = [aid for aid, r in self._allocs.items() if r["pod_name"] == pod_name]
            for aid in ids:
                self._allocs.pop(aid)
        return len(ids)

    # ── 집계 ─────────────────────────────────────────────────────────────────
    def totals(self) -> dict:
        with self._lock:
            allocs = list(self._allocs.values())
        virtual_sm_total = sum(r["virtual_sm"] for r in allocs)
        allocated_lsu = sum(r["lsu_amount"] for r in allocs)
        # mode: KETRIS controller overcommit::check_and_update 이식
        mode = MODE_OVERCOMMIT if virtual_sm_total > self.physical_sm_total else MODE_FREE
        overcommit_ratio = (virtual_sm_total / self.physical_sm_total
                            if self.physical_sm_total > 0 else 0.0)
        # free_lsu 는 논리값(할당 약속 기준) — GPU util 실측과 다를 수 있음.
        # 오버커밋 시 0 으로 바닥나며, 실측 참고치는 /pool/status 의 *_observed 필드 참조.
        free_lsu = max(0.0, self.device_lsu_capacity - allocated_lsu)
        return {
            "active_tenants": len(allocs),
            "virtual_sm_total": virtual_sm_total,
            "allocated_lsu": round(allocated_lsu, 2),
            "free_lsu": round(free_lsu, 2),
            "overcommit_ratio": round(overcommit_ratio, 4),
            "mode": mode,
        }

    def list_allocations(self) -> list[dict]:
        with self._lock:
            return list(self._allocs.values())
