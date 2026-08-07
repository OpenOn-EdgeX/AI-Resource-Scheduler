"""
자원풀 매니저 gRPC 클라이언트 (C-RP-01~03) — PoolClient 와 동일 인터페이스.

REST 판의 폴백 규약을 그대로 계승한다:
  1) 타임아웃 1초, 재시도 없음
  2) 실패 시 30초 내 성공 캐시(stale) 사용
  3) 그것도 없으면 None → 호출측 mock 폴백
  4) 서킷브레이커 10초
반환 dict 의 키는 REST 응답과 동일하게 맞춰 호출측(api) 수정을 없앤다.
"""
from __future__ import annotations

import logging
import threading
import time

import grpc

from .rpc import resource_pool_pb2 as pb
from .rpc import resource_pool_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

_MODE_STR = {pb.FREE: "FREE", pb.OVERCOMMIT: "OVERCOMMIT"}
_BOUND_STR = {pb.COMPUTE_BOUND: "COMPUTE_BOUND", pb.MEMORY_BOUND: "MEMORY_BOUND",
              pb.MIXED: "MIXED"}
_BOUND_PB = {v: k for k, v in _BOUND_STR.items()}


class PoolClientGrpc:
    def __init__(self, target: str, timeout_s: float = 1.0,
                 cache_ttl_s: float = 30.0, breaker_cooldown_s: float = 10.0):
        self.target = target
        self.timeout = timeout_s
        self.cache_ttl = cache_ttl_s
        self.cooldown = breaker_cooldown_s
        self._channel = grpc.insecure_channel(target)
        self._stub = pb_grpc.ResourcePoolStub(self._channel)
        self._lock = threading.Lock()
        self._status_cache: tuple[float, dict] | None = None
        self._breaker_until = 0.0
        self._healthy = True

    # ── 내부 호출 공통 ───────────────────────────────────────────────────────
    def _guard(self):
        return time.monotonic() >= self._breaker_until

    def _ok(self):
        if not self._healthy:
            logger.warning(f"[pool-grpc] 자원풀 매니저 복구됨 ({self.target})")
            self._healthy = True

    def _trip(self, reason: str):
        self._breaker_until = time.monotonic() + self.cooldown
        if self._healthy:
            logger.warning(f"[pool-grpc] 자원풀 매니저 gRPC 응답 없음 → mock 폴백 "
                           f"(사유: {reason}, {self.cooldown}s 후 재시도)")
            self._healthy = False

    # ── 공개 API (PoolClient 와 동일 시그니처) ───────────────────────────────
    def get_status(self) -> dict | None:
        """C-RP-01 — ④그룹 조회. 실패 시 stale 캐시 → None(mock 폴백)."""
        now = time.monotonic()
        if self._guard():
            try:
                r = self._stub.GetAvailableResourceStatus(
                    pb.GetAvailableResourceStatusRequest(device_id="gpu-0",
                                                         partition_id="full"),
                    timeout=self.timeout)
                data = {
                    "free_lsu": r.free_lsu,
                    "overcommit_ratio": r.overcommit_ratio,
                    "active_tenants": r.active_tenants,
                    "mode": _MODE_STR.get(r.mode, "FREE"),
                    "gpu_util_observed": r.gpu_util,
                    "device_lsu_capacity": r.device_lsu_capacity,
                    "physical_sm_total": r.physical_sm_total,
                    "transport": "grpc",
                }
                self._ok()
                with self._lock:
                    self._status_cache = (now, data)
                return data
            except grpc.RpcError as e:
                self._trip(e.code().name)
        with self._lock:
            if self._status_cache and now - self._status_cache[0] <= self.cache_ttl:
                return self._status_cache[1]
        return None

    def get_history(self, model_id: str) -> dict | None:
        """C-RP-02 — ⑤그룹 이력 조회. found=False/실패 → None(lsu_est 폴백)."""
        if not model_id or not self._guard():
            return None
        try:
            r = self._stub.GetModelExecutionHistory(
                pb.GetModelExecutionHistoryRequest(model_id=model_id),
                timeout=self.timeout)
            self._ok()
            if not r.found:
                return None
            return {
                "model_id": r.model_id,
                "kernels_per_iter": r.kernels_per_iter,
                "resource_bound_type": _BOUND_STR.get(r.resource_bound_type),
                "avg_gpu_util": r.avg_gpu_util,
                "run_count": r.run_count,
                "mps_pct": r.mps_pct,
                "mode": _MODE_STR.get(r.mode),
                "avg_throughput": r.avg_throughput,
            }
        except grpc.RpcError as e:
            self._trip(e.code().name)
            return None

    def register_allocation(self, model_id: str, pod_name: str,
                            lsu_amount: float) -> dict | None:
        """C-RP-03 RESERVED — 할당 결정 등록 (best-effort)."""
        if not self._guard():
            return None
        try:
            r = self._stub.UpdateAllocationDecision(
                pb.UpdateAllocationDecisionRequest(
                    workload_id=pod_name, model_id=model_id, pod_name=pod_name,
                    device_id="gpu-0", partition_id="full",
                    lsu_amount=lsu_amount, allocation_state=pb.RESERVED),
                timeout=self.timeout)
            self._ok()
            return {
                "virtual_sm": r.virtual_sm, "mps_pct": r.mps_pct,
                "pool": {"mode": _MODE_STR.get(r.mode, "FREE"),
                         "free_lsu": r.free_lsu,
                         "active_tenants": r.active_tenants,
                         "overcommit_ratio": r.overcommit_ratio},
                "success": r.success, "result_code": r.result_code,
            }
        except grpc.RpcError as e:
            self._trip(e.code().name)
            return None

    def release_by_pod(self, pod_name: str) -> dict | None:
        """C-RP-03 RELEASED — 종료 시 해제 (best-effort)."""
        if not self._guard():
            return None
        try:
            r = self._stub.UpdateAllocationDecision(
                pb.UpdateAllocationDecisionRequest(
                    workload_id=pod_name, pod_name=pod_name,
                    allocation_state=pb.RELEASED),
                timeout=self.timeout)
            self._ok()
            return {"released": 1 if r.success else 0,
                    "pool": {"mode": _MODE_STR.get(r.mode, "FREE"),
                             "free_lsu": r.free_lsu,
                             "active_tenants": r.active_tenants}}
        except grpc.RpcError as e:
            self._trip(e.code().name)
            return None

    def upsert_history(self, model_id: str, data: dict) -> dict | None:
        """실행 결과 이력 저장 (best-effort)."""
        if not model_id or not self._guard():
            return None
        try:
            r = self._stub.UpsertModelExecutionHistory(
                pb.UpsertModelExecutionHistoryRequest(
                    model_id=model_id,
                    gpu_util=float(data.get("gpu_util") or 0.0),
                    throughput=float(data.get("throughput") or 0.0),
                    resource_bound_type=_BOUND_PB.get(
                        data.get("resource_bound_type"),
                        pb.RESOURCE_BOUND_TYPE_UNSPECIFIED),
                    kernels_per_iter=int(data.get("kernels_per_iter") or 0),
                    mps_pct=float(data.get("mps_pct") or 0.0)),
                timeout=self.timeout)
            self._ok()
            return {"model_id": r.model_id, "run_count": r.run_count}
        except grpc.RpcError as e:
            self._trip(e.code().name)
            return None


__all__ = ["PoolClientGrpc"]
