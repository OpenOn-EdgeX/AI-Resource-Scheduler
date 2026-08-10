"""
Resource Pool Manager gRPC 서버 — C-RP-01/02/03 + 이력 upsert (레이어 내부 gRPC).

기존 REST(:8082)와 병행 기동한다(전환기 안전). 로직은 registry/history 를 그대로
재사용하며, KETRIS 연동은 여기서 하지 않는다 — SliceAllocation CR(spec/status)
경로가 담당 (스케줄러 spec 발행 → KETRIS status 갱신 → 본 모듈 status Watch).
"""
from __future__ import annotations

import logging
from concurrent import futures

import grpc

from .rpc import resource_pool_pb2 as pb
from .rpc import resource_pool_pb2_grpc as pb_grpc

logger = logging.getLogger("pool-manager.grpc")

_MODE = {"FREE": pb.FREE, "OVERCOMMIT": pb.OVERCOMMIT}
_BOUND = {"COMPUTE_BOUND": pb.COMPUTE_BOUND, "MEMORY_BOUND": pb.MEMORY_BOUND,
          "MIXED": pb.MIXED}
_BOUND_REV = {v: k for k, v in _BOUND.items()}


class ResourcePoolServicer(pb_grpc.ResourcePoolServicer):
    def __init__(self, registry, history, collector):
        self.registry = registry
        self.history = history
        self.collector = collector

    # ── C-RP-01 가용 자원·점유 상태 조회 ─────────────────────────────────────
    def GetAvailableResourceStatus(self, request, context):
        totals = self.registry.totals()
        node = self.collector.latest()
        logger.info(f"[C-RP-01] GetAvailableResourceStatus(device={request.device_id or '-'}, "
                    f"partition={request.partition_id or '-'}) -> free_lsu={totals['free_lsu']}, "
                    f"tenants={totals['active_tenants']}, mode={totals['mode']}")
        return pb.AvailableResourceStatus(
            free_lsu=totals["free_lsu"],
            overcommit_ratio=totals["overcommit_ratio"],
            active_tenants=totals["active_tenants"],
            mode=_MODE.get(totals["mode"], pb.MODE_UNSPECIFIED),
            gpu_util=node["gpu_util"],
            device_lsu_capacity=self.registry.device_lsu_capacity,
            physical_sm_total=self.registry.physical_sm_total,
        )

    # ── C-RP-02 모델 실행 이력 조회 ──────────────────────────────────────────
    def GetModelExecutionHistory(self, request, context):
        rec = self.history.get(request.model_id)
        if rec is None:
            logger.info(f"[C-RP-02] GetModelExecutionHistory({request.model_id}) -> 이력 없음")
            return pb.ModelExecutionHistory(model_id=request.model_id, found=False)
        logger.info(f"[C-RP-02] GetModelExecutionHistory({request.model_id}) -> "
                    f"bound={rec.get('resource_bound_type')}, run_count={rec.get('run_count')}")
        return pb.ModelExecutionHistory(
            model_id=rec["model_id"],
            kernels_per_iter=int(rec.get("kernels_per_iter") or 0),
            resource_bound_type=_BOUND.get(rec.get("resource_bound_type"),
                                           pb.RESOURCE_BOUND_TYPE_UNSPECIFIED),
            avg_gpu_util=float(rec.get("avg_gpu_util") or 0.0),
            run_count=int(rec.get("run_count") or 0),
            mps_pct=float(rec.get("mps_pct") or 0.0),
            mode=_MODE.get(rec.get("mode"), pb.MODE_UNSPECIFIED),
            avg_throughput=float(rec.get("avg_throughput") or 0.0),
            found=True,
        )

    # ── C-RP-03 스케줄링 결과 등록·갱신 (RESERVED / RELEASED) ────────────────
    def UpdateAllocationDecision(self, request, context):
        if request.allocation_state == pb.RESERVED:
            if request.lsu_amount <= 0:
                return pb.UpdateAllocationDecisionResult(success=False, result_code=2)
            rec = self.registry.register(
                model_id=request.model_id or request.workload_id,
                pod_name=request.pod_name or request.workload_id,
                lsu_amount=request.lsu_amount)
            totals = self.registry.totals()
            logger.info(f"[C-RP-03] RESERVED: {request.workload_id} lsu={request.lsu_amount} "
                        f"vsm={rec['virtual_sm']} → total_vsm={totals['virtual_sm_total']} "
                        f"mode={totals['mode']} (KETRIS 반영은 SliceAllocation CR 경로)")
            return pb.UpdateAllocationDecisionResult(
                success=True, result_code=0,
                virtual_sm=rec["virtual_sm"], mps_pct=rec["mps_pct"],
                mode=_MODE.get(totals["mode"], pb.MODE_UNSPECIFIED),
                free_lsu=totals["free_lsu"], active_tenants=totals["active_tenants"],
                overcommit_ratio=totals["overcommit_ratio"])

        if request.allocation_state == pb.RELEASED:
            n = self.registry.release_by_pod(request.pod_name or request.workload_id)
            totals = self.registry.totals()
            logger.info(f"[C-RP-03] RELEASED: {request.workload_id} released={n} "
                        f"→ total_vsm={totals['virtual_sm_total']} mode={totals['mode']}")
            return pb.UpdateAllocationDecisionResult(
                success=n > 0, result_code=0 if n > 0 else 1,
                mode=_MODE.get(totals["mode"], pb.MODE_UNSPECIFIED),
                free_lsu=totals["free_lsu"], active_tenants=totals["active_tenants"],
                overcommit_ratio=totals["overcommit_ratio"])

        return pb.UpdateAllocationDecisionResult(success=False, result_code=2)

    # ── 실행 결과 저장 (F-RP-06) ─────────────────────────────────────────────
    def UpsertModelExecutionHistory(self, request, context):
        data = {
            "gpu_util": request.gpu_util,
            "throughput": request.throughput,
            "mps_pct": request.mps_pct,
        }
        if request.resource_bound_type != pb.RESOURCE_BOUND_TYPE_UNSPECIFIED:
            data["resource_bound_type"] = _BOUND_REV[request.resource_bound_type]
        if request.kernels_per_iter:
            data["kernels_per_iter"] = request.kernels_per_iter
        if request.mode != pb.MODE_UNSPECIFIED:
            data["mode"] = "FREE" if request.mode == pb.FREE else "OVERCOMMIT"
        rec = self.history.upsert(request.model_id, data)
        logger.info(f"[F-RP-06] UpsertHistory({request.model_id}) -> run_count={rec['run_count']}")
        return pb.ModelExecutionHistory(
            model_id=rec["model_id"],
            kernels_per_iter=int(rec.get("kernels_per_iter") or 0),
            resource_bound_type=_BOUND.get(rec.get("resource_bound_type"),
                                           pb.RESOURCE_BOUND_TYPE_UNSPECIFIED),
            avg_gpu_util=float(rec.get("avg_gpu_util") or 0.0),
            run_count=int(rec.get("run_count") or 0),
            avg_throughput=float(rec.get("avg_throughput") or 0.0),
            found=True,
        )


def serve(registry, history, collector, port: int = 50052) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_ResourcePoolServicer_to_server(
        ResourcePoolServicer(registry, history, collector), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    logger.info(f"gRPC 서버 기동: :{port} (C-RP-01/02/03 + 이력 upsert, REST 병행)")
    return server


__all__ = ["serve", "ResourcePoolServicer"]
