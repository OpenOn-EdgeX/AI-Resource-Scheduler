"""
SliceAllocation CR 발행기 — C-IS-08 (스케줄러 → KETRIS ERA, K8s CR).

스케줄러가 할당 결정 시 spec 을 작성해 CR 을 생성하고, 워크로드 종료 시 삭제한다.
status 는 KETRIS Engine(또는 전환기 어댑터)이 갱신하며 스케줄러는 쓰지 않는다.

best-effort: 클러스터가 없거나 CRD 미등록이어도 할당 응답은 정상 진행.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

GROUP, VERSION, PLURAL = "keti.re.kr", "v1alpha1", "sliceallocations"
NAMESPACE = "edge-ai"

# 이력 one-hot(resource_bound_type) → 연속 boundness 임시 환산.
# 연속값 산출 주체(프로파일링/커널 분석) 확정 시 실측값으로 대체.
_BOUNDNESS = {
    "COMPUTE_BOUND": (1.0, 0.0),
    "MEMORY_BOUND": (0.0, 1.0),
    "MIXED": (0.5, 0.5),
}
_WORKLOAD_TYPE = {"INFERENCE": "INFER", "TRAINING": "TRAIN"}


class SliceAllocationPublisher:
    def __init__(self):
        from kubernetes import client, config
        config.load_kube_config()
        self._api = client.CustomObjectsApi()
        self._k8s = client
        logger.info(f"[SliceCR] 발행기 초기화 (ns={NAMESPACE}, "
                    f"{GROUP}/{VERSION}/{PLURAL})")

    def create(self, name: str, request, lsu_amount: float) -> bool:
        """할당 결정 → SliceAllocation spec 발행. name = workload 식별자."""
        cb, mb = _BOUNDNESS.get(request.resource_bound_type, (0.5, 0.5))
        cap = max(request.device_lsu_capacity, 1e-6)
        slo_type = "THROUGHPUT" if request.slo_target_throughput else "LATENCY"
        slo_target = (request.slo_target_throughput
                      or request.p95_latency_ms or 0.0)
        body = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "SliceAllocation",
            "metadata": {"name": name, "namespace": NAMESPACE},
            "spec": {
                "gpuId": "gpu-0",
                "migPartitionId": "full",           # MIG 미사용 — 전체 파티션
                "lsuAmount": float(lsu_amount),
                "slo": {
                    "type": slo_type,
                    "target": float(slo_target),
                    "priority": int(request.priority),
                },
                "profile": {
                    "computeBoundness": cb,
                    "memoryBoundness": mb,
                    "estPeakMemMb": int(request.peak_memory_mb or 4096),
                    "workloadType": _WORKLOAD_TYPE.get(request.workload_kind, "INFER"),
                },
                "residual": {
                    "freeSmRatio": round(min(max(request.free_lsu / cap, 0.0), 1.0), 4),
                    "freeTimeRatio": 1.0,           # 시간 여유율 — KETRIS 산출값 연동 전 1.0
                    "freeMemMb": 0,                 # 노드 메모리 연동 전 0 (협의 필드)
                },
            },
        }
        try:
            self._api.create_namespaced_custom_object(
                GROUP, VERSION, NAMESPACE, PLURAL, body)
            logger.info(f"[SliceCR] 생성: {name} lsuAmount={lsu_amount} "
                        f"boundness=({cb},{mb}) slo={slo_type}/{slo_target}")
            return True
        except Exception as e:
            logger.warning(f"[SliceCR] 생성 실패(계속 진행): {e}")
            return False

    def delete(self, name: str) -> bool:
        """워크로드 종료 → CR 삭제 (KETRIS 측이 DELETED 이벤트로 자원 회수)."""
        try:
            self._api.delete_namespaced_custom_object(
                GROUP, VERSION, NAMESPACE, PLURAL, name)
            logger.info(f"[SliceCR] 삭제: {name}")
            return True
        except Exception as e:
            logger.warning(f"[SliceCR] 삭제 실패(계속 진행): {e}")
            return False

    def get(self, name: str) -> dict | None:
        try:
            return self._api.get_namespaced_custom_object(
                GROUP, VERSION, NAMESPACE, PLURAL, name)
        except Exception:
            return None


__all__ = ["SliceAllocationPublisher"]
