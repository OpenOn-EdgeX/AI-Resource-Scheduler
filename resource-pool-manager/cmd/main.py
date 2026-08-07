#!/usr/bin/env python3
"""
KETI Resource Pool Manager — 엣지 AI 과제 (2단계).

역할 1: 노드 자원 실시간 수집 → ④그룹(free_lsu/overcommit_ratio/active_tenants) REST 노출
역할 2: 모델별 실행 이력(⑤그룹) SQLite 저장/조회 (KETRIS 실측 피드백은 3단계 수신)

실행: python3 cmd/main.py  (기본 0.0.0.0:8082)
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

from pkg.registry import PoolRegistry
from pkg.history import ModelHistoryStore
from pkg.collector import NodeCollector, detect_physical_sm
from pkg.ketris import KetrisClient
from pkg.feedback import KetrisMetricsFeedback

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pool-manager")

API_PORT = int(os.environ.get("POOL_API_PORT", "8082"))
COLLECT_INTERVAL_S = float(os.environ.get("COLLECT_INTERVAL_S", "5"))
DEVICE_LSU_CAPACITY = float(os.environ.get("DEVICE_LSU_CAPACITY", "174"))  # 실측
DATA_DIR = os.environ.get("POOL_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
NODE_NAME = os.environ.get("NODE_NAME", "edge-node")

KETRIS_URL = os.environ.get("KETRIS_URL", "http://127.0.0.1:8090")
FEEDBACK_INTERVAL_S = float(os.environ.get("FEEDBACK_INTERVAL_S", "10"))

app = Flask(__name__)
collector = NodeCollector(COLLECT_INTERVAL_S)
registry = PoolRegistry(detect_physical_sm(), DEVICE_LSU_CAPACITY)
history = ModelHistoryStore(os.path.join(DATA_DIR, "model_history.sqlite"))
ketris = KetrisClient(KETRIS_URL)                       # best-effort — 없으면 로컬 단독
feedback = KetrisMetricsFeedback(registry, history, interval_s=FEEDBACK_INTERVAL_S)


def sync_from_ketris() -> int:
    """
    기동 시 KETRIS GET /tenants 를 진실원으로 논리 할당 목록 복원.
    (자원풀 매니저 재기동 시 메모리 초기화 문제의 해소 경로 — 3단계 흡수)
    virtual_sm → lsu 역환산: lsu = vsm / physical_sm × capacity.
    """
    tenants = ketris.tenants()
    if not tenants:
        return 0
    n = 0
    for t in tenants:
        lsu = t["virtual_sm"] / max(registry.physical_sm_total, 1) * registry.device_lsu_capacity
        # model_id 는 복원 불가(레지스트리 메모리 소실) → tenant_id 로 대체, 이후 이력은 model_id 재매핑
        registry.register(model_id=t["tenant_id"], pod_name=t["tenant_id"], lsu_amount=round(lsu, 2))
        n += 1
    logger.info(f"[sync] KETRIS /tenants 동기화: {n}건 복원 → {registry.totals()}")
    return n


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "node": NODE_NAME})


@app.get("/pool/status")
def pool_status():
    """④그룹 일괄 조회 — PPO build_state 의 가용 자원 상태 소스."""
    node = collector.latest()
    totals = registry.totals()
    cap = registry.device_lsu_capacity
    return jsonify({
        # 논리값 (할당 약속 기준) — GPU util 실측과 다를 수 있음
        **totals,
        "physical_sm_total": registry.physical_sm_total,
        "device_lsu_capacity": cap,
        # 실측 참고 필드 (gpu_util 기반 관찰값 — 논리값과 병기)
        "gpu_util_observed": node["gpu_util"],
        "free_lsu_observed": round(cap * max(0.0, 1.0 - node["gpu_util"]), 2),
        "mem_used_mb": node["mem_used_mb"], "mem_total_mb": node["mem_total_mb"],
        "nvidia_smi_ok": node["nvidia_smi_ok"],
        "collected_at": node["collected_at"], "ts": time.time(),
        "node": NODE_NAME,
    })


@app.post("/allocations")
def register_allocation():
    """PPO /allocate 가 결정 직후 자동 호출 (연동 흐름의 등록 지점)."""
    body = request.get_json() or {}
    try:
        model_id = body["model_id"]
        lsu_amount = float(body["lsu_amount"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"model_id/lsu_amount 필수: {e}"}), 400
    pod_name = body.get("pod_name", model_id)
    rec = registry.register(model_id, pod_name, lsu_amount)

    # KETRIS 로 전달 (lsu_amount → virtual_sm 환산값. best-effort — 다운 시 로컬만)
    ketris_res = ketris.register(
        pod_name, virtual_sm=rec["virtual_sm"],
        virtual_mem_mb=int(body.get("virtual_mem_mb", 4096)),
        weight=float(body.get("weight", 1.0)))
    ketris_env = (ketris_res or {}).get("env")   # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 포함

    totals = registry.totals()
    logger.info(f"[alloc+] {model_id} lsu={lsu_amount} vsm={rec['virtual_sm']} "
                f"→ total_vsm={totals['virtual_sm_total']} mode={totals['mode']} "
                f"ketris={'OK' if ketris_res else 'SKIP(다운)'}")
    return jsonify({**rec, "pool": totals,
                    "ketris_registered": ketris_res is not None,
                    "ketris_env": ketris_env})


@app.delete("/allocations/<int:allocation_id>")
def release_allocation(allocation_id: int):
    ok = registry.release(allocation_id)
    return (jsonify({"ok": True}), 200) if ok else (jsonify({"error": "not found"}), 404)


@app.post("/allocations/release_by_pod")
def release_by_pod():
    """PPO /feedback(워크로드 종료) 시점 자동 호출 (연동 흐름의 해제 지점)."""
    body = request.get_json() or {}
    pod_name = body.get("pod_name")
    if not pod_name:
        return jsonify({"error": "pod_name 필수"}), 400
    n = registry.release_by_pod(pod_name)
    ketris_res = ketris.deregister_by_pod(pod_name)   # best-effort
    totals = registry.totals()
    logger.info(f"[alloc-] pod={pod_name} released={n} "
                f"→ total_vsm={totals['virtual_sm_total']} mode={totals['mode']} "
                f"ketris={'OK' if ketris_res else 'SKIP'}")
    return jsonify({"released": n, "pool": totals,
                    "ketris_deregistered": ketris_res is not None})


@app.get("/allocations")
def list_allocations():
    return jsonify({"allocations": registry.list_allocations(),
                    "pool": registry.totals()})


@app.get("/history/<model_id>")
def get_history(model_id: str):
    """⑤그룹 이력 조회. 404 = 이력 없음 → PPO 는 lsu_est 폴백."""
    rec = history.get(model_id)
    if rec is None:
        return jsonify({"error": "no history", "model_id": model_id}), 404
    return jsonify(rec)


@app.post("/history/<model_id>")
def put_history(model_id: str):
    """KETRIS 실측 결과 upsert (3단계 피드백 수신 지점)."""
    rec = history.upsert(model_id, request.get_json() or {})
    return jsonify(rec)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    collector.start()
    sync_from_ketris()      # 기동 시 KETRIS 진실원 동기화 (없으면 0건)
    feedback.start()        # KETRIS shm 실측 → /history 피드백 루프

    # 레이어 내부 gRPC (C-RP-01~03) — REST 와 병행 기동
    # (반환 서버 객체를 전역으로 보관 — 참조 소실 시 GC 로 서버가 중지됨)
    global _grpc_server
    grpc_port = int(os.environ.get("POOL_GRPC_PORT", "50052"))
    try:
        from pkg.grpc_server import serve as grpc_serve
        _grpc_server = grpc_serve(registry, history, collector, port=grpc_port)
    except Exception as e:
        logger.warning(f"gRPC 서버 기동 실패(REST 단독 운영): {e}")

    # SliceAllocation CR status Watch — KETRIS 실제 할당 결과 동기화 (구 C-RP-04 대체)
    if os.environ.get("SLICE_CR_WATCH", "1") == "1":
        try:
            from pkg.cr_watcher import SliceAllocationWatcher
            SliceAllocationWatcher(registry, history).start()
        except Exception as e:
            logger.warning(f"CR Watcher 기동 실패(클러스터 없음?): {e}")

    logger.info(f"Pool Manager 시작: :{API_PORT}(REST) :{grpc_port}(gRPC), "
                f"physical_sm={registry.physical_sm_total}, "
                f"capacity={DEVICE_LSU_CAPACITY} LSU, interval={COLLECT_INTERVAL_S}s, "
                f"ketris={KETRIS_URL}, history={history.count()}건")
    app.run(host="0.0.0.0", port=API_PORT, threaded=True)


if __name__ == "__main__":
    main()
