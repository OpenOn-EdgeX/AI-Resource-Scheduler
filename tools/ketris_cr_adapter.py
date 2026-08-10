#!/usr/bin/env python3
"""
KETRIS SliceAllocation CR 어댑터 — KETRIS Engine 의 CR Watch 구현 전 브리지.

역할 (KETRIS Engine 측 로직을 대행):
  ADDED   : spec 수신 → lsuAmount→virtual_sm 환산 → KETRIS controller /register
            (MPS SM% 실제 반영) → status.phase=PLACED + applied/occupancy 갱신
  DELETED : KETRIS /deregister → 자원 회수

주의: 본 어댑터는 전환기 브리지다. KETRIS Engine 이 자체 Watch·status 갱신을
구현하면 제거된다 (spec/status 계약은 동일하므로 스케줄러·자원풀 매니저 무변경).
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "resource-pool-manager"))
from pkg.ketris import KetrisClient  # noqa: E402

from kubernetes import client, config, watch  # noqa: E402

logging.basicConfig(level="INFO",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ketris-cr-adapter")

GROUP, VERSION, PLURAL = "keti.re.kr", "v1alpha1", "sliceallocations"
NAMESPACE = "edge-ai"

KETRIS_URL = os.environ.get("KETRIS_URL", "http://127.0.0.1:8090")
DEVICE_LSU_CAPACITY = float(os.environ.get("DEVICE_LSU_CAPACITY", "174"))
PHYSICAL_SM_TOTAL = int(os.environ.get("PHYSICAL_SM_TOTAL", "188"))


def lsu_to_virtual_sm(lsu: float) -> int:
    return max(1, round(lsu / DEVICE_LSU_CAPACITY * PHYSICAL_SM_TOTAL))


def main():
    config.load_kube_config()
    api = client.CustomObjectsApi()
    ketris = KetrisClient(KETRIS_URL)
    w = watch.Watch()
    logger.info(f"SliceAllocation Watch 시작 (ns={NAMESPACE}, ketris={KETRIS_URL}, "
                f"cap={DEVICE_LSU_CAPACITY} LSU / {PHYSICAL_SM_TOTAL} SM)")

    while True:
        try:
            for event in w.stream(api.list_namespaced_custom_object,
                                  GROUP, VERSION, NAMESPACE, PLURAL,
                                  timeout_seconds=60):
                obj, etype = event["object"], event["type"]
                name = obj["metadata"]["name"]
                spec = obj.get("spec", {})

                if etype == "ADDED" and not (obj.get("status") or {}).get("phase"):
                    lsu = float(spec.get("lsuAmount", 0))
                    vsm = lsu_to_virtual_sm(lsu)
                    mem_mb = int(spec.get("profile", {}).get("estPeakMemMb", 4096))
                    logger.info(f"watch event=ADDED sa={name} lsuAmount={lsu} vsm={vsm}")
                    res = ketris.register(name, virtual_sm=vsm,
                                          virtual_mem_mb=min(mem_mb, 8192))
                    if res is None:
                        phase, applied = "FAILED", {}
                        logger.warning(f"register sa={name} rc=FAIL")
                    else:
                        mps = (res.get("env") or {}).get(
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "?")
                        phase = "PLACED"
                        applied = {"spaceRatio": round(vsm / PHYSICAL_SM_TOTAL, 4),
                                   "timeRatio": 1.0, "memMb": min(mem_mb, 8192)}
                        logger.info(f"register sa={name} tenant_idx={res.get('tenant_idx')} "
                                    f"vsm={vsm} mem_mb={min(mem_mb, 8192)} mps_pct={mps}")
                    tenants = ketris.tenants() or []
                    used_sm = sum(t.get("virtual_sm", 0) for t in tenants)
                    status_body = {"status": {
                        "phase": phase,
                        "applied": applied,
                        "occupancy": {
                            "usedSmRatio": round(used_sm / PHYSICAL_SM_TOTAL, 4),
                            "usedTimeRatio": 1.0,
                            "usedMemMb": sum(t.get("virtual_mem_mb", 0) for t in tenants),
                            "activeWorkloads": len(tenants),
                        },
                        "timestamp": int(time.time()),
                    }}
                    api.patch_namespaced_custom_object_status(
                        GROUP, VERSION, NAMESPACE, PLURAL, name, status_body)
                    logger.info(f"status patch sa={name} phase={phase} "
                                f"spaceRatio={applied.get('spaceRatio')} "
                                f"activeWorkloads={len(tenants)}")

                elif etype == "DELETED":
                    res = ketris.deregister_by_pod(name)
                    logger.info(f"watch event=DELETED sa={name} deregister "
                                f"rc={'OK' if res is not None else 'SKIP'}")
        except Exception as e:
            logger.warning(f"Watch 재연결: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
