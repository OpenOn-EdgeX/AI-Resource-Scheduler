"""
SliceAllocation CR status Watcher — KETRIS 실제 할당 결과 동기화 (구 C-RP-04 대체).

새 연동 구조:
  스케줄러가 spec 작성 → KETRIS(어댑터)가 시공간 분할 적용 후 status 갱신
  → 본 모듈이 status.phase 를 Watch 하여 자원 풀 상태를 확정/회수한다.
직접 통신(gRPC Push / shm 폴링) 없이 K8s API Server 를 매개로 동기화한다.

phase 매핑 (설계서 상태 모델 ↔ CR):
  RESERVED(예약) → PLACED(실제 적용 = ALLOCATED 상당) → RELEASED(회수)
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("pool-manager.cr-watch")

GROUP, VERSION, PLURAL = "keti.re.kr", "v1alpha1", "sliceallocations"
NAMESPACE = "edge-ai"


class SliceAllocationWatcher:
    def __init__(self, registry, history):
        self.registry = registry
        self.history = history
        self._stop = threading.Event()

    def _handle(self, event):
        obj = event["object"]
        etype = event["type"]                      # ADDED / MODIFIED / DELETED
        name = obj["metadata"]["name"]
        spec = obj.get("spec", {})
        status = obj.get("status") or {}
        phase = status.get("phase")

        if etype == "DELETED":
            logger.info(f"[CR-WATCH] {name} DELETED — 워크로드 종료·CR 회수 확인")
            return
        if not phase:
            return

        if phase == "PLACED":
            applied = status.get("applied", {})
            logger.info(
                f"[CR-WATCH] {name} status.phase=PLACED — KETRIS 실제 적용 확인: "
                f"spaceRatio={applied.get('spaceRatio')}, timeRatio={applied.get('timeRatio')}, "
                f"memMb={applied.get('memMb')} → 예약(RESERVED)을 확정(ALLOCATED 상당)으로 동기화")
        elif phase == "RELEASED":
            n = self.registry.release_by_pod(name)
            logger.info(f"[CR-WATCH] {name} status.phase=RELEASED — 자원 회수 동기화 "
                        f"(released={n}) → {self.registry.totals()}")
        elif phase == "FAILED":
            n = self.registry.release_by_pod(name)
            logger.info(f"[CR-WATCH] {name} status.phase=FAILED — 예약 회수(released={n})")
        else:
            logger.info(f"[CR-WATCH] {name} status.phase={phase} (lsu={spec.get('lsuAmount')})")

    def _loop(self):
        from kubernetes import client, config, watch
        config.load_kube_config()
        api = client.CustomObjectsApi()
        w = watch.Watch()
        logger.info(f"[CR-WATCH] SliceAllocation status Watch 시작 (ns={NAMESPACE})")
        while not self._stop.is_set():
            try:
                for event in w.stream(api.list_namespaced_custom_object,
                                      GROUP, VERSION, NAMESPACE, PLURAL,
                                      timeout_seconds=30):
                    self._handle(event)
                    if self._stop.is_set():
                        break
            except Exception as e:
                logger.warning(f"[CR-WATCH] 재연결 예정: {e}")
                self._stop.wait(3)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()


__all__ = ["SliceAllocationWatcher"]
