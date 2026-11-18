#!/usr/bin/env python3
"""structure set 예시 JSON 생성 — 실데이터로 채울 수 있는 것은 클러스터에서 가져오고, 외부 입력은 형식 예시로 둔다.

  kubectl -n keti-exec-system port-forward svc/keti-queue-api 18090:80 &
  kubectl -n keti-exec-system port-forward svc/keti-scheduler 18092:80 &
  python3 make_examples.py [--workload <kfp 학습 단계 workload_id>]
"""
import argparse
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone

QA, SCHED = "http://localhost:18090", "http://localhost:18092"
NODE_NAME = os.environ.get("KETI_EXAMPLE_NODE", "example-node")


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def kubectl(*a):
    return json.loads(subprocess.check_output(["kubectl", *a, "-o", "json"]))


def save(name, obj, note):
    json.dump({"_note": note, **obj}, open(name, "w"), ensure_ascii=False, indent=2)
    print("wrote", name)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def partition_set():
    """선택한 노드 실측: GPU LSU 할당 가능량·사용량, NPU PE, 지금 올라간 워크로드별 몫."""
    node = kubectl("get", "node", NODE_NAME)
    alloc = node["status"]["allocatable"]
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total",
                                   "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    lsu_parts, pe_parts, lsu_used, pe_used = [], [], 0.0, 0
    for p in kubectl("get", "pods", "-A")["items"]:
        if p["spec"].get("nodeName") != NODE_NAME or p["status"].get("phase") in ("Succeeded", "Failed"):
            continue
        req = {}
        for c in p["spec"]["containers"]:
            for k, v in ((c.get("resources") or {}).get("requests") or {}).items():
                req[k] = req.get(k, 0) + float(v) if k.startswith("keti.re.kr/") else req.get(k)
        wid = (p["metadata"].get("labels") or {}).get("keti.re.kr/workload-id", f"{p['metadata']['namespace']}/{p['metadata']['name']}")
        if req.get("keti.re.kr/lsu"):
            lsu_used += req["keti.re.kr/lsu"]
            lsu_parts.append({"partition_id": f"lsu-{p['metadata']['name']}", "method": "SPACE_TIME", "state": "BOUND",
                              "lsu_amount": req["keti.re.kr/lsu"], "space_ratio": None, "time_ratio": None,
                              "workload_ids": [wid], "config_version": None})
        if req.get("keti.re.kr/npu-pe"):
            pe_used += int(req["keti.re.kr/npu-pe"])
            pe_parts.append({"partition_id": f"pe-{p['metadata']['name']}", "method": "NPU_PE", "state": "BOUND",
                             "pe_count": int(req["keti.re.kr/npu-pe"]), "workload_ids": [wid]})
    cap_lsu, cap_pe = float(alloc.get("keti.re.kr/lsu", 0)), int(alloc.get("keti.re.kr/npu-pe", 0))
    return {"schema_version": "0.1", "collected_at": now(), "nodes": [{
        "node_name": NODE_NAME, "flavor": "gpu-flavor / npu-flavor",
        "devices": [
            {"device_id": gpu[0], "device_type": "GPU", "product": gpu[1], "architecture": "Blackwell",
             "mem_total_mb": int(gpu[2]), "sm_total": 188, "lsu_capacity": cap_lsu, "pe_total": None,
             "capability": {"methods": ["SPACE_TIME"], "space_ratio_step": None, "time_ratio_step": None,
                            "min_space_ratio": None, "max_tenants": None, "mig_profiles": [], "pe_step": None,
                            "overcommit_allowed": True},
             "partitions": lsu_parts,
             "usage": {"free_lsu": cap_lsu - lsu_used, "active_workloads": len(lsu_parts),
                       "free_space_ratio": None, "free_time_ratio": None, "util": None,
                       "fragmentation": 0.0}},
            {"device_id": "npu0", "device_type": "NPU", "product": "RNGD", "architecture": "RNGD",
             "mem_total_mb": None, "pe_total": cap_pe,
             "capability": {"methods": ["NPU_PE"], "pe_step": 1, "overcommit_allowed": False},
             "partitions": pe_parts,
             "usage": {"free_pe": cap_pe - pe_used, "active_workloads": len(pe_parts)}}]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="", help="실행환경 정보 예시로 쓸 workload_id (없으면 최신 TRAIN)")
    a = ap.parse_args()

    wid = a.workload or next(g["workload_id"] for g in get(f"{QA}/api/v1/workloads?kind=train")["items"])
    save("exec-env-info.example.json", get(f"{QA}/api/v1/exec-env/{wid}"),
         f"실데이터 — {wid} (GET /api/v1/exec-env). 비어 있는 값은 수집 경로 미연결 (정의 파일의 now 참고)")

    dec = next((r for r in get(f"{SCHED}/decisions?limit=200")["items"]
                if r["decision"] == "APPROVED" and (r.get("resource_pool") or {}).get("state") == "RELEASED"), None)
    if dec:
        dec = {k: v for k, v in dec.items() if k not in ("kueue_uid", "repeat_count", "last_seen_at")}
        save("scheduler-decision.example.json", dec,
             "실데이터 — GPU(LSU) 승인 → 자원 풀 RESERVED → 종료 후 RELEASED 까지 기록된 레코드 (GET /decisions)")

    save("partition-set.example.json", partition_set(),
         "실데이터 — 선택한 노드 현재 상태 (allocatable, 올라간 Pod 요청, nvidia-smi). 공간·시간 비율은 KETRIS 연계 후, "
         "sm_total 은 자원 풀 실측값. 성균관대 형식 회신 전 KETI 안")


if __name__ == "__main__":
    main()
