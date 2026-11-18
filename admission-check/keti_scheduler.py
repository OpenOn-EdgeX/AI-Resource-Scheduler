#!/usr/bin/env python3
"""keti-scheduler — Kueue AdmissionCheck 컨트롤러 (P4 골격, 안 v0.1).

Kueue 가 쿼터를 잡아 둔 Workload 에 대해 승인(Ready)·보류(Pending/Retry)를 정하고, 판단마다 결정 이력을 남긴다.
메시지 정의: proto/scheduler_decision.proto. 화면(p.41~42)은 keti-queue-api 가 GET /decisions 를 읽어 보여준다.

  Filter  : Workload 에 배정된 ResourceFlavor 의 노드 중, 요청 자원이 '실제로' 남아 있는 노드
            (Kueue 쿼터는 가상값이라 cohort 빌려쓰기·다른 과제 Pod 때문에 물리 여유와 다를 수 있다)
  Score   : KETI 자체 요소 — 배치 후 여유 비율, 배치 후 단편화, SLO 여유(데이터 없으면 제외)
            성균관대 1·2회차 입력(PartitionPlan)이 생기면 skku_score 로 더한다 (fetch_plan, 지금은 없음)
  결정    : 후보 있음 → Ready (+ podSetUpdates: 계획 노드 nodeSelector, decision-id 라벨)
            후보 없음 → Retry  (RESOURCE_WAIT, 쿼터 반납 후 재대기)
            프로파일 대기 → Pending (PROFILE_WAIT, 쿼터 유지) — 대기 시간이 넘으면 KETI 스코어로 폴백
  적용 결과: 승인 후 Pod 가 뜨면 실제 노드를 같은 레코드에 붙인다 (p.42 계획 대비 실제)
  자원 풀  : GPU(LSU) 승인 → UpdateAllocationDecision RESERVED, Workload 종료·삭제 → RELEASED (C-RP-03)
             RPM_ADDR 미설정·무응답이면 승인은 그대로 하고 레코드에 오류만 남긴다 (자원 풀 장애가 제출을 막지 않음)

프로파일 대기 데모: Pod 템플릿 annotation keti.re.kr/profile-wait-seconds: "N" → 최대 N 초 PROFILE_WAIT 후 폴백.
"""
import json
import logging
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("keti-scheduler")

CONTROLLER = "keti.re.kr/scheduler"
CHECK = os.environ.get("CHECK_NAME", "keti-scheduler")
FIELD_MANAGER = "keti-scheduler"
POLL = float(os.environ.get("POLL_SECONDS", "2"))
DATA = os.environ.get("DATA_DIR", "/data")
INFLIGHT = float(os.environ.get("INFLIGHT_SECONDS", "180"))  # 승인 후 Pod 가 노드에 붙기 전까지 예약으로 셈
RETRY_AFTER = int(os.environ.get("RETRY_AFTER_SECONDS", "20"))  # 공간 부족 보류 후 Kueue 재대기 간격
W_FREE = float(os.environ.get("W_FREE", "0.4"))
W_FRAG = float(os.environ.get("W_FRAG", "0.6"))
KQ = "/apis/kueue.x-k8s.io/v1beta2"

L_WID = "keti.re.kr/workload-id"
L_KIND = "keti.re.kr/workload-kind"
L_DECISION = "keti.re.kr/decision-id"
A_PLAN_NODE = "keti.re.kr/plan-node"
A_PROFILE_WAIT = "keti.re.kr/profile-wait-seconds"
A_SLO_LATENCY = "keti.re.kr/slo-latency-ms"
BASE_RES = ("cpu", "memory", "pods", "ephemeral-storage")
LSU = "keti.re.kr/lsu"
RPM_ADDR = os.environ.get("RPM_ADDR", "")   # 자원 풀 매니저 gRPC (예: resource-pool.example.invalid:50052). 비우면 연계 안 함


# ── 수량 ──────────────────────────────────────────────────────────────────
_Q = re.compile(r"^([0-9.]+)([a-zA-Z]*)$")
_UNIT = {"": 1, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
         "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}


def qty(s):
    m = _Q.match(str(s).strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _UNIT.get(m.group(2), 1)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── K8s ───────────────────────────────────────────────────────────────────
class Kube:
    def __init__(self):
        base = os.environ.get("KUBE_API")
        sa = "/var/run/secrets/kubernetes.io/serviceaccount"
        if base:
            self.base, self.headers, self.ctx = base.rstrip("/"), {}, None
        else:
            self.base = "https://kubernetes.default.svc"
            self.headers = {"Authorization": "Bearer " + open(f"{sa}/token").read()}
            self.ctx = ssl.create_default_context(cafile=f"{sa}/ca.crt")

    def call(self, method, path, body=None, ctype="application/json"):
        data = json.dumps(body).encode() if body is not None else None
        h = dict(self.headers)
        if data is not None:
            h["Content-Type"] = ctype
        req = urllib.request.Request(self.base + path, data=data, headers=h, method=method)
        with urllib.request.urlopen(req, context=self.ctx, timeout=5) as r:
            return json.load(r)

    def get(self, path):
        return self.call("GET", path)

    def list(self, path, selector=None):
        q = ("?labelSelector=" + urllib.parse.quote(selector)) if selector else ""
        return self.get(path + q).get("items", [])

    def apply_status(self, path, body):
        """server-side apply — 우리 필드(이 체크 항목)만 소유한다."""
        q = f"?fieldManager={FIELD_MANAGER}&force=true"
        return self.call("PATCH", path + "/status" + q, body, "application/apply-patch+yaml")


# ── 결정 이력 저장소 ──────────────────────────────────────────────────────
class Store:
    """JSONL 한 파일. 레코드 수가 작아 변경 시 통째로 다시 쓴다 (applied 를 같은 레코드에 붙이기 위해)."""

    def __init__(self, path):
        self.path, self.lock, self.items = path, threading.Lock(), []
        if os.path.exists(path):
            with open(path) as f:
                self.items = [json.loads(x) for x in f if x.strip()]

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for r in self.items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def last_for(self, kueue_uid):
        for r in reversed(self.items):
            if r.get("kueue_uid") == kueue_uid:
                return r
        return None

    def add(self, rec):
        with self.lock:
            self.items.append(rec)
            self._flush()

    def touch(self, rec):
        with self.lock:
            rec["repeat_count"] = rec.get("repeat_count", 1) + 1
            rec["last_seen_at"] = now_iso()
            self._flush()

    def set_applied(self, rec, applied):
        self.set(rec, "applied", applied)

    def set(self, rec, key, value):
        with self.lock:
            rec[key] = value
            self._flush()

    def query(self, workload_id=None, limit=100):
        with self.lock:
            out = [r for r in self.items if not workload_id or r["workload_id"] == workload_id]
        return list(reversed(out))[:limit]


# ── 입력 수집 ─────────────────────────────────────────────────────────────
def pod_requests(pod):
    tot = {}
    for c in pod["spec"].get("containers", []):
        for k, v in ((c.get("resources") or {}).get("requests") or {}).items():
            tot[k] = tot.get(k, 0.0) + qty(v)
    return tot


def node_usage(kube):
    """노드별 이미 올라간 Pod 의 요청 합 (종료된 Pod 제외)."""
    used = {}
    for p in kube.list("/api/v1/pods"):
        node = p["spec"].get("nodeName")
        if not node or p.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        u = used.setdefault(node, {})
        for k, v in pod_requests(p).items():
            u[k] = u.get(k, 0.0) + v
    return used


def add_reserved(usage, node, req):
    u = usage.setdefault(node, {})
    for k, v in req.items():
        u[k] = u.get(k, 0.0) + v


def reserve_inflight(usage, records, now_ts):
    """승인했지만 아직 Pod 가 노드에 안 붙은 몫을 사용량에 더한다 — 동시 승인으로 같은 여유를 두 번 주지 않게."""
    for r in records:
        if r["decision"] != "APPROVED" or r.get("applied") or not r.get("plan"):
            continue
        t = datetime.strptime(r["decided_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        if now_ts - t <= INFLIGHT:
            add_reserved(usage, r["plan"]["node_name"], r["input"]["requests"])


def flavor_nodes(kube, flavor, nodes):
    rf = kube.get(f"{KQ}/resourceflavors/{flavor}")
    sel = (rf.get("spec") or {}).get("nodeLabels") or {}
    return [n for n in nodes if all((n["metadata"].get("labels") or {}).get(k) == v for k, v in sel.items())]


def workload_inputs(wl):
    """Workload → (자원별 요청 합, 자원별 flavor, pod 템플릿 annotation 합, workload_kind)."""
    req, flv, annos, kind = {}, {}, {}, ""
    for psa in (wl["status"].get("admission") or {}).get("podSetAssignments") or []:
        for k, v in (psa.get("resourceUsage") or {}).items():
            req[k] = req.get(k, 0.0) + qty(v)
        flv.update(psa.get("flavors") or {})
    for ps in wl["spec"].get("podSets") or []:
        md = (ps.get("template") or {}).get("metadata") or {}
        annos.update(md.get("annotations") or {})
        kind = kind or (md.get("labels") or {}).get(L_KIND, "")
    return req, flv, annos, kind


# ── Filter → Score ────────────────────────────────────────────────────────
def primary_resource(req):
    acc = [r for r in req if r not in BASE_RES]
    return acc[0] if acc else "cpu"


def score_candidates(req, nodes_free):
    """nodes_free: {node: {res: (allocatable, requested)}} → (후보 [(node, breakdown)], 노드 스냅샷)."""
    snap = []
    for node, rs in nodes_free.items():
        for r, (alloc, used) in rs.items():
            snap.append({"node_name": node, "resource": r, "allocatable": alloc, "requested": used,
                         "free": alloc - used})
    fits = [n for n, rs in nodes_free.items()
            if all(rs.get(r, (0.0, 0.0))[0] - rs.get(r, (0.0, 0.0))[1] >= v for r, v in req.items())]
    pr = primary_resource(req)
    frees = {n: rs.get(pr, (0.0, 0.0))[0] - rs.get(pr, (0.0, 0.0))[1] for n, rs in nodes_free.items()}
    cands = []
    for n in fits:
        alloc = nodes_free[n].get(pr, (0.0, 0.0))[0]
        after = dict(frees)
        after[n] -= req.get(pr, 0.0)
        tot_free = sum(max(v, 0.0) for v in after.values())
        frag = 0.0 if tot_free <= 0 else 1 - max(after.values()) / tot_free
        free_space = after[n] / alloc if alloc else 0.0
        bd = {"free_space": round(free_space, 4), "fragmentation_after": round(frag, 4),
              "slo_margin": -1, "skku_score": -1,
              "total": round(W_FREE * free_space + W_FRAG * (1 - frag), 4)}
        cands.append((n, bd))
    cands.sort(key=lambda c: c[1]["total"], reverse=True)
    return cands, snap, pr


def fetch_plan(wl):
    """성균관대 계획안(1·2회차) 조회 자리 — 형식 회신 전까지 항상 None (KETI 스코어로 판단)."""
    return None


def decide(wl, nodes_free, now_ts):
    """순수 판단 함수 → dict(decision, reason, state, message, plan, input)."""
    req, flv, annos, kind = workload_inputs(wl)
    cands, snap, pr = score_candidates(req, nodes_free)
    inp = {"requests": {k: v for k, v in req.items()}, "flavors": flv, "nodes": snap,
           "workload_kind": {"train": "TRAIN", "infer": "INFER", "preprocess": "PREPROC"}.get(kind, "OTHER"),
           "slo_target_latency_ms": qty(annos.get(A_SLO_LATENCY, "0")), "profile_available": False}
    wait = qty(annos.get(A_PROFILE_WAIT, "0"))
    plan = fetch_plan(wl)
    reserved = _quota_reserved_at(wl)
    if plan is None and wait > 0 and now_ts - reserved < wait:
        return {"decision": "HELD", "reason": "PROFILE_WAIT", "state": "Pending", "input": inp, "plan": None,
                "plan_source": "KETI",
                "message": f"waiting for profile result ({int(wait - (now_ts - reserved))}s left, then KETI-score fallback)"}
    source = "KETI" if plan is None else plan.get("source", "SKKU_ROUND1")
    fallback = " (profile timeout → KETI score)" if wait > 0 and plan is None else ""
    if not cands:
        short = [s for s in snap if s["resource"] in req and s["free"] < req[s["resource"]]]
        what = ", ".join(f'{s["node_name"]} {s["resource"]} free {s["free"]:g} < {req[s["resource"]]:g}'
                         for s in short) or "no node in flavor"
        return {"decision": "HELD", "reason": "RESOURCE_WAIT", "state": "Retry", "input": inp, "plan": None,
                "plan_source": source, "message": f"insufficient free resources: {what}"}
    node, bd = cands[0]
    return {"decision": "APPROVED", "reason": "RESOURCE_FIT", "state": "Ready", "input": inp, "plan_source": source,
            "plan": {"node_name": node, "resource": pr, "amount": req.get(pr, 0.0), "score": bd,
                     "alternatives": [{"node_name": n, "score": b} for n, b in cands[1:3]]},
            "message": f"approved on {node} (score {bd['total']}){fallback}"}


def _quota_reserved_at(wl):
    for c in wl["status"].get("conditions") or []:
        if c["type"] == "QuotaReserved" and c["status"] == "True":
            return datetime.strptime(c["lastTransitionTime"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc).timestamp()
    return time.time()


# ── 자원 풀 연계 (C-RP-03 UpdateAllocationDecision) ─────────────────────────
class Pool:
    """resource_pool.proto 확정 계약 그대로. 키: workload_id = keti.re.kr/workload-id, pod_name = decision_id
    (Pod 이름은 승인 시점에 아직 없고, Pod 단위 서비스는 decision 마다 몫이 따로 잡히므로)."""

    def __init__(self, addr):
        self.addr, self.stub, self.pb = addr, None, None

    def _call(self, state, rec):
        if not self.addr:
            return {"state": "DISABLED"}
        try:
            if self.stub is None:
                import grpc
                from rpc import resource_pool_pb2 as pb, resource_pool_pb2_grpc as pbg
                self.pb, self.stub = pb, pbg.ResourcePoolStub(grpc.insecure_channel(self.addr))
            pb = self.pb
            r = self.stub.UpdateAllocationDecision(pb.UpdateAllocationDecisionRequest(
                workload_id=rec["workload_id"], device_id=rec["plan"]["node_name"], partition_id="",
                lsu_amount=float(rec["plan"]["amount"]), model_id=rec["workload_id"],
                pod_name=rec["decision_id"],
                allocation_state=pb.RESERVED if state == "RESERVED" else pb.RELEASED), timeout=2)
            return {"state": state if r.success else f"{state}_FAILED", "result_code": r.result_code,
                    "free_lsu": r.free_lsu, "virtual_sm": r.virtual_sm, "active_tenants": r.active_tenants,
                    "overcommit_ratio": round(r.overcommit_ratio, 4), "at": now_iso()}
        except Exception as e:  # 자원 풀 장애가 승인 흐름을 막지 않는다
            return {"state": f"{state}_ERROR", "error": str(e)[:200], "at": now_iso()}

    def reserve(self, rec):
        if (rec.get("plan") or {}).get("resource") != LSU:
            return {"state": "NOT_APPLICABLE", "reason": "LSU(GPU) 할당만 자원 풀 회계 대상"}
        return self._call("RESERVED", rec)

    def release(self, rec):
        return self._call("RELEASED", rec)


def to_release(records, live):
    """RESERVED 인데 Kueue Workload 가 끝났거나 사라진 레코드. live: {uid: workload}."""
    out = []
    for r in records:
        if (r.get("resource_pool") or {}).get("state") != "RESERVED":
            continue
        w = live.get(r.get("kueue_uid"))
        done = w is None or any(c["type"] == "Finished" and c["status"] == "True"
                                for c in (w.get("status") or {}).get("conditions") or [])
        if done:
            out.append(r)
    return out


# ── 컨트롤러 ──────────────────────────────────────────────────────────────
class Controller:
    def __init__(self, kube, store, pool=None):
        self.kube, self.store, self.pool = kube, store, pool or Pool(RPM_ADDR)

    def ensure_check_active(self):
        self.kube.apply_status(f"{KQ}/admissionchecks/{CHECK}", {
            "apiVersion": "kueue.x-k8s.io/v1beta2", "kind": "AdmissionCheck", "metadata": {"name": CHECK},
            "status": {"conditions": [{"type": "Active", "status": "True", "reason": "Active",
                                       "message": "keti-scheduler is running", "lastTransitionTime": now_iso()}]}})

    def our_check(self, wl):
        for c in (wl.get("status") or {}).get("admissionChecks") or []:
            if c["name"] == CHECK:
                return c
        return None

    def nodes_free(self, wl, nodes, usage):
        _, flv, _, _ = workload_inputs(wl)
        req, _, _, _ = workload_inputs(wl)
        out = {}
        for flavor in set(flv.values()):
            for n in flavor_nodes(self.kube, flavor, nodes):
                name = n["metadata"]["name"]
                alloc = n["status"].get("allocatable") or {}
                out[name] = {r: (qty(alloc.get(r, "0")), usage.get(name, {}).get(r, 0.0)) for r in req}
        return out

    def owner_labels(self, wl):
        ns = wl["metadata"]["namespace"]
        for o in wl["metadata"].get("ownerReferences") or []:
            path = {"Pod": f"/api/v1/namespaces/{ns}/pods/{o['name']}",
                    "Job": f"/apis/batch/v1/namespaces/{ns}/jobs/{o['name']}",
                    "TrainJob": f"/apis/trainer.kubeflow.org/v1alpha1/namespaces/{ns}/trainjobs/{o['name']}",
                    "JobSet": f"/apis/jobset.x-k8s.io/v1alpha2/namespaces/{ns}/jobsets/{o['name']}"}.get(o["kind"])
            if path:
                try:
                    return o["kind"], o["name"], self.kube.get(path)["metadata"].get("labels") or {}
                except urllib.error.HTTPError:
                    pass
        return "", "", {}

    def reconcile(self, wl, nodes, usage):
        meta = wl["metadata"]
        ns, name = meta["namespace"], meta["name"]
        okind, oname, olabels = self.owner_labels(wl)
        wid = olabels.get(L_WID) or f"kueue:{name}"
        d = decide(wl, self.nodes_free(wl, nodes, usage), time.time())

        last = self.store.last_for(meta["uid"])
        if last and (last["decision"], last["reason"]) == (d["decision"], d["reason"]) and d["decision"] == "HELD":
            self.store.touch(last)  # 같은 보류가 이어짐 — 이력 줄을 늘리지 않음
            rec = last
        else:
            rec = {"decision_id": "d-" + uuid.uuid4().hex[:12], "workload_id": wid, "kueue_workload": name,
                   "kueue_uid": meta["uid"], "namespace": ns, "owner_kind": okind, "owner_name": oname,
                   "decided_at": now_iso(), "decision": d["decision"], "reason": d["reason"],
                   "reason_text": d["message"], "plan_source": d["plan_source"], "plan_id": "",
                   "input": d["input"], "plan": d["plan"], "applied": None}
            self.store.add(rec)
        entry = {"name": CHECK, "state": d["state"], "message": d["message"][:1000],
                 "lastTransitionTime": now_iso()}
        if d["state"] == "Retry":
            entry["requeueAfterSeconds"] = RETRY_AFTER  # 없으면 Kueue 가 바로 다시 넣어 2초마다 재판정됨
        if d["state"] == "Ready":
            entry["podSetUpdates"] = [{
                "name": ps["name"],
                "labels": {L_DECISION: rec["decision_id"]},
                "annotations": {A_PLAN_NODE: d["plan"]["node_name"]},
                "nodeSelector": {"kubernetes.io/hostname": d["plan"]["node_name"]}}
                for ps in wl["spec"].get("podSets") or []]
        self.kube.apply_status(f"{KQ}/namespaces/{ns}/workloads/{name}", {
            "apiVersion": "kueue.x-k8s.io/v1beta2", "kind": "Workload",
            "metadata": {"name": name, "namespace": ns}, "status": {"admissionChecks": [entry]}})
        if d["decision"] == "APPROVED":
            add_reserved(usage, d["plan"]["node_name"], d["input"]["requests"])  # 같은 루프의 다음 판단에 반영
            if not rec.get("resource_pool"):
                self.store.set(rec, "resource_pool", self.pool.reserve(rec))
        log.info("%s/%s wid=%s → %s %s (%s)", ns, name, wid, d["decision"], d["reason"], d["message"])

    def track_applied(self):
        """승인했는데 적용 결과가 없는 레코드 → 실제 Pod 노드를 붙인다."""
        for rec in self.store.query(limit=200):
            if rec["decision"] != "APPROVED" or rec.get("applied"):
                continue
            pods = self.kube.list(f"/api/v1/namespaces/{rec['namespace']}/pods", f"{L_DECISION}={rec['decision_id']}")
            placed = [p for p in pods if p["spec"].get("nodeName")]
            if not placed:
                continue
            node = placed[0]["spec"]["nodeName"]
            self.store.set_applied(rec, {"observed_at": now_iso(), "node_name": node,
                                         "pods": [p["metadata"]["name"] for p in placed],
                                         "matches_plan": node == rec["plan"]["node_name"],
                                         "ketris_config_version": ""})

    def loop(self):
        self.ensure_check_active()
        last_active = time.time()
        while True:
            try:
                if time.time() - last_active > 60:
                    self.ensure_check_active()
                    last_active = time.time()
                all_wl = self.kube.list(f"{KQ}/workloads")
                todo = [w for w in all_wl
                        if (w.get("status") or {}).get("admission")
                        and (self.our_check(w) or {}).get("state") == "Pending"]
                if todo:
                    nodes, usage = self.kube.list("/api/v1/nodes"), node_usage(self.kube)
                    reserve_inflight(usage, self.store.query(limit=500), time.time())
                    for w in todo:
                        try:
                            self.reconcile(w, nodes, usage)
                        except Exception:
                            log.exception("reconcile 실패 %s", w["metadata"]["name"])
                self.track_applied()
                live = {w["metadata"]["uid"]: w for w in all_wl}
                for rec in to_release(self.store.query(limit=500), live):
                    res = self.pool.release(rec)
                    self.store.set(rec, "resource_pool", {**rec["resource_pool"], "released": res})
                    if res["state"] in ("RELEASED", "RELEASED_FAILED", "DISABLED"):
                        self.store.set(rec, "resource_pool", {**rec["resource_pool"], "state": "RELEASED"})
                    log.info("POOL_RELEASE wid=%s decision=%s -> %s", rec["workload_id"], rec["decision_id"], res["state"])
            except Exception:
                log.exception("loop 실패")
            time.sleep(POLL)


# ── HTTP (결정 이력 조회) ─────────────────────────────────────────────────
def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            if u.path == "/healthz":
                body, code = {"ok": True}, 200
            elif u.path == "/decisions":
                body = {"as_of": now_iso(), "items": store.query(q.get("workload_id"), int(q.get("limit", "100")))}
                code = 200
            else:
                body, code = {"error": "not found"}, 404
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(DATA, exist_ok=True)
    store = Store(os.path.join(DATA, "decisions.jsonl"))
    srv = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("check=%s controller=%s", CHECK, CONTROLLER)
    Controller(Kube(), store).loop()


if __name__ == "__main__":
    main()
