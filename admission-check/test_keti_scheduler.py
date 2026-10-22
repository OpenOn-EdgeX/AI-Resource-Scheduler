"""keti_scheduler.decide 단위 테스트 — python3 -m unittest test_keti_scheduler (admission-check/ 에서)."""
import time
import unittest

import keti_scheduler as s


def wl(usage, annos=None, kind="train", reserved_ago=0):
    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - reserved_ago))
    return {"metadata": {"name": "w", "namespace": "ns", "uid": "u"},
            "spec": {"podSets": [{"name": "main", "template": {"metadata": {
                "annotations": annos or {}, "labels": {s.L_KIND: kind}}}}]},
            "status": {"admission": {"podSetAssignments": [{"name": "main", "resourceUsage": usage,
                                                            "flavors": {k: "f" for k in usage}}]},
                       "conditions": [{"type": "QuotaReserved", "status": "True", "lastTransitionTime": t}]}}


class Decide(unittest.TestCase):
    def test_qty(self):
        self.assertEqual(s.qty("100m"), 0.1)
        self.assertEqual(s.qty("2Gi"), 2 * 2**30)
        self.assertEqual(s.qty("8"), 8)

    def test_approve_weights(self):
        # n1 여유 30, n2 여유 100 에 LSU 20 요청.
        # 기본 가중(여유 0.4 / 단편화 0.6): 여유 항 차이가 커서 분산(n2) — n1 0.585 vs n2 0.756
        nodes = {"n1": {"keti.re.kr/lsu": (100, 70)}, "n2": {"keti.re.kr/lsu": (100, 0)}}
        d = s.decide(wl({"keti.re.kr/lsu": "20"}), nodes, time.time())
        self.assertEqual((d["decision"], d["reason"], d["state"]), ("APPROVED", "RESOURCE_FIT", "Ready"))
        self.assertEqual(d["plan"]["node_name"], "n2")
        self.assertAlmostEqual(d["plan"]["score"]["total"], 0.7564, places=3)
        # 단편화만 보면 큰 여유를 남기는 best-fit(n1)
        s.W_FREE, s.W_FRAG = 0.0, 1.0
        try:
            self.assertEqual(s.decide(wl({"keti.re.kr/lsu": "20"}), nodes, time.time())["plan"]["node_name"], "n1")
        finally:
            s.W_FREE, s.W_FRAG = 0.4, 0.6
        self.assertEqual(d["input"]["workload_kind"], "TRAIN")
        self.assertEqual(len(d["plan"]["alternatives"]), 1)

    def test_resource_wait(self):
        nodes = {"n1": {"keti.re.kr/npu-pe": (8, 8)}}
        d = s.decide(wl({"keti.re.kr/npu-pe": "1"}), nodes, time.time())
        self.assertEqual((d["decision"], d["reason"], d["state"]), ("HELD", "RESOURCE_WAIT", "Retry"))
        self.assertIn("free 0 < 1", d["message"])

    def test_profile_wait_then_fallback(self):
        nodes = {"n1": {"cpu": (4, 0)}}
        a = {s.A_PROFILE_WAIT: "30"}
        d = s.decide(wl({"cpu": "1"}, a, reserved_ago=5), nodes, time.time())
        self.assertEqual((d["reason"], d["state"]), ("PROFILE_WAIT", "Pending"))
        self.assertIn("profile", d["message"])
        d = s.decide(wl({"cpu": "1"}, a, reserved_ago=40), nodes, time.time())
        self.assertEqual(d["decision"], "APPROVED")
        self.assertIn("KETI score", d["message"])

    def test_inflight_reserved(self):
        # 승인했지만 Pod 가 아직 안 붙은 LSU 200 → 남은 40 으로는 LSU 100 불가
        usage = {}
        rec = {"decision": "APPROVED", "applied": None, "decided_at": s.now_iso(),
               "plan": {"node_name": "n1"}, "input": {"requests": {"keti.re.kr/lsu": 200.0}}}
        old = dict(rec, decided_at="2020-01-01T00:00:00Z")      # 오래된 레코드는 무시
        done = dict(rec, applied={"node_name": "n1"})            # 이미 붙은 것은 node_usage 가 셈
        s.reserve_inflight(usage, [rec, old, done], time.time())
        self.assertEqual(usage, {"n1": {"keti.re.kr/lsu": 200.0}})
        nodes = {"n1": {"keti.re.kr/lsu": (240, usage["n1"]["keti.re.kr/lsu"])}}
        self.assertEqual(s.decide(wl({"keti.re.kr/lsu": "100"}), nodes, time.time())["reason"], "RESOURCE_WAIT")

    def test_pool_reserve_only_lsu_and_release_rule(self):
        p = s.Pool("")
        self.assertEqual(p.reserve({"plan": {"resource": "keti.re.kr/npu-pe"}})["state"], "NOT_APPLICABLE")
        self.assertEqual(p.reserve({"plan": {"resource": s.LSU, "node_name": "n", "amount": 10},
                                    "workload_id": "w", "decision_id": "d"})["state"], "DISABLED")
        fin = {"status": {"conditions": [{"type": "Finished", "status": "True"}]}}
        run = {"status": {"conditions": [{"type": "Admitted", "status": "True"}]}}
        recs = [{"kueue_uid": "a", "resource_pool": {"state": "RESERVED"}},
                {"kueue_uid": "b", "resource_pool": {"state": "RESERVED"}},
                {"kueue_uid": "c", "resource_pool": {"state": "RESERVED"}},
                {"kueue_uid": "d", "resource_pool": {"state": "NOT_APPLICABLE"}}]
        out = s.to_release(recs, {"a": fin, "b": run})          # c 는 사라짐
        self.assertEqual([r["kueue_uid"] for r in out], ["a", "c"])

    def test_pool_unreachable_does_not_raise(self):
        r = s.Pool("127.0.0.1:1")._call("RESERVED", {"workload_id": "w", "decision_id": "d",
                                                     "plan": {"node_name": "n", "amount": 1}})
        self.assertTrue(r["state"].startswith("RESERVED_"))

    def test_no_nodes(self):
        d = s.decide(wl({"cpu": "1"}), {}, time.time())
        self.assertEqual(d["reason"], "RESOURCE_WAIT")


if __name__ == "__main__":
    unittest.main()
