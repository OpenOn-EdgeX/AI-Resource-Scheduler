#!/bin/bash
# PPO Agent 학습 테스트 스크립트
# 사용법: ./scripts/test_training.sh

BASE="http://localhost:8081"

echo "=============================="
echo " PPO Agent 학습 테스트 시작"
echo "=============================="

# 학습 전 evaluate
echo ""
echo "[학습 전 evaluate]"
curl -s -X POST $BASE/evaluate \
  -H "Content-Type: application/json" \
  -d "{\"requested_cores\": 50, \"requested_memory\": 2048, \"gpu_util\": 0.3, \"mem_util\": 0.3, \"running_pods\": 2}"
echo ""

echo ""
echo "[32개 allocate + feedback 시작]"
echo ""

run() {
  POD=$1; CORES=$2; MEM=$3; GPU=$4; MEM_UTIL=$5; PODS=$6; PERF=$7; DONE=$8
  echo -n "  $POD → "
  curl -s -X POST $BASE/allocate -H "Content-Type: application/json" \
    -d "{\"pod_name\":\"$POD\",\"requested_cores\":$CORES,\"requested_memory\":$MEM,\"gpu_util\":$GPU,\"mem_util\":$MEM_UTIL,\"running_pods\":$PODS}" > /dev/null
  curl -s -X POST $BASE/feedback -H "Content-Type: application/json" \
    -d "{\"pod_name\":\"$POD\",\"allocated_cores\":$CORES,\"allocated_memory\":$MEM,\"actual_performance\":$PERF,\"completed\":$DONE}"
  echo ""
}

# 낮은 부하 / 좋은 성능
run pod-1  30  1024 0.1 0.1 1 0.90 true
run pod-2  20   512 0.0 0.1 0 0.95 true
run pod-3  40  2048 0.2 0.2 1 0.85 true
run pod-4  10   512 0.1 0.0 0 0.92 true
run pod-5  50  1024 0.2 0.1 2 0.88 true

# 중간 부하 / 양호한 성능
run pod-6  60  2048 0.4 0.3 2 0.80 true
run pod-7  50  4096 0.3 0.4 3 0.75 true
run pod-8  70  2048 0.5 0.4 2 0.78 true
run pod-9  40  3072 0.4 0.5 3 0.72 true
run pod-10 60  4096 0.5 0.3 2 0.82 true

# 중간 부하 / QoS 간신히 충족
run pod-11 80  4096 0.5 0.5 3 0.71 true
run pod-12 70  3072 0.6 0.4 4 0.73 true
run pod-13 50  2048 0.5 0.6 3 0.70 true
run pod-14 60  4096 0.4 0.5 2 0.74 true
run pod-15 90  8192 0.6 0.5 4 0.71 true

# 높은 부하 / QoS 미충족
run pod-16 100 8192 0.8 0.7 5 0.55 false
run pod-17 80  6144 0.9 0.8 6 0.50 false
run pod-18 90  8192 0.8 0.9 7 0.45 false
run pod-19 70  4096 0.9 0.7 5 0.60 false
run pod-20 100 8192 0.7 0.8 6 0.52 false

# 높은 부하 / 그래도 성공
run pod-21 80  6144 0.7 0.6 4 0.75 true
run pod-22 90  4096 0.8 0.7 5 0.72 true
run pod-23 70  8192 0.6 0.7 4 0.76 true

# 소규모 / 다양한 결과
run pod-24 20  1024 0.3 0.2 1 0.88 true
run pod-25 30   512 0.2 0.3 2 0.91 true
run pod-26 10  2048 0.1 0.2 1 0.93 true

# 과부하 시나리오
run pod-27 100 8192 0.9 0.9 8 0.40 false
run pod-28 90  6144 0.9 0.8 7 0.48 false

# 복구 시나리오
run pod-29 60  4096 0.6 0.5 3 0.80 true
run pod-30 40  2048 0.4 0.3 2 0.85 true
run pod-31 30  1024 0.2 0.2 1 0.90 true
run pod-32 20   512 0.1 0.1 0 0.95 true

echo ""
echo "[학습 후 evaluate]"
curl -s -X POST $BASE/evaluate \
  -H "Content-Type: application/json" \
  -d "{\"requested_cores\": 50, \"requested_memory\": 2048, \"gpu_util\": 0.3, \"mem_util\": 0.3, \"running_pods\": 2}"
echo ""

echo ""
echo "=============================="
echo " 테스트 완료"
echo "=============================="
