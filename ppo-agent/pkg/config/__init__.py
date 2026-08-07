"""
KETI PPO Agent Configuration (엣지 AI 과제 · v2 재설계)

v1 → v2 변경 요지:
- Action: [cores_percent, memory_mb] (dim=2)  →  lsu_amount 단일 스칼라 (dim=1)
- State : nvidia-smi 5차원  →  4카테고리(워크로드/프로파일링/디바이스·파티션/가용자원) 12차원
- 출력 단위: SM% / MB  →  LSU (Logical Slice Unit, 절대 단위)

용어: lsu_amount 는 "논리 자원 할당량"이며, KETRIS(ERA 분할 비율 결정 + 런타임 엔진)가
      이 값을 받아 시간/공간 분할 비율(virtual_sm, weight, round_duration)로 변환한다.
      지능형 스케줄러는 자원을 직접 제어하지 않고 판단값(lsu_amount)만 제공한다.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# API Server settings
# ─────────────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', '8081'))   # 실 배포 포트(8081)에 맞춤

# ─────────────────────────────────────────────────────────────────────────────
# PPO Model settings
# ─────────────────────────────────────────────────────────────────────────────
# spec 6: emptyDir 탈피 → 영구 경로. 온라인 에이전트 기본값도 영구 경로로.
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', './checkpoints')
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(CHECKPOINT_DIR, 'ppo_lsu_model.pt'))
HISTORY_PATH = os.environ.get('HISTORY_PATH', os.path.join(CHECKPOINT_DIR, 'training_history.csv'))

LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '3e-4'))
GAMMA = float(os.environ.get('GAMMA', '0.99'))            # Discount factor
CLIP_EPSILON = float(os.environ.get('CLIP_EPSILON', '0.2'))  # PPO clip range

# ─────────────────────────────────────────────────────────────────────────────
# State space (5 그룹 · 22차원)
#   ① 워크로드 정보     : requested_lsu, workload_kind(one-hot×2), priority       → 4
#   ② 프로파일링 결과   : p95_latency_ms, batch_size, throughput, peak_memory_mb  → 4
#   ③ 디바이스·파티션   : device_kind(one-hot×2), partition_sm_fraction,
#                          device_lsu_capacity                                     → 4
#   ④ 가용 자원 상태    : free_lsu, overcommit_ratio, active_tenants              → 3
#   ⑤ 모델별 실행 이력  : kernels_per_iter, mps_pct, mode, lsu_est,
#                          resource_bound_type(one-hot×3)                          → 7
#   실연동 소스는 pkg/agent build_state() 각 필드 주석 참조.
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM = int(os.environ.get('STATE_DIM', '22'))

# State 정규화 기준 상수
BATCH_MAX = int(os.environ.get('BATCH_MAX', '16'))                # batch_size 1~16
P95_LOG_REF_MS = float(os.environ.get('P95_LOG_REF_MS', '10000'))  # p95 log 스케일 상한(10s)
THROUGHPUT_LOG_REF = float(os.environ.get('THROUGHPUT_LOG_REF', '100000'))  # throughput log 폴백 상한
KERNELS_REF = float(os.environ.get('KERNELS_REF', '1000'))         # kernels_per_iter 정규화 상한

# ─────────────────────────────────────────────────────────────────────────────
# Action space — 단일 스칼라 lsu_amount (연속)
# ─────────────────────────────────────────────────────────────────────────────
ACTION_DIM = int(os.environ.get('ACTION_DIM', '1'))

# ─────────────────────────────────────────────────────────────────────────────
# LSU (Logical Slice Unit) 범위
#   DEVICE_LSU_CAPACITY: 이 노드 GPU 1장의 총 LSU.
#     실측 기준: NVIDIA RTX PRO 6000 Blackwell = 174 LSU
#     (2026-07-15 실측, resnet50 fp16, unit=42.16 img/s/LSU,
#      LSU 벤치마크 실측 결과 2026-07-15)
#     실연동 시: LSU 디바이스 카탈로그의 선택 device LSU 값으로 대체.
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_LSU_CAPACITY = float(os.environ.get('DEVICE_LSU_CAPACITY', '174'))
LSU_MIN = float(os.environ.get('LSU_MIN', '1'))        # 최소 할당 LSU
LSU_MAX = float(os.environ.get('LSU_MAX', str(DEVICE_LSU_CAPACITY)))  # 단일 워크로드 상한 = 장비 용량
# State 정규화용 기준 최대 LSU (이기종 device 비교용, A100=100·Blackwell=174 포괄)
MAX_LSU_REF = float(os.environ.get('MAX_LSU_REF', '256'))

# 메모리는 이제 Action이 아니라 State(워크로드 정보/프로파일링)의 입력값으로만 사용
MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', '98304'))  # Blackwell ~96GB

# ─────────────────────────────────────────────────────────────────────────────
# 자원풀 매니저 연동 (2단계) — ④⑤그룹 실값 소스. 응답 없으면 mock 폴백.
# ─────────────────────────────────────────────────────────────────────────────
POOL_MANAGER_URL = os.environ.get('POOL_MANAGER_URL', 'http://127.0.0.1:8082')
# 레이어 내부 통신: gRPC(C-RP-01~03) 기본. 'rest' 로 내리면 기존 REST 프로토타입 경로.
POOL_TRANSPORT = os.environ.get('POOL_TRANSPORT', 'grpc')
POOL_GRPC_TARGET = os.environ.get('POOL_GRPC_TARGET', '127.0.0.1:50052')
# C-IS-08: SliceAllocation CR 발행 (스케줄러 spec 작성 → KETRIS status 갱신)
SLICE_CR_ENABLED = os.environ.get('SLICE_CR_ENABLED', '1') == '1'

# ─────────────────────────────────────────────────────────────────────────────
# 오프라인 학습(SB3) 모델 서빙 — checkpoints_v3_state22 정식 학습본을 PPO.load 로
# 직접 로드하여 /allocate·/evaluate 의 할당 판단에 사용 (가중치 변환 없음).
# 로드 성공 시 온라인 에이전트(랜덤 초기화·탐색)는 할당 판단에서 배제되며,
# 온라인 PPO 업데이트도 수행하지 않는다. SB3_SERVING=0 으로 끄면 기존 동작.
# ─────────────────────────────────────────────────────────────────────────────
SB3_MODEL_PATH = os.environ.get(
    'SB3_MODEL_PATH',
    os.path.join(CHECKPOINT_DIR, 'sb3_ppo_ratio_prio.zip'))
SB3_SERVING = os.environ.get('SB3_SERVING', '1') == '1'
# 파라미터화: capacity_ratio(신규 — action=용량 대비 할당률, 학습 눈금 100) |
#             request_ratio(구 — action=요청 대비 비율, v3 모델용)
SB3_PARAM = os.environ.get('SB3_PARAM', 'capacity_ratio')
SB3_TRAIN_CAPACITY = float(os.environ.get('SB3_TRAIN_CAPACITY', '100'))

# ─────────────────────────────────────────────────────────────────────────────
# Node info
# ─────────────────────────────────────────────────────────────────────────────
NODE_NAME = os.environ.get('NODE_NAME', 'unknown')

# Logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


def get_config_summary():
    return {
        "api_host": API_HOST,
        "api_port": API_PORT,
        "model_path": MODEL_PATH,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "node_name": NODE_NAME,
        "device_lsu_capacity": DEVICE_LSU_CAPACITY,
        "lsu_range": f"{LSU_MIN}-{LSU_MAX} LSU",
    }
