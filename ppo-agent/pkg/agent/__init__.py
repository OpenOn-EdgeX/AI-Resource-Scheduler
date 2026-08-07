"""
KETI PPO Agent — 엣지 AI 과제 · v2 (lsu_amount 재설계)

역할 (엣지 AI 시스템 SW 과제):
  지능형 스케줄러는 다중 워크로드 큐의 요청을 기준으로 워크로드별 "논리 자원
  할당량(lsu_amount)"을 산출한다. 이 값은 KETRIS(= ERA 분할 비율 결정 모듈 +
  runtime/libbless) 정책 엔진이 시간/공간 자원 분할 비율을 결정하는 "입력값"이다.
  실제 SM Core 분할·커널 실행 제어·시공간 분할 적용은 KETRIS가 수행한다.
  → 스케줄러는 자원을 직접 제어하지 않고 판단값(lsu_amount)만 제공한다.

Actor-Critic:
- Actor : State → lsu_amount 평균(μ) + log_std (탐색 수준, 학습 가능 파라미터)
- Critic: State → V(s) (구조는 v1 유지, 입력 차원만 변경)

PPO 알고리즘 골격(GAE·Clipped Surrogate·K-epochs)은 v1과 동일하며,
action_dim 에 무관하게 동작한다(log_prob/entropy 를 마지막 차원 합산).
"""

import os
import csv
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# PyTorch import (optional - fallback to simple heuristic if not available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using heuristic-based allocation")

from ..config import (
    STATE_DIM, ACTION_DIM, LEARNING_RATE, GAMMA, CLIP_EPSILON,
    DEVICE_LSU_CAPACITY, LSU_MIN, MAX_LSU_REF, MAX_MEMORY_MB,
    BATCH_MAX, P95_LOG_REF_MS, THROUGHPUT_LOG_REF, KERNELS_REF,
    MODEL_PATH, HISTORY_PATH,
)

# ============================================================================
# PPO 하이퍼파라미터
# ============================================================================
GAE_LAMBDA = 0.95      # GAE lambda (bias-variance tradeoff)
K_EPOCHS = 4           # 같은 배치로 몇 번 학습할지
BATCH_SIZE = 32        # 최소 transition 수
ENTROPY_COEF = 0.01    # 엔트로피 보너스 계수 (탐색 장려)
VALUE_LOSS_COEF = 0.5  # Critic loss 가중치
MAX_GRAD_NORM = 0.5    # Gradient clipping

# Reward 가중치 (활용률/낭비 밸런스 — type-aware right-sizing 유도)
W_UTIL = 0.6    # 활용률: 목표점유(lsu/cap) vs 실측 GPU Util 근접 보상
W_WASTE = 0.8   # 낭비: (allocated - used) 페널티 — 과대할당 억제(타입별 right-sizing 선명화)

# priority 가중 (SLO·처리량 항에 곱함) — 자원 경합 시 고순위가 필요량에 더 붙고
# 저순위가 더 양보하는 최적점을 만든다 (SFR.WSS.006 긴급 우대의 학습적 구현).
# 값은 설계 상수(협의 가능). priority=1 이 기존과 동일(하위호환).
PRIORITY_WEIGHT = {0: 0.7, 1: 1.0, 2: 1.4}

# one-hot 인코딩 순서 (스키마 v3 · 5그룹)
WORKLOAD_KINDS = ("INFERENCE", "TRAINING")                       # ① 워크로드 종류
DEVICE_KINDS = ("GPU", "NPU")                                    # ③ 디바이스 종류
RESOURCE_BOUND_TYPES = ("COMPUTE_BOUND", "MEMORY_BOUND", "MIXED")  # ⑤ 자원 바운드 특성
# (구 workload_type → resource_bound_type 로 개명: ①workload_kind 와 이름 충돌 제거,
#  런타임 커널 특성 분석 모듈의 WORKLOAD_TYPES 관례값과 동일)
WORKLOAD_TYPES = RESOURCE_BOUND_TYPES  # 하위호환 alias

MODE_FREE, MODE_OVERCOMMIT = "FREE", "OVERCOMMIT"


@dataclass
class AllocationRequest:
    """
    자원 할당 요청 — State 5그룹 원천값 (스키마 v3, 22차원).

    각 필드의 "실연동" 주석은 실 클러스터 통합 시 실제로 값을 받아야 하는 모듈이다.
    현재(학습/단독 동작)에는 시뮬레이터/휴리스틱이 mock 값을 채운다.
    """
    # ── ① 워크로드 정보 (사용자/큐에서 전달) ────────────────────────────────
    requested_lsu: float                 # 최소 보장 요청량(LSU), 1 이상 필수. lsu=action×requested
                                         # 파라미터화라 상한 역할 겸함  # 실연동: 고려대 워크로드 큐
    workload_kind: str = "INFERENCE"     # INFERENCE/TRAINING  # 실연동: 워크로드 큐 메타데이터
    priority: int = 1                    # 0=LOW 1=MED 2=HIGH  # 실연동: 워크로드 큐 / scheduler.py 관례

    # ── ② 프로파일링 결과 (연세대 I/F 대응 — 현재 mock) ─────────────────────
    p95_latency_ms: float = 0.0          # p95 지연  # 실연동: 연세대 프로파일링 I/F
    batch_size: int = 1                  # 1~16  # 실연동: 연세대 프로파일링 I/F
    throughput: float = 0.0              # img/s 또는 tokens/s (단위 혼재 → build_state 에서 정규화)
                                         # 실연동: 연세대 프로파일링 I/F
    slo_target_throughput: float = 0.0   # SLO 목표 throughput (같은 단위). 있으면 비율 정규화.
                                         # 연세대 I/F 협의 필요 (신규 필드 요청)
    peak_memory_mb: float = 0.0          # 피크 메모리  # 실연동: 연세대 프로파일링 I/F

    # ── ③ 선택된 디바이스·파티션 정보 (성균관대 매칭 결과 대응) ─────────────
    device_kind: str = "GPU"             # GPU/NPU  # 실연동: 성균관대 디바이스 매칭 결과
    partition_sm_fraction: float = 1.0   # 선택 파티션 SM / 디바이스 총 SM (0-1)
                                         # 실연동: 성균관대 파티셔닝 모듈 (파티션 ID 대신 스펙 비율)
    device_lsu_capacity: float = DEVICE_LSU_CAPACITY  # 장비 고정값(실측 174)
                                         # 실연동: LSU 디바이스 카탈로그

    # ── ④ 현재 가용 자원 상태 (자원풀 매니저가 채울 자리 — 현재 mock) ───────
    free_lsu: float = DEVICE_LSU_CAPACITY  # 현재 가용 LSU  # 실연동: 자원풀 매니저 (2단계)
    overcommit_ratio: float = 1.0        # virtual/physical SM 비  # 실연동: 자원풀 매니저 (2단계)
    active_tenants: int = 0              # 활성 워크로드 수  # 실연동: 자원풀 매니저 (2단계)

    # ── ⑤ 모델별 실행 이력 (KETRIS 실측 결과 저장→재사용 — 현재 mock) ───────
    kernels_per_iter: int = 0            # iter 당 커널 수  # 실연동: KETRIS 실측 이력
                                         # (실측 코드: 런타임 커널 특성 분석 모듈)
    mps_pct: float = 0.0                 # MPS thread % (0-100). MPS 미구성 시 → mock
                                         # 실연동: KETRIS(controller/registry._calc_mps_pct) 이력
    mode: str = MODE_FREE                # FREE/OVERCOMMIT  # 실연동: 자원풀 매니저
                                         # (계산식: controller/overcommit.py check_and_update)
    lsu_est: float = 0.0                 # 실행 이력 없는 신규 모델의 사전 추정 대체값
                                         # 실연동: 정적 LSU 추정 모듈(lsu_est)
    resource_bound_type: str = "MIXED"   # COMPUTE_BOUND/MEMORY_BOUND/MIXED
                                         # 실연동: KETRIS 실측 이력 (런타임 커널 특성 분석)


@dataclass
class AllocationResponse:
    """자원 할당 응답 — KETRIS 로 넘길 판단값."""
    lsu_amount: float         # 논리 자원 할당량(LSU) — KETRIS 정책 엔진 입력값
    confidence: float         # 결정 신뢰도(0-1)
    reason: str               # 결정 이유


@dataclass
class Experience:
    """PPO 학습용 transition (on-policy)."""
    state: np.ndarray
    action: np.ndarray        # 정규화된 행동 [0-1] (shape=(1,))
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


# ============================================================================
# State 구성 — 5그룹 22차원 (단일 소스 오브 트루스, 스키마 v3)
# ============================================================================
def _norm_throughput(throughput: float, slo_target: float) -> float:
    """
    throughput 정규화 — 단위(img/s vs tokens/s)가 모델마다 다르므로:
      1) slo_target_throughput 이 있으면 목표 대비 비율(무차원): clip(x/target,0,2)/2
         # 연세대 I/F 협의 필요 (신규 필드 요청 — slo_target_throughput)
      2) 없으면 log 폴백: log1p(x)/log1p(THROUGHPUT_LOG_REF)
    """
    if slo_target and slo_target > 0:
        return min(max(throughput / slo_target, 0.0), 2.0) / 2.0
    return min(np.log1p(max(throughput, 0.0)) / np.log1p(THROUGHPUT_LOG_REF), 1.0)


def build_state(request: AllocationRequest) -> np.ndarray:
    """
    AllocationRequest → 정규화 State 벡터(22차원, 5그룹).

    ①워크로드(큐) ②프로파일링(연세대 I/F) ③디바이스·파티션(성균관대 매칭)
    ④가용 자원(자원풀 매니저) ⑤모델별 실행 이력(KETRIS 실측 저장→재사용)
    """
    cap = max(request.device_lsu_capacity, 1e-6)
    wk, dk, rb = request.workload_kind, request.device_kind, request.resource_bound_type

    return np.array([
        # ── ① 워크로드 정보 (4) ──
        min(max(request.requested_lsu, LSU_MIN) / cap, 1.0),   # 0  요청 LSU 비율(1 이상 강제)
        1.0 if wk == "INFERENCE" else 0.0,                     # 1  one-hot 추론
        1.0 if wk == "TRAINING" else 0.0,                      # 2  one-hot 학습
        request.priority / 2.0,                                # 3  우선순위(0/0.5/1)
        # ── ② 프로파일링 결과 (4) — 실연동: 연세대 I/F (현재 mock) ──
        min(np.log1p(max(request.p95_latency_ms, 0.0)) / np.log1p(P95_LOG_REF_MS), 1.0),  # 4
        (min(max(request.batch_size, 1), BATCH_MAX) - 1) / (BATCH_MAX - 1),               # 5
        _norm_throughput(request.throughput, request.slo_target_throughput),              # 6
        min(max(request.peak_memory_mb, 0.0) / MAX_MEMORY_MB, 1.0),                       # 7
        # ── ③ 디바이스·파티션 (4) — 실연동: 성균관대 매칭 결과 ──
        1.0 if dk == "GPU" else 0.0,                           # 8  one-hot GPU
        1.0 if dk == "NPU" else 0.0,                           # 9  one-hot NPU
        min(max(request.partition_sm_fraction, 0.0), 1.0),     # 10 파티션 SM 비율
        min(request.device_lsu_capacity / MAX_LSU_REF, 1.0),   # 11 device 용량(이기종 정규화)
        # ── ④ 가용 자원 상태 (3) — 실연동: 자원풀 매니저 (현재 mock) ──
        min(max(request.free_lsu / cap, 0.0), 1.0),            # 12 가용 LSU 비율
        min(max(request.overcommit_ratio, 0.0), 2.0) / 2.0,    # 13 오버커밋 비(0-2→0-1)
        min(request.active_tenants / 10.0, 1.0),               # 14 활성 워크로드 수
        # ── ⑤ 모델별 실행 이력 (7) — 실연동: KETRIS 실측 이력 (현재 mock) ──
        min(request.kernels_per_iter / KERNELS_REF, 1.0),      # 15 커널 수
        min(max(request.mps_pct, 0.0), 100.0) / 100.0,         # 16 MPS % (미구성 시 → mock)
        1.0 if request.mode == MODE_OVERCOMMIT else 0.0,       # 17 mode (FREE=0/OVER=1)
        min(max(request.lsu_est, 0.0) / cap, 1.0),             # 18 사전 LSU 추정(이력 폴백)
        1.0 if rb == "COMPUTE_BOUND" else 0.0,                 # 19 one-hot COMPUTE
        1.0 if rb == "MEMORY_BOUND" else 0.0,                  # 20 one-hot MEMORY
        1.0 if rb == "MIXED" else 0.0,                         # 21 one-hot MIXED
    ], dtype=np.float32)


if TORCH_AVAILABLE:
    class ActorNetwork(nn.Module):
        """
        Actor (Policy) Network.

        입력: State (STATE_DIM)
        출력: lsu_amount 의 평균 μ(s) ∈ [0,1] (정규화) 와 표준편차 σ

        연속 행동: π(a|s) = N(μ(s), σ), σ = exp(log_std) (학습 가능, 상태 무관).
        v1 대비 mean/log_std 의 출력 차원만 2 → 1 로 축소.
        """
        def __init__(self, state_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.mean = nn.Linear(hidden_dim, action_dim)      # v1: action_dim=2 → v2: 1
            self.log_std = nn.Parameter(torch.zeros(action_dim))  # 탐색 수준(학습 가능) — v1 유지

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.shared(state)
            mean = torch.sigmoid(self.mean(x))                 # [0,1] 정규화
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std

        def get_distribution(self, state: torch.Tensor) -> Normal:
            mean, std = self.forward(state)
            return Normal(mean, std)

        def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            dist = self.get_distribution(states)
            log_prob = dist.log_prob(actions).sum(dim=-1)      # 마지막 차원 합(dim=1이면 스칼라와 동일)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy

    class CriticNetwork(nn.Module):
        """Critic (Value) Network — 구조 v1 유지, 입력 차원(STATE_DIM)만 변경."""
        def __init__(self, state_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state)


class PPOAgent:
    """
    PPO Agent — lsu_amount 산출.

    PPO 골격(Clipped Surrogate / GAE / K-epochs)은 v1과 동일.
    변경점: 네트워크 출력 차원(1), State 구성(build_state), Action 변환(→LSU),
            confidence 기반 보수적 할당(lsu 기준), Reward 3기준 재설계.
    """

    def __init__(self):
        self.use_torch = TORCH_AVAILABLE

        if self.use_torch:
            self.actor = ActorNetwork(STATE_DIM, ACTION_DIM)
            self.critic = CriticNetwork(STATE_DIM)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
            self._load_model()
        else:
            self.actor = None
            self.critic = None

        self.experiences: List[Experience] = []
        self.training_stats = {
            'total_updates': 0,
            'total_experiences': 0,
            'avg_reward': 0.0,
            'avg_actor_loss': 0.0,
            'avg_critic_loss': 0.0,
        }
        logger.info(f"PPOAgent initialized (torch={self.use_torch}, "
                    f"state_dim={STATE_DIM}, action_dim={ACTION_DIM}, cap={DEVICE_LSU_CAPACITY} LSU)")

    # ── 모델 저장/로드 ────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, weights_only=True)
                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                if 'training_stats' in checkpoint:
                    self.training_stats = checkpoint['training_stats']
                logger.info(f"Loaded model from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

    def save_model(self):
        if not self.use_torch:
            return
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'training_stats': self.training_stats,
            }, MODEL_PATH)
            logger.info(f"Saved model to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    # ── Action → LSU 변환 + 보수적 할당 ──────────────────────────────────────
    def _action_to_lsu(self, action_scalar: float, std_mean: float,
                       request: AllocationRequest) -> Tuple[float, float, float]:
        """
        정규화 행동[0,1] → lsu_amount(LSU) 변환 + confidence 기반 보수적 할당.

        파라미터화: action = "요청 대비 할당 비율" → base = action × requested_lsu
          → 요청 초과가 구조적으로 불가(over-allocation 방지). 적정 action 은 워크로드
            타입별 상수(예: COMPUTE≈1.0 / MIXED≈0.8 / MEMORY≈0.6)로 학습된다.
        보수적 할당: 확신도(confidence=1/(1+σ)) 낮을수록 하한(LSU_MIN)쪽으로 축소.
        반환: (base_lsu, lsu_amount, confidence)
        """
        base = float(action_scalar) * max(request.requested_lsu, LSU_MIN)
        confidence = float(1.0 / (1.0 + std_mean))
        conservative = LSU_MIN + (base - LSU_MIN) * confidence   # 낮은 confidence → 보수적
        lsu_amount = max(LSU_MIN, conservative)
        return base, lsu_amount, confidence

    # ── 추론 ──────────────────────────────────────────────────────────────────
    def get_allocation(self, request: AllocationRequest) -> AllocationResponse:
        if self.use_torch:
            return self._get_allocation_ppo(request)
        return self._get_allocation_heuristic(request)

    def _get_allocation_ppo(self, request: AllocationRequest) -> AllocationResponse:
        """PPO 기반 할당(추론: 결정론적 mean 사용)."""
        state = build_state(request)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        logger.info("=" * 60)
        logger.info("[PPO Inference] lsu_amount 결정")
        logger.info(f"  [INPUT] requested_lsu={request.requested_lsu:.1f}, kind={request.workload_kind}, "
                    f"bound={request.resource_bound_type}, prio={request.priority}, lsu_est={request.lsu_est:.1f}")
        logger.info(f"  [INPUT] free_lsu={request.free_lsu:.1f}, overcommit={request.overcommit_ratio:.2f}, "
                    f"tenants={request.active_tenants}, cap={request.device_lsu_capacity:.0f}")

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            value = self.critic(state_tensor).item()
            action = float(mean.squeeze().item())              # 결정론적 mean

        std_mean = float(std.mean().item())
        base_lsu, lsu_amount, confidence = self._action_to_lsu(action, std_mean, request)

        logger.info(f"  [ACTOR] mean={action:.4f}, std={std_mean:.4f} | [CRITIC] V(s)={value:.4f}")
        logger.info(f"  [CONVERT] base={base_lsu:.1f} LSU (action×req) → conservative → {lsu_amount:.1f} LSU")
        logger.info(f"  [RESULT] lsu_amount={lsu_amount:.1f} LSU, confidence={confidence:.4f}")
        logger.info("=" * 60)

        return AllocationResponse(
            lsu_amount=round(lsu_amount, 2),
            confidence=confidence,
            reason=(f"PPO decision (req={request.requested_lsu:.0f}, bound={request.resource_bound_type}, "
                    f"free={request.free_lsu:.0f}, tenants={request.active_tenants})")
        )

    def _get_allocation_heuristic(self, request: AllocationRequest) -> AllocationResponse:
        """휴리스틱 할당(PyTorch 없을 때) — 부하·가용량 반영 축소."""
        load_factor = max(0.4, 1.0 - request.overcommit_ratio * 0.2)
        pod_factor = max(0.5, 1.0 - request.active_tenants * 0.1)
        lsu = request.requested_lsu * load_factor * pod_factor
        lsu = min(lsu, request.requested_lsu, max(request.free_lsu, LSU_MIN))
        lsu = max(LSU_MIN, lsu)
        return AllocationResponse(
            lsu_amount=round(lsu, 2),
            confidence=0.7,
            reason=f"Heuristic (load={load_factor:.2f}, pods={request.active_tenants})"
        )

    # ── 학습용 행동 선택(탐색 포함) ──────────────────────────────────────────
    def select_action_for_training(self, request: AllocationRequest) -> Tuple[AllocationResponse, Optional[Experience]]:
        if not self.use_torch:
            return self._get_allocation_heuristic(request), None

        state = build_state(request)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            dist = self.actor.get_distribution(state_tensor)
            action = dist.sample()                             # 탐색 샘플링
            log_prob = dist.log_prob(action).sum().item()
            value = self.critic(state_tensor).item()
            _, std = self.actor(state_tensor)
            action_scalar = float(action.squeeze().item())
            std_mean = float(std.mean().item())

        # 샘플은 [0,1] 밖으로 나갈 수 있으므로 클리핑 후 변환
        action_clipped = min(max(action_scalar, 0.0), 1.0)
        _, lsu_amount, _ = self._action_to_lsu(action_clipped, std_mean, request)

        response = AllocationResponse(
            lsu_amount=round(lsu_amount, 2),
            confidence=0.5,                                     # 탐색 중이므로 낮음
            reason="PPO training exploration"
        )
        partial_exp = Experience(
            state=state,
            action=np.array([action_scalar], dtype=np.float32),  # 정규화 행동 저장(학습 일관성)
            reward=0.0,
            next_state=state,
            done=False,
            log_prob=log_prob,
            value=value,
        )
        return response, partial_exp

    def record_experience(self, experience: Experience):
        self.experiences.append(experience)
        self.training_stats['total_experiences'] += 1
        logger.debug(f"Recorded experience (total: {len(self.experiences)})")

    # ========================================================================
    # PPO 학습 핵심 (v1 과 동일 — action_dim 무관)
    # ========================================================================
    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + GAMMA * next_values[t] - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train_step(self) -> Optional[Dict]:
        if not self.use_torch:
            return None
        if len(self.experiences) < BATCH_SIZE:
            logger.debug(f"Not enough experiences: {len(self.experiences)} < {BATCH_SIZE}")
            return None

        logger.info(f"=== PPO Training Start (experiences: {len(self.experiences)}) ===")

        states = torch.tensor(np.array([e.state for e in self.experiences]), dtype=torch.float32)
        actions = torch.tensor(np.array([e.action for e in self.experiences]), dtype=torch.float32)
        rewards = [e.reward for e in self.experiences]
        old_log_probs = torch.tensor([e.log_prob for e in self.experiences], dtype=torch.float32)
        values = [e.value for e in self.experiences]
        dones = [e.done for e in self.experiences]

        next_states = torch.tensor(np.array([e.next_state for e in self.experiences]), dtype=torch.float32)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1).tolist()
            if not isinstance(next_values, list):
                next_values = [next_values]

        avg_reward = sum(rewards) / len(rewards)
        logger.info(f"  Average reward: {avg_reward:.4f}")

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        logger.info(f"  Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

        total_actor_loss = total_critic_loss = total_entropy = 0.0
        for epoch in range(K_EPOCHS):
            new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()
            current_values = self.critic(states).squeeze(-1)
            critic_loss = nn.functional.mse_loss(current_values, returns)
            loss = actor_loss + VALUE_LOSS_COEF * critic_loss + ENTROPY_COEF * entropy_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            clipped = (ratio < 1 - CLIP_EPSILON) | (ratio > 1 + CLIP_EPSILON)
            logger.info(f"  [EPOCH {epoch+1}/{K_EPOCHS}] actor={actor_loss.item():.4f} "
                        f"critic={critic_loss.item():.4f} entropy={entropy.mean().item():.4f} "
                        f"clip_frac={clipped.float().mean().item():.2%}")

        self.training_stats['total_updates'] += 1
        self.training_stats['avg_reward'] = avg_reward
        self.training_stats['avg_actor_loss'] = total_actor_loss / K_EPOCHS
        self.training_stats['avg_critic_loss'] = total_critic_loss / K_EPOCHS
        self.experiences = []
        self.save_model()

        result = {
            'actor_loss': total_actor_loss / K_EPOCHS,
            'critic_loss': total_critic_loss / K_EPOCHS,
            'entropy': total_entropy / K_EPOCHS,
            'avg_reward': avg_reward,
            'updates': self.training_stats['total_updates'],
        }
        logger.info(f"=== PPO Training Complete (updates={result['updates']}) ===")
        self._save_history(result)
        return result

    def _save_history(self, result: Dict):
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH) or '.', exist_ok=True)
            file_exists = os.path.exists(HISTORY_PATH)
            with open(HISTORY_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'round', 'avg_reward', 'actor_loss', 'critic_loss', 'entropy'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'round': result['updates'],
                    'avg_reward': round(result['avg_reward'], 4),
                    'actor_loss': round(result['actor_loss'], 4),
                    'critic_loss': round(result['critic_loss'], 4),
                    'entropy': round(result['entropy'], 4),
                })
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    # ========================================================================
    # Reward — 3기준 재설계 (활용률 / 단편화 / SLO) + 데이터 출처 명시
    # ========================================================================
    @staticmethod
    def compute_reward(lsu_amount: float,
                       device_lsu_capacity: float,
                       actual_gpu_util: float,
                       slo_met: bool,
                       throughput_ratio: float = 1.0,
                       active_tenants: int = 1,
                       requested_lsu: float = 0.0,
                       priority: int = 1) -> float:
        """
        Reward — 할당 결정 품질 평가. 3기준 + 데이터 출처.

        1) 자원 활용률 (W_UTIL):
           목표 점유 = lsu_amount / capacity, 실측 GPU Util 과의 차이가 작을수록 +
           # 실연동: KETRIS 모니터링 exec_rate_pct (또는 DCGM GPU Util)
        2) 자원 낭비/단편화 (W_WASTE):
           할당했지만 실제로 안 쓴 양 = lsu_amount − used_lsu 를 페널티.
           used_lsu = actual_gpu_util × capacity (실측 활용률 기반).
           → 과대할당(요청/타입 대비 과다)일수록 waste↑ → type-aware right-sizing 유도.
           # 실연동: 실사용 = KETRIS 모니터링 GPU Util × capacity,
           #         단편화 구조는 KETRIS SM 상태·overcommit 모듈로 정밀화
        3) SLO 충족 (weight 1.0):
           워크로드 throughput/latency 목표 충족 시 +, 미충족 시 - (+ throughput 보너스)
           # 실연동: KETRIS 모니터링 + tests(iter/s) + LSU 실측 latency_ms_per_batch

        Args (모두 실측/시뮬 mock):
            lsu_amount           : Action 결과(LSU)
            device_lsu_capacity  : 장비 총 LSU
            actual_gpu_util      : 실측 GPU 사용률(0-1)  # monitor → 실사용 LSU 환산에도 사용
            slo_met              : SLO 충족 여부           # monitor/exporter
            throughput_ratio     : 실측/목표 throughput(0-1)
            active_tenants       : 동시 워크로드 수        # shm.tenant_count
            requested_lsu        : 요청량(과소할당 판정용)
        """
        cap = max(device_lsu_capacity, 1e-6)
        util = min(max(actual_gpu_util, 0.0), 1.0)
        reward = 0.0

        # 1) 자원 활용률 — 목표 점유 vs 실측 GPU Util 근접도
        util_target = min(lsu_amount / cap, 1.0)
        efficiency = 1.0 - abs(util - util_target)             # 1.0 이면 완벽 정합
        reward += W_UTIL * max(efficiency, 0.0)

        # 2) 자원 낭비 — 할당했지만 실제로 안 쓴 양(allocated - used) 페널티
        used_lsu = util * cap
        waste_ratio = min(max((lsu_amount - used_lsu) / cap, 0.0), 1.0)
        reward -= W_WASTE * waste_ratio

        # 3) SLO/throughput — throughput 비례 연속 보상(주) + 소폭 이진 SLO(보조)
        #    이진 cliff(±) 대신 throughput_ratio 비례로 gradient 를 부드럽게 하여
        #    적정 할당점(과소=throughput↓ / 과대=waste↑)이 매끈한 단봉 최적이 되게 함.
        #    priority 가중: 고순위일수록 SLO 달성의 보상/미달의 손실이 커짐.
        pw = PRIORITY_WEIGHT.get(int(priority), 1.0)
        reward += pw * 1.0 * min(max(throughput_ratio, 0.0), 1.0)
        reward += (0.2 * pw) if slo_met else (-0.2 * pw)

        # (보조) 공정성 — 동시 워크로드 대비 독점 방지
        if active_tenants > 1:
            fair_share = cap / active_tenants
            if lsu_amount > fair_share * 1.5:
                reward -= 0.3

        # (보조) 과소 할당 페널티 — 요청 대비 지나치게 적으면
        if requested_lsu > 0 and lsu_amount < requested_lsu * 0.1:
            reward -= 0.2

        return reward


__all__ = [
    "PPOAgent", "AllocationRequest", "AllocationResponse", "Experience",
    "build_state", "BATCH_SIZE",
    "WORKLOAD_KINDS", "DEVICE_KINDS", "RESOURCE_BOUND_TYPES", "WORKLOAD_TYPES",
    "MODE_FREE", "MODE_OVERCOMMIT",
]
