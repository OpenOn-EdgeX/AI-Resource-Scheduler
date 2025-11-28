"""
KETI PPO Agent - Proximal Policy Optimization for GPU Resource Allocation

Actor-Critic 구조:
- Actor (Policy Network): 상태를 보고 자원 할당 결정
- Critic (Value Network): 상태의 가치 추정

PPO 알고리즘:
1. 환경과 상호작용하여 경험 수집 (state, action, reward, log_prob, value)
2. GAE (Generalized Advantage Estimation)로 Advantage 계산
3. Clipped Surrogate Objective로 정책 업데이트
4. Value Function 업데이트
"""

import os
import logging
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field

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
    MIN_CORES_PERCENT, MAX_CORES_PERCENT, MIN_MEMORY_MB, MAX_MEMORY_MB,
    MODEL_PATH
)

# ============================================================================
# PPO 하이퍼파라미터
# ============================================================================
GAE_LAMBDA = 0.95      # GAE lambda (bias-variance tradeoff)
K_EPOCHS = 4           # 같은 배치로 몇 번 학습할지
BATCH_SIZE = 32        # 최소 경험 수
ENTROPY_COEF = 0.01    # 엔트로피 보너스 계수 (탐색 장려)
VALUE_LOSS_COEF = 0.5  # Critic loss 가중치
MAX_GRAD_NORM = 0.5    # Gradient clipping


@dataclass
class AllocationRequest:
    """자원 할당 요청"""
    requested_cores: int      # 요청한 GPU 코어 %
    requested_memory: int     # 요청한 메모리 MB
    node_gpu_util: float      # 노드 현재 GPU 사용률 (0-1)
    node_mem_util: float      # 노드 현재 메모리 사용률 (0-1)
    running_pods: int         # 현재 실행 중인 GPU Pod 수


@dataclass
class AllocationResponse:
    """자원 할당 응답"""
    allocated_cores: int      # 할당할 GPU 코어 %
    allocated_memory: int     # 할당할 메모리 MB
    confidence: float         # 결정 신뢰도 (0-1)
    reason: str               # 결정 이유


@dataclass
class Experience:
    """
    PPO 학습을 위한 경험 데이터

    PPO는 on-policy 알고리즘이므로 현재 정책으로 수집한 데이터만 사용
    """
    state: np.ndarray         # 상태 s
    action: np.ndarray        # 행동 a (정규화된 값 0-1)
    reward: float             # 보상 r
    next_state: np.ndarray    # 다음 상태 s'
    done: bool                # 에피소드 종료 여부
    log_prob: float           # π(a|s)의 로그 확률 (PPO 비율 계산용)
    value: float              # V(s) - Critic의 가치 추정 (Advantage 계산용)


if TORCH_AVAILABLE:
    class ActorNetwork(nn.Module):
        """
        Actor (Policy) Network - 정책 네트워크

        입력: 상태 벡터 [요청코어, 요청메모리, GPU사용률, 메모리사용률, Pod수]
        출력: 행동의 평균(mean)과 표준편차(std)

        연속 행동 공간에서 Gaussian Policy 사용:
        - π(a|s) = N(μ(s), σ)
        - μ(s): 상태에 따라 변하는 평균 (네트워크 출력)
        - σ: 학습 가능한 파라미터 (상태 무관)
        """
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
            super().__init__()
            # 공유 레이어 (특징 추출)
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            # 평균 출력 레이어
            self.mean = nn.Linear(hidden_dim, action_dim)
            # 로그 표준편차 (학습 가능한 파라미터, 상태 무관)
            # log_std를 학습하면 항상 양수인 std = exp(log_std) 보장
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            순전파: 상태 → (평균, 표준편차)
            """
            x = self.shared(state)
            mean = torch.sigmoid(self.mean(x))  # 0-1 범위로 정규화
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std

        def get_distribution(self, state: torch.Tensor) -> Normal:
            """
            상태에서 행동 분포 반환
            """
            mean, std = self.forward(state)
            return Normal(mean, std)

        def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            주어진 상태-행동 쌍의 log_prob과 entropy 계산

            PPO 학습에서 사용:
            - log_prob: 정책 비율(ratio) 계산용
            - entropy: 탐색 보너스용
            """
            dist = self.get_distribution(states)
            log_prob = dist.log_prob(actions).sum(dim=-1)  # 각 행동 차원의 log_prob 합
            entropy = dist.entropy().sum(dim=-1)  # 엔트로피도 합
            return log_prob, entropy

    class CriticNetwork(nn.Module):
        """
        Critic (Value) Network - 가치 네트워크

        입력: 상태 벡터
        출력: 상태 가치 V(s) - 스칼라 값

        V(s) = E[R_t | s_t = s]
        현재 상태에서 기대되는 미래 보상의 총합
        """
        def __init__(self, state_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # 스칼라 출력
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state)


class PPOAgent:
    """
    PPO Agent for GPU Resource Allocation

    ============================================================================
    PPO (Proximal Policy Optimization) 알고리즘 설명
    ============================================================================

    1. 목표: 정책 π를 개선하여 기대 보상 최대화
       J(π) = E[Σ γ^t * r_t]

    2. Policy Gradient의 문제점:
       - 업데이트가 너무 크면 성능 붕괴 (catastrophic forgetting)
       - 업데이트가 너무 작으면 학습 느림

    3. PPO 해결책: Clipped Surrogate Objective
       - 정책 변화량을 제한하여 안정적 학습
       - ratio = π_new(a|s) / π_old(a|s)
       - clip(ratio, 1-ε, 1+ε) 로 비율 제한

    4. Advantage 함수 A(s,a):
       - A(s,a) = Q(s,a) - V(s)
       - "이 행동이 평균보다 얼마나 좋은가?"
       - GAE로 계산: A_t = Σ (γλ)^l * δ_{t+l}
         where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """

    def __init__(self):
        self.use_torch = TORCH_AVAILABLE

        if self.use_torch:
            self.actor = ActorNetwork(STATE_DIM, ACTION_DIM)
            self.critic = CriticNetwork(STATE_DIM)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

            # 저장된 모델 로드 시도
            self._load_model()
        else:
            self.actor = None
            self.critic = None

        # 경험 버퍼 (학습용) - Experience 객체 리스트
        self.experiences: List[Experience] = []

        # 학습 통계
        self.training_stats = {
            'total_updates': 0,
            'total_experiences': 0,
            'avg_reward': 0.0,
            'avg_actor_loss': 0.0,
            'avg_critic_loss': 0.0,
        }

        logger.info(f"PPOAgent initialized (torch={self.use_torch})")

    def _load_model(self):
        """저장된 모델 로드"""
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
        """모델 저장"""
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

    def get_allocation(self, request: AllocationRequest) -> AllocationResponse:
        """자원 할당 결정 (추론)"""
        if self.use_torch:
            return self._get_allocation_ppo(request)
        else:
            return self._get_allocation_heuristic(request)

    def _get_allocation_ppo(self, request: AllocationRequest) -> AllocationResponse:
        """
        PPO 기반 할당 결정 (추론 모드)

        학습 시에는 탐색을 위해 분포에서 샘플링하지만,
        실제 서비스에서는 평균값(결정론적)을 사용
        """
        state = self._request_to_state(request)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        logger.info("=" * 60)
        logger.info("[PPO Inference] 할당 결정 시작")
        logger.info(f"  [INPUT] requested_cores={request.requested_cores}%, "
                    f"requested_memory={request.requested_memory}MB")
        logger.info(f"  [INPUT] node_gpu_util={request.node_gpu_util:.2%}, "
                    f"node_mem_util={request.node_mem_util:.2%}, "
                    f"running_pods={request.running_pods}")
        logger.info(f"  [STATE] vector={state.tolist()}")

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            value = self.critic(state_tensor).item()
            # 추론 시에는 결정론적으로 평균 사용
            action = mean.squeeze().numpy()

        logger.info(f"  [ACTOR] mean={mean.squeeze().tolist()}, std={std.squeeze().tolist()}")
        logger.info(f"  [CRITIC] V(s)={value:.4f}")
        logger.info(f"  [ACTION] raw(0-1)={action.tolist()}")

        # 행동을 실제 값으로 변환 (0-1 → 실제 범위)
        raw_cores = action[0] * (MAX_CORES_PERCENT - MIN_CORES_PERCENT) + MIN_CORES_PERCENT
        raw_memory = action[1] * (MAX_MEMORY_MB - MIN_MEMORY_MB) + MIN_MEMORY_MB
        cores_percent = int(raw_cores)
        memory_mb = int(raw_memory)

        logger.info(f"  [CONVERT] raw_cores={raw_cores:.1f}%, raw_memory={raw_memory:.0f}MB")

        # 원본 요청 대비 조정 (요청보다 많이 할당하지 않음)
        cores_before_cap = cores_percent
        mem_before_cap = memory_mb
        cores_percent = min(cores_percent, request.requested_cores)
        memory_mb = min(memory_mb, request.requested_memory)

        # 최소값 보장
        cores_percent = max(cores_percent, MIN_CORES_PERCENT)
        memory_mb = max(memory_mb, MIN_MEMORY_MB)

        if cores_before_cap != cores_percent or mem_before_cap != memory_mb:
            logger.info(f"  [CAP] cores: {cores_before_cap}% -> {cores_percent}% "
                       f"(max={request.requested_cores}%), "
                       f"memory: {mem_before_cap}MB -> {memory_mb}MB "
                       f"(max={request.requested_memory}MB)")

        confidence = float(1.0 / (1.0 + std.mean().item()))

        logger.info(f"  [RESULT] allocated_cores={cores_percent}%, "
                    f"allocated_memory={memory_mb}MB, confidence={confidence:.4f}")
        logger.info("=" * 60)

        return AllocationResponse(
            allocated_cores=cores_percent,
            allocated_memory=memory_mb,
            confidence=confidence,
            reason=f"PPO decision (gpu_util={request.node_gpu_util:.2f}, pods={request.running_pods})"
        )


__all__ = ['PPOAgent', 'AllocationRequest', 'AllocationResponse', 'Experience']
