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


__all__ = ['AllocationRequest', 'AllocationResponse', 'Experience']
