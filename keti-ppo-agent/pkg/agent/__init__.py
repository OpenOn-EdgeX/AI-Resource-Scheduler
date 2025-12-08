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

    def _get_allocation_heuristic(self, request: AllocationRequest) -> AllocationResponse:
        """휴리스틱 기반 할당 결정 (PyTorch 없을 때)"""
        gpu_factor = 1.0 - (request.node_gpu_util * 0.3)
        mem_factor = 1.0 - (request.node_mem_util * 0.3)
        pod_factor = max(0.5, 1.0 - (request.running_pods * 0.1))

        cores_percent = int(request.requested_cores * gpu_factor * pod_factor)
        memory_mb = int(request.requested_memory * mem_factor * pod_factor)

        cores_percent = max(MIN_CORES_PERCENT, min(MAX_CORES_PERCENT, cores_percent))
        memory_mb = max(MIN_MEMORY_MB, min(MAX_MEMORY_MB, memory_mb))

        return AllocationResponse(
            allocated_cores=cores_percent,
            allocated_memory=memory_mb,
            confidence=0.7,
            reason=f"Heuristic (gpu_util={request.node_gpu_util:.2f}, factor={gpu_factor:.2f})"
        )

    def _request_to_state(self, request: AllocationRequest) -> np.ndarray:
        """요청을 상태 벡터로 변환 (정규화)"""
        """인풋값으로 requested_cores 는 요청한 sm값  """
        return np.array([
            request.requested_cores / 100.0,
            request.requested_memory / MAX_MEMORY_MB,
            request.node_gpu_util,
            request.node_mem_util,
            min(request.running_pods / 10.0, 1.0)
        ], dtype=np.float32)

    # ========================================================================
    # 경험 수집 (학습 데이터)
    # ========================================================================

    def select_action_for_training(self, request: AllocationRequest) -> Tuple[AllocationResponse, Experience]:
        """
        학습을 위한 행동 선택 (탐색 포함)

        추론과 달리 분포에서 샘플링하여 탐색(exploration) 수행
        log_prob과 value도 함께 반환하여 나중에 학습에 사용
        """
        if not self.use_torch:
            response = self._get_allocation_heuristic(request)
            return response, None

        state = self._request_to_state(request)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            # Actor에서 분포 얻기
            dist = self.actor.get_distribution(state_tensor)

            # 분포에서 샘플링 (탐색!)
            action = dist.sample()

            # log_prob 계산 (PPO ratio 계산에 필요)
            log_prob = dist.log_prob(action).sum().item()

            # Critic에서 V(s) 얻기 (Advantage 계산에 필요)
            value = self.critic(state_tensor).item()

            action = action.squeeze().numpy()

        # 행동을 실제 값으로 변환
        cores_percent = int(action[0] * (MAX_CORES_PERCENT - MIN_CORES_PERCENT) + MIN_CORES_PERCENT)
        memory_mb = int(action[1] * (MAX_MEMORY_MB - MIN_MEMORY_MB) + MIN_MEMORY_MB)

        cores_percent = min(max(cores_percent, MIN_CORES_PERCENT), request.requested_cores)
        memory_mb = min(max(memory_mb, MIN_MEMORY_MB), request.requested_memory)

        response = AllocationResponse(
            allocated_cores=cores_percent,
            allocated_memory=memory_mb,
            confidence=0.5,  # 탐색 중이므로 신뢰도 낮음
            reason="PPO training exploration"
        )

        # 부분 경험 생성 (reward와 next_state는 나중에 채움)
        partial_exp = Experience(
            state=state,
            action=action,  # 정규화된 값 (0-1)
            reward=0.0,     # 나중에 채움
            next_state=state,  # 나중에 채움
            done=False,
            log_prob=log_prob,
            value=value
        )

        return response, partial_exp

    def record_experience(self, experience: Experience):
        """
        완성된 경험 기록

        reward와 next_state가 채워진 후 호출
        """
        self.experiences.append(experience)
        self.training_stats['total_experiences'] += 1
        logger.debug(f"Recorded experience (total: {len(self.experiences)})")

    # ========================================================================
    # PPO 학습 핵심 로직
    # ========================================================================

    def _compute_gae(self, rewards: List[float], values: List[float],
                     next_values: List[float], dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE (Generalized Advantage Estimation) 계산

        ========================================================================
        GAE 수식:
        ========================================================================

        δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
              ~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~
              보상   다음 상태의 할인된 가치            현재 가치

        이것이 TD Error (Temporal Difference Error)
        "예상보다 얼마나 좋았나/나빴나"

        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

        - λ=0: A_t = δ_t (1-step TD, high bias, low variance)
        - λ=1: A_t = Σ γ^l * r_{t+l} - V(s_t) (Monte Carlo, low bias, high variance)
        - 0<λ<1: bias-variance tradeoff

        Returns:
            advantages: A_t 값들
            returns: R_t = A_t + V(s_t) (Critic 학습 타겟)
        """
        advantages = []
        gae = 0.0

        # 역순으로 계산 (미래 → 과거)
        for t in reversed(range(len(rewards))):
            # TD Error: δ_t = r_t + γV(s') - V(s)
            # done이면 다음 상태 가치 = 0
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta  # 에피소드 끝나면 GAE 리셋
            else:
                delta = rewards[t] + GAMMA * next_values[t] - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Returns = Advantages + Values (Critic 학습 타겟)
        returns = advantages + torch.tensor(values, dtype=torch.float32)

        # Advantage 정규화 (학습 안정성)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(self) -> Optional[Dict]:
        """
        PPO 학습 스텝

        ========================================================================
        PPO 알고리즘 상세
        ========================================================================

        1. 경험 버퍼에서 데이터 추출
        2. GAE로 Advantage 계산
        3. K epochs 동안 같은 데이터로 학습:

           a) Policy Loss (Actor):
              ratio = π_new(a|s) / π_old(a|s) = exp(log_prob_new - log_prob_old)

              L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

              - A > 0 (좋은 행동): ratio 증가 시키고 싶지만 1+ε로 제한
              - A < 0 (나쁜 행동): ratio 감소 시키고 싶지만 1-ε로 제한

           b) Value Loss (Critic):
              L_V = (V(s) - R_t)^2

              R_t = A_t + V_old(s) (타겟)

           c) Entropy Bonus:
              L_E = -H(π) = -Σ π(a|s) log π(a|s)

              엔트로피가 높으면 → 탐색 많이 함
              보너스로 주면 → 탐색 장려

           d) 총 Loss:
              L = -L_CLIP + c1 * L_V - c2 * L_E

        4. Gradient Clipping 적용
        5. 모델 저장
        """
        if not self.use_torch:
            return None
        # 슈도코드상에서 T Time Steps가 Batch Size라는 뜻
        if len(self.experiences) < BATCH_SIZE:
            logger.debug(f"Not enough experiences: {len(self.experiences)} < {BATCH_SIZE}")
            return None

        logger.info(f"=== PPO Training Start (experiences: {len(self.experiences)}) ===")

        # --------------------------------------------------------------------
        # 1. 경험 데이터 추출
        # --------------------------------------------------------------------
        states = torch.tensor(np.array([e.state for e in self.experiences]), dtype=torch.float32)
        actions = torch.tensor(np.array([e.action for e in self.experiences]), dtype=torch.float32)
        rewards = [e.reward for e in self.experiences]
        old_log_probs = torch.tensor([e.log_prob for e in self.experiences], dtype=torch.float32)
        values = [e.value for e in self.experiences]
        dones = [e.done for e in self.experiences]

        # next_state의 V(s') 계산
        next_states = torch.tensor(np.array([e.next_state for e in self.experiences]), dtype=torch.float32)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze().tolist()
            if not isinstance(next_values, list):
                next_values = [next_values]

        # 평균 보상 로깅
        avg_reward = sum(rewards) / len(rewards)
        logger.info(f"  Average reward: {avg_reward:.4f}")

        # --------------------------------------------------------------------
        # 2. GAE로 Advantage 계산
        # --------------------------------------------------------------------
        advantages, returns = self._compute_gae(rewards, values, next_values, dones)

        logger.info(f"  Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

        # --------------------------------------------------------------------
        # 3. K Epochs 학습
        # --------------------------------------------------------------------
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for epoch in range(K_EPOCHS):
            # a) 새 정책에서 log_prob, entropy 계산
            new_log_probs, entropy = self.actor.evaluate_actions(states, actions)

            # b) Policy Ratio 계산
            # ratio = π_new / π_old = exp(log π_new - log π_old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # c) Clipped Surrogate Objective
            # L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
            surr1 = ratio * advantages                                    # ratio * A
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages  # clipped

            # 최소값 취하고 음수로 (gradient ascent → descent)
            actor_loss = -torch.min(surr1, surr2).mean()

            # d) Entropy Bonus (탐색 장려)
            entropy_loss = -entropy.mean()  # 음수로 해서 최대화

            # e) Value Loss (MSE)
            current_values = self.critic(states).squeeze()
            critic_loss = nn.functional.mse_loss(current_values, returns)

            # f) 총 Loss
            loss = actor_loss + VALUE_LOSS_COEF * critic_loss + ENTROPY_COEF * entropy_loss

            # g) Actor 업데이트
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (폭발 방지)
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

            # 클리핑 비율 모니터링
            clipped = (ratio < 1 - CLIP_EPSILON) | (ratio > 1 + CLIP_EPSILON)
            clip_fraction = clipped.float().mean().item()

            logger.debug(f"  Epoch {epoch+1}/{K_EPOCHS}: "
                        f"actor_loss={actor_loss.item():.4f}, "
                        f"critic_loss={critic_loss.item():.4f}, "
                        f"entropy={entropy.mean().item():.4f}, "
                        f"clip_frac={clip_fraction:.2%}")

        # --------------------------------------------------------------------
        # 4. 통계 업데이트 및 저장
        # --------------------------------------------------------------------
        self.training_stats['total_updates'] += 1
        self.training_stats['avg_reward'] = avg_reward
        self.training_stats['avg_actor_loss'] = total_actor_loss / K_EPOCHS
        self.training_stats['avg_critic_loss'] = total_critic_loss / K_EPOCHS

        # 경험 버퍼 클리어 (on-policy이므로)
        self.experiences = []

        # 모델 저장
        self.save_model()

        result = {
            'actor_loss': total_actor_loss / K_EPOCHS,
            'critic_loss': total_critic_loss / K_EPOCHS,
            'entropy': total_entropy / K_EPOCHS,
            'avg_reward': avg_reward,
            'updates': self.training_stats['total_updates'],
        }

        logger.info(f"=== PPO Training Complete ===")
        logger.info(f"  Actor Loss: {result['actor_loss']:.4f}")
        logger.info(f"  Critic Loss: {result['critic_loss']:.4f}")
        logger.info(f"  Entropy: {result['entropy']:.4f}")
        logger.info(f"  Total Updates: {result['updates']}")

        return result

    # ========================================================================
    # Reward 함수 (GPU 자원 할당용)
    # ========================================================================

    @staticmethod
    def compute_reward(allocated_cores: int, allocated_memory: int,
                       actual_usage_pct: float, qos_met: bool,
                       total_pods: int) -> float:
        """
        Reward 함수 - 할당 결정의 품질 평가

        ========================================================================
        Reward 설계 원칙:
        ========================================================================

        1. 자원 효율성 (Efficiency):
           - 할당한 만큼 실제로 사용하면 좋음
           - waste = allocated - actual_usage
           - efficiency_reward = -waste (낭비 페널티)

        2. QoS 만족도 (Quality of Service):
           - 워크로드가 필요한 성능 달성했는지
           - qos_reward = +1 if satisfied, -1 otherwise

        3. 공정성 (Fairness):
           - 여러 Pod이 있을 때 자원 독점 방지
           - fairness_penalty = -α if allocated > fair_share

        Args:
            allocated_cores: 할당한 GPU 코어 %
            allocated_memory: 할당한 메모리 MB
            actual_usage_pct: 실제 GPU 사용률 (0-100)
            qos_met: QoS 조건 만족 여부
            total_pods: 전체 GPU Pod 수

        Returns:
            reward: 스칼라 보상 값
        """
        reward = 0.0

        # 1. 효율성 보상: 할당 대비 사용률
        # 이상적: actual_usage ≈ allocated
        if allocated_cores > 0:
            efficiency = actual_usage_pct / allocated_cores
            efficiency = min(efficiency, 1.0)  # 최대 1.0
            reward += efficiency * 0.5  # 가중치 0.5

        # 2. QoS 보상
        if qos_met:
            reward += 1.0
        else:
            reward -= 0.5  # QoS 미충족 페널티

        # 3. 공정성: Pod당 공평한 할당 기준
        if total_pods > 1:
            fair_share = 100.0 / total_pods
            if allocated_cores > fair_share * 1.5:  # 공평 할당의 1.5배 초과시
                reward -= 0.3  # 독점 페널티

        # 4. 과소 할당 페널티 (너무 적게 주면)
        if allocated_cores < 10:
            reward -= 0.2

        return reward


__all__ = ["PPOAgent", "AllocationRequest", "AllocationResponse", "Experience"]
