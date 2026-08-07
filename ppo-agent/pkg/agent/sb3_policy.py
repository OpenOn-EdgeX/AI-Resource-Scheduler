"""
SB3 오프라인 학습 정책 서빙 래퍼 — 학습 완료 모델의 결정론 추론 전용.

파라미터화 2종 지원 (SB3_PARAM):
  capacity_ratio (기본, v4): action = 용량 대비 할당률.
    학습 눈금(용량 100)으로 입력을 환산해 추론하고, 실제 용량으로 되돌린다:
      요청·가용·추정치 × (100/실용량) → build_state → action
      lsu_amount = clip(action × 실용량, LSU_MIN, 실용량)
      lsu_amount = min(lsu_amount, requested_lsu)   ← 요청 초과 후처리 캡
    (LSU 값 자체는 다운스트림 계약(lsu_amount) 유지를 위해 계속 산출)
  request_ratio (구, v3): action = 요청 대비 비율.
      lsu_amount = clip(action × requested_lsu, LSU_MIN, 실용량)  — env.py 와 동일

confidence 는 학습된 log_std 로부터 1/(1+exp(log_std)) 산출 (보고용).
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
from typing import Tuple

import numpy as np

from ..config import (STATE_DIM, LSU_MIN, SB3_MODEL_PATH,
                      SB3_PARAM, SB3_TRAIN_CAPACITY)
from . import AllocationRequest, AllocationResponse, build_state

logger = logging.getLogger(__name__)


class SB3Policy:
    """오프라인 학습 SB3 PPO 모델의 결정론 서빙 어댑터 (추론 전용, 학습 없음)."""

    def __init__(self, model_path: str = SB3_MODEL_PATH, param: str = SB3_PARAM):
        from stable_baselines3 import PPO   # 지연 import — SB3 없으면 호출부에서 폴백

        self.model_path = model_path
        self.param = param
        self.train_capacity = SB3_TRAIN_CAPACITY
        self.model = PPO.load(model_path, device="cpu")

        obs_shape = tuple(self.model.observation_space.shape)
        if obs_shape != (STATE_DIM,):
            raise ValueError(
                f"State 차원 불일치: 모델 obs={obs_shape}, 서빙 STATE_DIM=({STATE_DIM},)")

        log_std = self.model.policy.log_std.detach().cpu().numpy()
        self.std = float(np.exp(log_std).mean())
        self.confidence = float(1.0 / (1.0 + self.std))

        with open(model_path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()[:8]
        self.model_version = (
            f"sb3-{param}-{self.model.num_timesteps}steps-{digest}")

        logger.info(f"[SB3Policy] loaded {model_path} "
                    f"(param={param}, version={self.model_version}, "
                    f"std={self.std:.4f}, confidence={self.confidence:.4f})")

    def predict_action(self, state: np.ndarray) -> float:
        a, _ = self.model.predict(state, deterministic=True)
        return float(np.clip(np.asarray(a).reshape(-1)[0], 0.0, 1.0))

    def decide(self, request: AllocationRequest
               ) -> Tuple[AllocationResponse, np.ndarray, float]:
        cap = max(request.device_lsu_capacity, LSU_MIN)

        if self.param == "capacity_ratio":
            # 학습 눈금(용량 100)으로 환산해 추론 — LSU 값 특징만 스케일,
            # 비율 특징(state[0]=req/cap 등)은 불변이라 분포가 학습과 일치한다
            s = self.train_capacity / cap
            scaled = dataclasses.replace(
                request,
                requested_lsu=request.requested_lsu * s,
                free_lsu=request.free_lsu * s,
                lsu_est=request.lsu_est * s,
                device_lsu_capacity=self.train_capacity)
            state = build_state(scaled)
            action = self.predict_action(state)          # = 용량 대비 할당률
            lsu_amount = float(np.clip(action * cap, LSU_MIN, cap))
            lsu_amount = min(lsu_amount, max(request.requested_lsu, LSU_MIN))
            reason = (f"SB3 capacity-ratio policy deterministic "
                      f"(ratio={action:.4f}, req={request.requested_lsu:.0f}, "
                      f"bound={request.resource_bound_type}, "
                      f"version={self.model_version})")
        else:  # request_ratio (v3 모델)
            state = build_state(request)
            action = self.predict_action(state)          # = 요청 대비 비율
            lsu_amount = float(np.clip(action * request.requested_lsu, LSU_MIN, cap))
            reason = (f"SB3 offline policy deterministic "
                      f"(req={request.requested_lsu:.0f}, "
                      f"bound={request.resource_bound_type}, "
                      f"version={self.model_version})")

        response = AllocationResponse(
            lsu_amount=round(lsu_amount, 2),
            confidence=round(self.confidence, 4),
            reason=reason,
        )
        return response, state, action


__all__ = ["SB3Policy"]
