"""
KETI PPO Agent - REST API Server

Webhook에서 호출하는 API 제공:
- POST /allocate: 자원 할당 요청
- POST /partition: MPS 파티션 ID 조회
- GET /health: 헬스 체크
- GET /status: 상태 조회
"""

import logging
import os
import subprocess
from flask import Flask, request, jsonify
from typing import Optional, Dict, List

from ..config import API_HOST, API_PORT, NODE_NAME, get_config_summary
from ..agent import PPOAgent, AllocationRequest, AllocationResponse

logger = logging.getLogger(__name__)


class PPOAgentAPI:
    """PPO Agent REST API Server"""

    def __init__(self, agent: PPOAgent):
        self.agent = agent
        self.app = Flask(__name__)
        self._setup_routes()
        self._request_count = 0

        # MPS 파티션 매핑: 이름(A,B,C) -> 파티션 ID
        self._partition_map: Dict[str, str] = {}
        self._load_partitions()

        logger.info("PPOAgentAPI initialized")

    def _load_partitions(self):
        """
        MPS lspart 명령으로 현재 파티션 목록을 조회하여
        A, B, C 순서로 매핑

        lspart 출력 예시:
        GPU-89  Dw8PDw8PDwAAAAAAAAAAAAAAAAAAAAAAAAAA  7  56  No
        GPU-89  AADw8AAAAA8PDw8PAAAAAAAAAAAAAAAAAAAA  7  56  No
        GPU-89  AAAAAPDw8PDw8PAwAAAAAAAAAAAAAAAAAAAA   8  64  No
        """
        try:
            # MPS pipe directory 환경변수 설정
            env = os.environ.copy()
            env.setdefault('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')

            result = subprocess.run(
                ['bash', '-c', 'echo lspart | nvidia-cuda-mps-control'],
                capture_output=True, text=True, timeout=5, env=env
            )
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning(f"MPS not running or no partitions found (rc={result.returncode}, stderr={result.stderr.strip()})")
                return

            logger.info(f"lspart raw output:\n{result.stdout.strip()}")

            # GPU UUID: 환경변수 우선, 없으면 nvidia-smi 시도
            gpu_uuid = os.environ.get('GPU_UUID', '')
            if not gpu_uuid:
                gpu_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=gpu_uuid', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5, env=env
                )
                gpu_uuid = gpu_result.stdout.strip() if gpu_result.returncode == 0 else ''

            # lspart 파싱: 파티션 ID 추출
            partition_ids = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    # 첫 번째 열은 GPU 축약 ID, 두 번째부터 파티션 정보
                    # 헤더 행과 총합 행(free/used 포함)은 스킵
                    candidate = parts[1]
                    # 파티션 ID는 긴 Base64 형태 문자열
                    if len(candidate) > 20 and candidate not in ('Partition', 'free', 'used', 'chunks', 'SM', 'clients'):
                        full_id = f"{gpu_uuid}/{candidate}"
                        partition_ids.append(full_id)

            # A, B, C 순서로 매핑
            names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for i, pid in enumerate(partition_ids):
                if i < len(names):
                    self._partition_map[names[i]] = pid
                    logger.info(f"Partition {names[i]} -> {pid}")

            logger.info(f"Loaded {len(self._partition_map)} partitions")

        except Exception as e:
            logger.warning(f"Failed to load partitions: {e}")

    def _setup_routes(self):
        """Setup Flask routes"""
        self.app.add_url_rule('/allocate', 'allocate', self.allocate, methods=['POST'])
        self.app.add_url_rule('/partition', 'partition', self.partition, methods=['POST'])
        self.app.add_url_rule('/partitions', 'partitions', self.list_partitions, methods=['GET'])
        self.app.add_url_rule('/health', 'health', self.health, methods=['GET'])
        self.app.add_url_rule('/status', 'status', self.status, methods=['GET'])
        self.app.add_url_rule('/feedback', 'feedback', self.feedback, methods=['POST'])


__all__ = ['PPOAgentAPI']
