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

    def partition(self):
        """
        파티션 이름(A, B, C)을 MPS 파티션 ID로 변환

        Request body:
        {
            "partition_name": "A"
        }

        Response:
        {
            "partition_name": "A",
            "partition_id": "GPU-895722c8-.../Dw8PDw8PDwAAAAAAAAAAAAAAAAAAAAAAAAAA"
        }
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Empty request body"}), 400

            partition_name = data.get('partition_name', '').upper()

            if not partition_name:
                return jsonify({"error": "Missing partition_name"}), 400

            # 매핑이 비어있으면 다시 로드 시도
            if not self._partition_map:
                self._load_partitions()

            partition_id = self._partition_map.get(partition_name)

            if partition_id:
                logger.info(f"Partition lookup: {partition_name} -> {partition_id}")
                return jsonify({
                    "partition_name": partition_name,
                    "partition_id": partition_id
                })
            else:
                logger.warning(f"Partition '{partition_name}' not found. Available: {list(self._partition_map.keys())}")
                return jsonify({
                    "error": f"Partition '{partition_name}' not found",
                    "available": list(self._partition_map.keys())
                }), 404

        except Exception as e:
            logger.error(f"Error processing partition request: {e}")
            return jsonify({"error": str(e)}), 500

    def list_partitions(self):
        """
        현재 노드의 MPS 파티션 목록 반환

        Response:
        {
            "partitions": {"A": "GPU-.../xxx", "B": "GPU-.../yyy", "C": "GPU-.../zzz"},
            "count": 3
        }
        """
        # 매핑이 비어있으면 다시 로드
        if not self._partition_map:
            self._load_partitions()

        return jsonify({
            "partitions": self._partition_map,
            "count": len(self._partition_map),
            "node": NODE_NAME
        })

    def health(self):
        """Health check endpoint"""
        return jsonify({"status": "healthy", "node": NODE_NAME})

    def status(self):
        """Status endpoint"""
        return jsonify({
            "status": "running",
            "node": NODE_NAME,
            "config": get_config_summary(),
            "request_count": self._request_count,
            "gpu_status": self._get_gpu_status()
        })

    def allocate(self):
        """
        자원 할당 요청 처리

        Request body:
        {
            "requested_cores": 80,      # 요청 GPU 코어 %
            "requested_memory": 4000,   # 요청 메모리 MB
            "pod_name": "workload-a",   # Pod 이름 (선택)
            "namespace": "default"      # Namespace (선택)
        }

        Response:
        {
            "allocated_cores": 60,
            "allocated_memory": 3000,
            "confidence": 0.85,
            "reason": "PPO decision ..."
        }
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Empty request body"}), 400

            # 필수 필드 확인
            requested_cores = data.get('requested_cores')
            requested_memory = data.get('requested_memory')

            if requested_cores is None or requested_memory is None:
                return jsonify({
                    "error": "Missing required fields: requested_cores, requested_memory"
                }), 400

            # 요청 메타데이터
            pod_name = data.get('pod_name', 'unknown')
            namespace = data.get('namespace', 'default')

            logger.info("=" * 60)
            logger.info(f"[API] /allocate 요청 수신 (#{self._request_count + 1})")
            logger.info(f"  [REQ] pod={namespace}/{pod_name}, "
                       f"cores={requested_cores}%, memory={requested_memory}MB")

            # 노드 상태 조회
            gpu_util, mem_util = self._get_node_utilization()
            running_pods = self._get_running_gpu_pods()

            logger.info(f"  [NODE] gpu_util={gpu_util:.2%}, mem_util={mem_util:.2%}, "
                       f"running_pods={running_pods}")

            # 할당 요청 생성
            alloc_request = AllocationRequest(
                requested_cores=int(requested_cores),
                requested_memory=int(requested_memory),
                node_gpu_util=gpu_util,
                node_mem_util=mem_util,
                running_pods=running_pods
            )

            # PPO Agent에게 할당 결정 요청
            response = self.agent.get_allocation(alloc_request)

            self._request_count += 1

            logger.info(f"  [DECISION] {requested_cores}%/{requested_memory}MB -> "
                       f"{response.allocated_cores}%/{response.allocated_memory}MB "
                       f"(confidence={response.confidence:.2f})")
            logger.info(f"  [REASON] {response.reason}")
            logger.info("=" * 60)

            return jsonify({
                "allocated_cores": response.allocated_cores,
                "allocated_memory": response.allocated_memory,
                "confidence": response.confidence,
                "reason": response.reason,
                "node_status": {
                    "gpu_util": gpu_util,
                    "mem_util": mem_util,
                    "running_pods": running_pods
                }
            })

        except Exception as e:
            logger.error(f"Error processing allocation request: {e}")
            return jsonify({"error": str(e)}), 500

    def feedback(self):
        """
        학습 피드백 수신 (나중에 reward 계산용)

        Request body:
        {
            "pod_name": "workload-a",
            "allocated_cores": 60,
            "allocated_memory": 3000,
            "actual_performance": 0.85,  # 실제 성능 (0-1)
            "completed": true
        }
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Empty request body"}), 400

            # TODO: 피드백을 reward로 변환하여 학습에 사용
            logger.info(f"Received feedback: {data}")

            return jsonify({"status": "recorded"})

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return jsonify({"error": str(e)}), 500

    def _get_node_utilization(self) -> tuple:
        """노드 GPU/메모리 사용률 조회"""
        try:
            # nvidia-smi로 GPU 사용률 조회
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # 첫 번째 GPU 기준
                    parts = lines[0].split(',')
                    gpu_util = float(parts[0].strip()) / 100.0
                    mem_util = float(parts[1].strip()) / 100.0
                    return gpu_util, mem_util
        except Exception as e:
            logger.warning(f"Failed to get GPU utilization: {e}")

        # 기본값 반환
        return 0.0, 0.0

    def _get_running_gpu_pods(self) -> int:
        """현재 노드에서 실행 중인 GPU Pod 수"""
        # TODO: kubelet API 또는 다른 방법으로 조회
        # 지금은 간단히 nvidia-smi pmon 사용
        try:
            result = subprocess.run(
                ['nvidia-smi', 'pmon', '-c', '1'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # 프로세스 수 계산 (헤더 제외)
                lines = [l for l in result.stdout.strip().split('\n')
                        if l and not l.startswith('#')]
                return len(lines)
        except Exception as e:
            logger.warning(f"Failed to get running pods: {e}")

        return 0

    def _get_gpu_status(self) -> dict:
        """GPU 상태 조회"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    "name": parts[0].strip() if len(parts) > 0 else "unknown",
                    "memory_total": parts[1].strip() if len(parts) > 1 else "unknown",
                    "memory_used": parts[2].strip() if len(parts) > 2 else "unknown",
                    "utilization": parts[3].strip() if len(parts) > 3 else "unknown"
                }
        except Exception as e:
            logger.warning(f"Failed to get GPU status: {e}")

        return {"error": "nvidia-smi not available"}

    def run(self, host: str = None, port: int = None):
        """Run the API server"""
        host = host or API_HOST
        port = port or API_PORT
        logger.info(f"Starting PPO Agent API on {host}:{port}")
        self.app.run(host=host, port=port, threaded=True)


__all__ = ["PPOAgentAPI"]
