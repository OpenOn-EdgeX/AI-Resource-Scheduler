"""
KETI PPO Agent - REST API Server

Webhook에서 호출하는 API 제공:
- POST /allocate: 자원 할당 요청
- POST /partition: MPS 파티션 ID 조회
- GET /health: 헬스 체크
- GET /status: 상태 조회
"""

import csv
import logging
import os
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from typing import Optional, Dict, List

from ..config import (API_HOST, API_PORT, NODE_NAME, get_config_summary, HISTORY_PATH,
                      DEVICE_LSU_CAPACITY, POOL_MANAGER_URL,
                      SB3_SERVING, SB3_MODEL_PATH,
                      POOL_TRANSPORT, POOL_GRPC_TARGET, SLICE_CR_ENABLED)
from ..agent import PPOAgent, AllocationRequest, AllocationResponse, Experience, BATCH_SIZE
from ..pool_client import PoolClient

logger = logging.getLogger(__name__)


class PPOAgentAPI:
    """PPO Agent REST API Server"""

    def __init__(self, agent: PPOAgent):
        self.agent = agent
        self.app = Flask(__name__)
        self._setup_routes()
        self._request_count = 0

        # allocate 후 feedback 전까지 임시 보관 (pod_name -> (partial_exp, lsu_amount, alloc_request))
        self._pending_experiences: Dict[str, tuple] = {}

        # 자원풀 매니저 클라이언트 — gRPC(C-RP-01~03) 기본, 실패 시 REST 폴백
        if POOL_TRANSPORT == 'grpc':
            try:
                from ..pool_client_grpc import PoolClientGrpc
                self._pool = PoolClientGrpc(POOL_GRPC_TARGET)
                logger.info(f"[POOL] gRPC 클라이언트 (C-RP-01~03, {POOL_GRPC_TARGET})")
            except Exception as e:
                logger.warning(f"[POOL] gRPC 초기화 실패 → REST 폴백: {e}")
                self._pool = PoolClient(POOL_MANAGER_URL)
        else:
            self._pool = PoolClient(POOL_MANAGER_URL)

        # C-IS-08: SliceAllocation CR 발행기 (spec 작성 — status 는 KETRIS 몫)
        self._slice_cr = None
        if SLICE_CR_ENABLED:
            try:
                from ..slice_cr import SliceAllocationPublisher
                self._slice_cr = SliceAllocationPublisher()
            except Exception as e:
                logger.warning(f"[SliceCR] 비활성 (클러스터 미접근?): {e}")

        # 오프라인 학습(SB3) 모델 서빙 — 로드 성공 시 온라인 에이전트는 할당 판단에서
        # 배제되고, 온라인 PPO 업데이트도 수행하지 않는다 (SB3_SERVING=0 으로 비활성).
        self.sb3 = None
        if SB3_SERVING:
            try:
                from ..agent.sb3_policy import SB3Policy
                self.sb3 = SB3Policy(SB3_MODEL_PATH)
                logger.info(f"[SB3] 학습 모델 서빙 활성: {SB3_MODEL_PATH} "
                            f"(version={self.sb3.model_version}, "
                            f"confidence={self.sb3.confidence:.4f})")
            except Exception as e:
                logger.warning(f"[SB3] 모델 로드 실패 → 온라인 에이전트 폴백: {e}")

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
        self.app.add_url_rule('/evaluate', 'evaluate', self.evaluate, methods=['POST'])
        self.app.add_url_rule('/history', 'history', self.history, methods=['GET'])
        self.app.add_url_rule('/dashboard', 'dashboard', self.dashboard, methods=['GET'])

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

    def history(self):
        """학습 기록 조회"""
        try:
            if not os.path.exists(HISTORY_PATH):
                return jsonify({"rounds": [], "total": 0})

            rows = []
            with open(HISTORY_PATH, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        'timestamp': row['timestamp'],
                        'round': int(row['round']),
                        'avg_reward': float(row['avg_reward']),
                        'actor_loss': float(row['actor_loss']),
                        'critic_loss': float(row['critic_loss']),
                        'entropy': float(row['entropy']),
                    })

            return jsonify({"rounds": rows, "total": len(rows), "node": NODE_NAME})

        except Exception as e:
            logger.error(f"Error reading history: {e}")
            return jsonify({"error": str(e)}), 500

    def dashboard(self):
        """학습 현황 웹 대시보드"""
        html = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KETI PPO Agent Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
    h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; color: #fff; }
    .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 24px; }
    .stats { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
    .stat-card {
      background: #1c1f2e; border: 1px solid #2a2d3e; border-radius: 10px;
      padding: 16px 24px; min-width: 160px;
    }
    .stat-label { font-size: 0.75rem; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #7c9eff; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .chart-card {
      background: #1c1f2e; border: 1px solid #2a2d3e; border-radius: 10px; padding: 20px;
    }
    .chart-title { font-size: 0.85rem; font-weight: 600; color: #aaa; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em; }
    canvas { max-height: 220px; }
    .toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .btn {
      background: #2a2d3e; border: 1px solid #3a3d50; color: #ccc;
      padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.82rem;
    }
    .btn:hover { background: #353850; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4caf50; display: inline-block; margin-right: 6px; }
    .no-data { text-align: center; padding: 60px; color: #555; font-size: 0.9rem; }
    @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>KETI PPO Agent &mdash; Training Dashboard</h1>
  <p class="subtitle" id="subtitle">로딩 중...</p>

  <div class="toolbar">
    <button class="btn" onclick="loadData()">새로고침</button>
    <span id="auto-label" style="font-size:0.8rem;color:#666;"></span>
  </div>

  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Total Rounds</div>
      <div class="stat-value" id="stat-rounds">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Latest Avg Reward</div>
      <div class="stat-value" id="stat-reward">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Latest Actor Loss</div>
      <div class="stat-value" id="stat-actor">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Latest Critic Loss</div>
      <div class="stat-value" id="stat-critic">-</div>
    </div>
  </div>

  <div id="no-data" class="no-data" style="display:none;">
    아직 학습 데이터가 없습니다.<br>
    <span style="font-size:0.8rem;color:#444;">/feedback 호출 후 32개 transition이 쌓이면 학습이 시작됩니다.</span>
  </div>

  <div class="charts" id="charts">
    <div class="chart-card">
      <div class="chart-title"><span class="status-dot"></span>Average Reward</div>
      <canvas id="chart-reward"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span class="status-dot" style="background:#ff7043"></span>Actor Loss</div>
      <canvas id="chart-actor"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span class="status-dot" style="background:#ab47bc"></span>Critic Loss</div>
      <canvas id="chart-critic"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span class="status-dot" style="background:#ffa726"></span>Entropy</div>
      <canvas id="chart-entropy"></canvas>
    </div>
  </div>

  <script>
    const COLORS = {
      reward: '#4caf50',
      actor:  '#ff7043',
      critic: '#ab47bc',
      entropy:'#ffa726',
    };
    const chartInstances = {};

    function makeChart(id, label, color, rounds, values) {
      const ctx = document.getElementById(id).getContext('2d');
      if (chartInstances[id]) chartInstances[id].destroy();
      chartInstances[id] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: rounds,
          datasets: [{
            label: label,
            data: values,
            borderColor: color,
            backgroundColor: color + '22',
            borderWidth: 2,
            pointRadius: rounds.length > 50 ? 0 : 3,
            tension: 0.3,
            fill: true,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              ticks: { color: '#666', maxTicksLimit: 8 },
              grid: { color: '#1e2130' },
              title: { display: true, text: 'Round', color: '#555', font: { size: 11 } }
            },
            y: {
              ticks: { color: '#666' },
              grid: { color: '#1e2130' },
            }
          }
        }
      });
    }

    async function loadData() {
      try {
        const res = await fetch('/history');
        const data = await res.json();
        const rows = data.rounds || [];

        if (rows.length === 0) {
          document.getElementById('no-data').style.display = 'block';
          document.getElementById('charts').style.display = 'none';
          document.getElementById('subtitle').textContent = '학습 데이터 없음';
          return;
        }

        document.getElementById('no-data').style.display = 'none';
        document.getElementById('charts').style.display = 'grid';

        const rounds  = rows.map(r => r.round);
        const rewards = rows.map(r => r.avg_reward);
        const actors  = rows.map(r => r.actor_loss);
        const critics = rows.map(r => r.critic_loss);
        const entropy = rows.map(r => r.entropy);
        const latest  = rows[rows.length - 1];

        document.getElementById('stat-rounds').textContent = rows.length;
        document.getElementById('stat-reward').textContent = latest.avg_reward.toFixed(4);
        document.getElementById('stat-actor').textContent  = latest.actor_loss.toFixed(4);
        document.getElementById('stat-critic').textContent = latest.critic_loss.toFixed(4);

        const last = rows[rows.length - 1];
        document.getElementById('subtitle').textContent =
          'Node: ' + (data.node || 'unknown') + '  |  마지막 업데이트: ' + last.timestamp;

        makeChart('chart-reward', 'Avg Reward', COLORS.reward,  rounds, rewards);
        makeChart('chart-actor',  'Actor Loss',  COLORS.actor,  rounds, actors);
        makeChart('chart-critic', 'Critic Loss', COLORS.critic, rounds, critics);
        makeChart('chart-entropy','Entropy',     COLORS.entropy,rounds, entropy);

      } catch (e) {
        document.getElementById('subtitle').textContent = '데이터 로드 실패: ' + e.message;
      }
    }

    // 30초마다 자동 갱신
    loadData();
    setInterval(loadData, 30000);
    let countdown = 30;
    setInterval(() => {
      countdown -= 1;
      if (countdown <= 0) countdown = 30;
      document.getElementById('auto-label').textContent = countdown + '초 후 자동 갱신';
    }, 1000);
  </script>
</body>
</html>"""
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

    def _parse_alloc_request(self, data: dict) -> AllocationRequest:
        """
        JSON body → AllocationRequest (lsu_amount 재설계).

        가용 자원 상태(free_lsu/overcommit/tenants)는 요청에 없으면 노드에서 조회.
        실연동 시: 프로파일링/디바이스 필드는 커널 특성 분석·SM 상태 모듈에서 채워진다.
        """
        requested_lsu = max(float(data.get('requested_lsu', 40)), 1.0)  # 1 이상 필수

        # ── ④⑤ 실값 조회: 자원풀 매니저 우선, 실패 시 mock 폴백 ──────────────
        # 우선순위: 요청 본문 명시값 > 자원풀 매니저 > mock 기본값
        pool = self._pool.get_status()          # None 이면 매니저 다운 → 폴백
        hist = self._pool.get_history(data.get('model_id', ''))  # None 이면 이력 없음

        if pool is not None:
            # free_lsu 는 논리값(할당 약속 기준) — GPU util 실측과 다를 수 있음
            # (실측 참고치는 pool['free_lsu_observed'])
            free_lsu_default = pool['free_lsu']
            overcommit_default = pool['overcommit_ratio']
            tenants_default = pool['active_tenants']
            mode_default = pool['mode']
        else:
            # mock 폴백: nvidia-smi 기반 근사 (자원풀 매니저 도입 전 방식)
            gpu_util, _ = self._get_node_utilization()
            free_lsu_default = DEVICE_LSU_CAPACITY * max(0.0, 1.0 - gpu_util)
            overcommit_default = 1.0
            tenants_default = self._get_running_gpu_pods()
            mode_default = 'FREE'

        h = hist or {}
        free_lsu = float(data.get('free_lsu', free_lsu_default))
        return AllocationRequest(
            # ① 워크로드 (큐)
            requested_lsu=requested_lsu,
            workload_kind=data.get('workload_kind', 'INFERENCE'),
            priority=int(data.get('priority', 1)),
            # ② 프로파일링 (실연동: 연세대 I/F — 현재 요청 본문 mock)
            p95_latency_ms=float(data.get('p95_latency_ms', 0.0)),
            batch_size=int(data.get('batch_size', 1)),
            throughput=float(data.get('throughput', 0.0)),
            # 연세대 I/F 협의 필요 (신규 필드 요청 — slo_target_throughput)
            slo_target_throughput=float(data.get('slo_target_throughput', 0.0)),
            peak_memory_mb=float(data.get('peak_memory_mb', 0.0)),
            # ③ 디바이스·파티션 (실연동: 성균관대 매칭 결과)
            device_kind=data.get('device_kind', 'GPU'),
            partition_sm_fraction=float(data.get('partition_sm_fraction', 1.0)),
            device_lsu_capacity=float(data.get('device_lsu_capacity', DEVICE_LSU_CAPACITY)),
            # ④ 가용 자원 (자원풀 매니저 실값; 매니저 다운 시 mock 폴백)
            free_lsu=free_lsu,
            overcommit_ratio=float(data.get('overcommit_ratio', overcommit_default)),
            active_tenants=int(data.get('active_tenants', tenants_default)),
            # ⑤ 모델별 실행 이력 (자원풀 매니저 이력 저장소; 이력 없으면 lsu_est 폴백)
            kernels_per_iter=int(data.get('kernels_per_iter',
                                          h.get('kernels_per_iter') or 0)),
            mps_pct=float(data.get('mps_pct', h.get('mps_pct') or 0.0)),
            mode=data.get('mode', h.get('mode') or mode_default),
            lsu_est=float(data.get('lsu_est', requested_lsu)),  # 이력 없는 신규 모델 폴백
            resource_bound_type=data.get('resource_bound_type',
                                         h.get('resource_bound_type')
                                         or data.get('workload_type', 'MIXED')),  # 구 필드명 하위호환
        )

    def evaluate(self):
        """
        학습 결과 확인용 결정론적 추론 엔드포인트

        /allocate와 달리 탐색 없이 Actor mean값만 사용 → 같은 입력이면 항상 같은 출력
        학습 전후 동일 입력으로 호출하여 정책 변화 확인

        Request body: /allocate와 동일 (requested_lsu 등)
        """
        try:
            data = request.get_json() or {}
            alloc_request = self._parse_alloc_request(data)

            # 결정론적 추론 — SB3 학습 모델 우선, 미로드 시 온라인 에이전트 mean
            det_action = None
            if self.sb3 is not None:
                response, state_vec, det_action = self.sb3.decide(alloc_request)
                logger.info(f"  [MODEL] applied=True version={self.sb3.model_version} "
                           f"action={det_action:.6f}")
            else:
                response = self.agent.get_allocation(alloc_request)

            stats = self.agent.training_stats

            logger.info(f"[EVALUATE] lsu_amount={response.lsu_amount} LSU, "
                       f"confidence={response.confidence:.4f} | "
                       f"총 학습={stats['total_updates']}회, "
                       f"avg_reward={stats['avg_reward']:.4f}")

            return jsonify({
                "input": {
                    "requested_lsu": alloc_request.requested_lsu,
                    "workload_kind": alloc_request.workload_kind,
                    "resource_bound_type": alloc_request.resource_bound_type,
                    "priority": alloc_request.priority,
                    "free_lsu": alloc_request.free_lsu,
                    "active_tenants": alloc_request.active_tenants,
                },
                "decision": {
                    "lsu_amount": response.lsu_amount,
                    "confidence": response.confidence,
                    "reason": response.reason
                },
                "model": ({
                    "applied": True,
                    "model_path": self.sb3.model_path,
                    "model_version": self.sb3.model_version,
                    "deterministic_action": round(det_action, 6),
                } if self.sb3 is not None else {"applied": False}),
                "training": {
                    "total_updates": stats['total_updates'],
                    "total_experiences": stats['total_experiences'],
                    "avg_reward": stats['avg_reward'],
                    "avg_actor_loss": stats['avg_actor_loss'],
                    "avg_critic_loss": stats['avg_critic_loss'],
                    "pending_experiences": len(self.agent.experiences),
                    "batch_size": BATCH_SIZE
                }
            })

        except Exception as e:
            logger.error(f"Error processing evaluate request: {e}")
            return jsonify({"error": str(e)}), 500

    def allocate(self):
        """
        자원 할당 요청 처리 (lsu_amount 산출)

        Request body:
        {
            "requested_lsu": 40,          # 요청 논리 자원량(LSU) [필수]
            "workload_kind": "INFERENCE",      # ① 선택 (INFERENCE/TRAINING)
            "priority": 1,                     # ① 선택 0/1/2
            "p95_latency_ms": 12.5,            # ② 선택 (실연동: 연세대 I/F)
            "batch_size": 8,                   # ② 선택 1~16
            "throughput": 850,                 # ② 선택 (img/s | tokens/s)
            "slo_target_throughput": 1000,     # ② 선택 — 연세대 I/F 협의 필요 (신규 필드 요청)
            "peak_memory_mb": 8192,            # ② 선택
            "device_kind": "GPU",              # ③ 선택 (실연동: 성균관대 매칭)
            "partition_sm_fraction": 0.5,      # ③ 선택
            "kernels_per_iter": 460,           # ⑤ 선택 (실연동: KETRIS 이력)
            "mode": "FREE",                    # ⑤ 선택
            "lsu_est": 38,                     # ⑤ 선택 (이력 없을 때 폴백)
            "resource_bound_type": "COMPUTE_BOUND",  # ⑤ 선택
            "pod_name": "workload-a",          # 선택
            "namespace": "default"             # 선택
        }

        Response:
        {
            "lsu_amount": 32.5,           # KETRIS 로 넘길 논리 자원 할당량(LSU)
            "confidence": 0.85,
            "reason": "PPO decision ..."
        }
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Empty request body"}), 400

            if data.get('requested_lsu') is None:
                return jsonify({"error": "Missing required field: requested_lsu"}), 400

            pod_name = data.get('pod_name', 'unknown')
            namespace = data.get('namespace', 'default')

            # 할당 요청 생성 (④ 가용자원은 노드에서 조회하여 채움)
            alloc_request = self._parse_alloc_request(data)

            logger.info("=" * 60)
            logger.info(f"[API] /allocate 요청 수신 (#{self._request_count + 1})")
            logger.info(f"  [REQ] pod={namespace}/{pod_name}, requested_lsu={alloc_request.requested_lsu:.1f}, "
                       f"kind={alloc_request.workload_kind}, bound={alloc_request.resource_bound_type}")
            logger.info(f"  [NODE] free_lsu={alloc_request.free_lsu:.1f}, "
                       f"active_tenants={alloc_request.active_tenants}")

            if self.sb3 is not None:
                # 오프라인 학습 모델 결정론 추론 — 온라인 에이전트/탐색/학습 미사용
                response, state_vec, det_action = self.sb3.decide(alloc_request)
                partial_exp = None
                logger.info(f"  [MODEL] applied=True (offline SB3)")
                logger.info(f"  [MODEL] model_path={self.sb3.model_path}")
                logger.info(f"  [MODEL] model_version={self.sb3.model_version}")
                logger.info(f"  [STATE] {np.array2string(state_vec, precision=4, separator=', ', max_line_width=200)}")
                if self.sb3.param == 'capacity_ratio':
                    logger.info(f"  [ACTION] allocation_ratio={det_action:.6f} "
                               f"-> lsu_amount={response.lsu_amount} LSU "
                               f"(= min(clip({det_action:.4f} x {alloc_request.device_lsu_capacity:.0f}, "
                               f"1, {alloc_request.device_lsu_capacity:.0f}), req {alloc_request.requested_lsu:.1f}))")
                else:
                    logger.info(f"  [ACTION] deterministic={det_action:.6f} "
                               f"-> lsu_amount={response.lsu_amount} LSU "
                               f"(= clip({det_action:.4f} x {alloc_request.requested_lsu:.1f}, "
                               f"1, {alloc_request.device_lsu_capacity:.0f}))")
            else:
                # 폴백: 온라인 에이전트 (SB3 미로드 시에만)
                logger.info("  [MODEL] applied=False (online agent fallback)")
                response, partial_exp = self.agent.select_action_for_training(alloc_request)

            self._request_count += 1

            # feedback 연계 정보 보관 (SB3 모드에서는 partial_exp=None — 학습 미수행,
            # 해제/이력 축적 경로만 사용)
            self._pending_experiences[pod_name] = (partial_exp, response.lsu_amount, alloc_request)

            # 자원풀 매니저에 논리 할당 자동 등록 (best-effort — 실패해도 응답은 정상 진행)
            # 이게 있어야 virtual_sm_total 이 쌓여 mode/overcommit_ratio 가 유의미해짐
            reg = self._pool.register_allocation(
                model_id=data.get('model_id', pod_name),
                pod_name=pod_name,
                lsu_amount=response.lsu_amount)
            if reg is not None:
                logger.info(f"  [POOL] 등록 완료(C-RP-03 RESERVED): vsm={reg.get('virtual_sm')} "
                           f"mode={reg.get('pool', {}).get('mode')}")

            # C-IS-08: SliceAllocation CR 발행 — KETRIS 가 Watch 하여 실제 분할 적용
            if self._slice_cr is not None:
                self._slice_cr.create(pod_name, alloc_request, response.lsu_amount)

            exp_count = len(self.agent.experiences)
            logger.info(f"  [DECISION] requested_lsu={alloc_request.requested_lsu:.1f} -> "
                       f"lsu_amount={response.lsu_amount} LSU (confidence={response.confidence:.2f})")
            logger.info(f"  [REASON] {response.reason}")
            logger.info(f"  [BUFFER] transition {exp_count}/{BATCH_SIZE} | 대기 중 {len(self._pending_experiences)}개")
            logger.info("=" * 60)

            model_info = {"applied": self.sb3 is not None}
            if self.sb3 is not None:
                model_info.update({
                    "model_path": self.sb3.model_path,
                    "model_version": self.sb3.model_version,
                    "deterministic_action": round(det_action, 6),
                })

            return jsonify({
                "lsu_amount": response.lsu_amount,
                "confidence": response.confidence,
                "reason": response.reason,
                "model": model_info,
                "node_status": {
                    "free_lsu": alloc_request.free_lsu,
                    "active_tenants": alloc_request.active_tenants,
                    "device_lsu_capacity": alloc_request.device_lsu_capacity,
                }
            })

        except Exception as e:
            logger.error(f"Error processing allocation request: {e}")
            return jsonify({"error": str(e)}), 500

    def feedback(self):
        """
        학습 피드백 수신 → reward 계산 → transition 기록 → 학습 트리거

        Request body:
        {
            "pod_name": "workload-a",
            "lsu_amount": 32.5,          # allocate 가 반환한 값 (없으면 저장분 사용)
            "actual_gpu_util": 0.82,     # 실측 GPU Util (0-1)  # 실연동: KETRIS 모니터링
            "slo_met": true,             # SLO 충족 여부         # 실연동: KETRIS 모니터링 exporter
            "throughput_ratio": 0.9,     # 실측/목표 throughput (0-1)
            "completed": true
        }
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Empty request body"}), 400

            pod_name = data.get('pod_name', 'unknown')
            completed = bool(data.get('completed', False))
            # 실측 성능 소스 (현재 요청 본문 mock; 실연동: KETRIS 모니터링 + exporter)
            actual_gpu_util = float(data.get('actual_gpu_util', data.get('actual_performance', 0.5)))
            slo_met = bool(data.get('slo_met', completed and actual_gpu_util >= 0.7))
            throughput_ratio = float(data.get('throughput_ratio', actual_gpu_util))

            logger.info("=" * 60)
            logger.info(f"[API] /feedback 수신 (pod={pod_name})")
            logger.info(f"  [FEEDBACK] gpu_util={actual_gpu_util:.2f}, slo_met={slo_met}, "
                       f"throughput={throughput_ratio:.2f}, completed={completed}")

            # 대기 중인 transition 찾기
            pending = self._pending_experiences.pop(pod_name, None)
            if pending is None:
                logger.warning(f"  [WARN] pod '{pod_name}' 의 대기 transition 없음 (allocate 먼저 호출 필요)")
                logger.info("=" * 60)
                return jsonify({"status": "recorded", "warning": "no pending experience found"})

            partial_exp, stored_lsu, alloc_request = pending
            lsu_amount = float(data.get('lsu_amount', stored_lsu))

            # reward 계산 (3기준: 활용률/낭비/SLO)
            reward = PPOAgent.compute_reward(
                lsu_amount=lsu_amount,
                device_lsu_capacity=alloc_request.device_lsu_capacity,
                actual_gpu_util=actual_gpu_util,
                slo_met=slo_met,
                throughput_ratio=throughput_ratio,
                active_tenants=max(alloc_request.active_tenants, 1),
                requested_lsu=alloc_request.requested_lsu,
            )
            used_lsu = actual_gpu_util * alloc_request.device_lsu_capacity
            logger.info(f"  [REWARD] {reward:.4f} "
                       f"(util={actual_gpu_util:.2f}, slo={slo_met}, "
                       f"waste={max(lsu_amount-used_lsu,0):.1f} LSU)")

            # 워크로드 종료 → 자원풀 매니저 논리 할당 자동 해제 (best-effort)
            rel = self._pool.release_by_pod(pod_name)
            if rel is not None:
                logger.info(f"  [POOL] 해제(C-RP-03 RELEASED): released={rel.get('released')} "
                           f"mode={rel.get('pool', {}).get('mode')}")

            # C-IS-08: CR 삭제 — KETRIS 가 DELETED 이벤트로 자원 회수
            if self._slice_cr is not None:
                self._slice_cr.delete(pod_name)
            # 실행 결과를 모델 이력에 기록 (⑤그룹 축적; 3단계에서 KETRIS 실측으로 대체/보강)
            model_id = data.get('model_id', pod_name)
            self._pool.upsert_history(model_id, {
                "gpu_util": actual_gpu_util,
                "throughput": throughput_ratio,
                "mode": alloc_request.mode,
                "mps_pct": alloc_request.mps_pct,
                "resource_bound_type": alloc_request.resource_bound_type,
                "kernels_per_iter": alloc_request.kernels_per_iter,
            })

            # transition 기록·학습 — SB3 서빙 모드(partial_exp=None)에서는 수행하지
            # 않는다 (오프라인 모델 고정 서빙, 온라인 PPO 업데이트 없음)
            train_result = None
            if partial_exp is not None:
                exp = Experience(
                    state=partial_exp.state,
                    action=partial_exp.action,
                    reward=reward,
                    next_state=partial_exp.state,  # terminal
                    done=completed,
                    log_prob=partial_exp.log_prob,
                    value=partial_exp.value
                )
                self.agent.record_experience(exp)

                exp_count = len(self.agent.experiences)
                logger.info(f"  [BUFFER] {exp_count}/{BATCH_SIZE} transition 수집됨 "
                           f"({'학습 시작!' if exp_count >= BATCH_SIZE else f'{BATCH_SIZE - exp_count}개 더 필요'})")

                # BATCH_SIZE 충족 시 학습
                train_result = self.agent.train_step()
                if train_result:
                    logger.info(f"  [TRAIN] 학습 완료 - "
                               f"actor_loss={train_result['actor_loss']:.4f}, "
                               f"critic_loss={train_result['critic_loss']:.4f}, "
                               f"avg_reward={train_result['avg_reward']:.4f}, "
                               f"총 업데이트={train_result['updates']}")
            else:
                logger.info("  [BUFFER] SB3 서빙 모드 — transition 기록/온라인 학습 생략")

            logger.info("=" * 60)

            return jsonify({
                "status": "recorded",
                "reward": reward,
                "experiences": len(self.agent.experiences),
                "batch_size": BATCH_SIZE,
                "trained": train_result is not None,
                "online_training": self.sb3 is None,
            })

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
