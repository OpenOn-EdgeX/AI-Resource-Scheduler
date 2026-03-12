# AI-Resource-Scheduler
# KETI AI Resource Scheduler

엣지 AI 서버 환경에서 이기종 가속 자원(GPU, NPU, PIM, CSD)의 최적 할당 및 Fine-grained 스케줄링을 수행하는 강화학습 기반 AI 리소스 스케줄러

## Overview

KETI AI Resource Scheduler는 [KETRIS(Kernel-based Elastic Temporal-spatio for Resource Integration System)](https://github.com/keti) 프레임워크의 핵심 구성요소로, Kubernetes 엣지 AI 컴퓨팅 환경에서 DaemonSet으로 동작하며 PPO(Proximal Policy Optimization) 강화학습 에이전트를 통해 워크로드 특성에 따른 최적의 SM Cores 및 VRAM 할당을 결정합니다.

```
┌─────────────────────────────────────────────────────┐
│  Kueue Scheduling Queue                             │
│  (Gang / Topology Aware / Priority Scheduling)      │
├─────────────────────────────────────────────────────┤
│  API Server ──► Mutating Webhook ──► Scheduler      │
│                 (keti-gpu-webhook)                   │
├──────────────────┬──────────────────────────────────┤
│  KETI AI Device  │  Resource Scheduler (DaemonSet)  │
│  Plugin          │  RL Agent (Actor-Critic, PPO)    │
│  (DaemonSet)     │                                  │
├──────────────────┴──────────────────────────────────┤
│  SM Partition A  │  SM Partition B  │  SM Partition C│
│  Kernel Hook     │  Kernel Hook     │  Kernel Hook  │
├─────────────────────────────────────────────────────┤
│  AI Accelerator Driver (NVIDIA, NPU)                │
└─────────────────────────────────────────────────────┘
```

## Features

- **PPO 기반 강화학습 스케줄링** — Actor-Critic 신경망을 통해 워크로드 상태를 입력받아 SM Cores 비율과 VRAM 크기를 최적 결정
- **Offline Training / Online Optimization** — 오프라인 학습된 PPO 모델 계수를 기반으로 온라인 환경에서 지속적 피드백 업데이트
- **Kubernetes Native 연동** — DaemonSet 배포, Mutating Admission Webhook 기반 Pod 자동 구성, gRPC 통신
- **Kueue 스케줄링 통합** — Gang Scheduling, Topology Aware Scheduling, Priority Scheduling 정책 지원
- **KETRIS 시공간 다중화 연계** — 할당 결과를 KETRIS의 Temporal-Spatio Sharing Module로 전달하여 SM 파티션 및 시간 슬라이스 배정
- **동적 자원 재할당** — `POST /redefine-allocate` API를 통해 실행 중인 워크로드의 파티션 및 어노테이션 재설정
- **자원 단편화 예측** — Fragmentation Predictor를 통해 시공간 자원 단편화를 사전 감지하고 Best Fit / Re-config 기반 파티셔닝 수행

## Architecture

```
Workload Analyzer                    엣지AI 가속 자원 최적 배치 스케줄러
┌──────────────────┐    ┌─────────────────────────────────────────────┐
│ Workload Profiler│───►│ API Server      Resource        Spatio-    │
│ Resource Cost    │    │ Gateway         Management      temporal   │
│ Profiler         │    │      │               │          multiplex  │
└──────────────────┘    │      ▼               ▼               │     │
                        │ Resource       Resource          Policy    │
자원 파티셔닝             │ Analyzer ◄──► Provisioning      Engine    │
┌──────────────────┐    │      │                               │     │
│ Best Fit         │    │      ▼                               │     │
│ Re-config        │◄──│ AI Resource Monitor                   │     │
│                  │    │      │                               │     │
└──────────────────┘    │      ▼               ▼               │     │
                        │ Isolation    Fragmentation    Fine-grained │
                        │ Checker      Predictor        Scheduler    │
                        └─────────────────────────────────────────────┘
```

## Prerequisites

- Kubernetes v1.26+
- KubeEdge v1.15+ (엣지 노드 연동 시)
- NVIDIA GPU Driver + CUDA 11/12
- NVIDIA MIG 지원 GPU (A100, A6000 등)
- HAMi (Heterogeneous AI Computing Virtualization Middleware)
- Kueue v0.5+
- Python 3.10+
- PyTorch 2.0+

## Installation

### 1. Helm 기반 배포

```bash
# Helm 차트 추가
helm repo add keti https://keti-charts.github.io/edge-ai
helm repo update

# Resource Scheduler 배포
helm install keti-resource-scheduler keti/resource-scheduler \
  --namespace keti-system \
  --create-namespace \
  --set scheduler.mode=ppo \
  --set scheduler.gpuPlugin.enabled=true \
  --set scheduler.npuPlugin.enabled=true
```

### 2. KETI AI Device Plugin 배포

```bash
kubectl apply -f deploy/device-plugin-daemonset.yaml
```

### 3. Mutating Webhook 등록

```bash
kubectl apply -f deploy/mutating-webhook-config.yaml
```

### 4. PPO 모델 계수 로딩

```bash
# 사전 학습된 모델 계수 ConfigMap 생성
kubectl create configmap ppo-weights \
  --from-file=actor_weights.pth=models/actor_weights.pth \
  --from-file=encoder.pth=models/encoder.pth \
  --from-file=critic_weights.pth=models/critic_weights.pth \
  -n keti-system
```

## Configuration

### 스케줄러 설정

```yaml
# config/scheduler-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: keti-scheduler-config
data:
  config.yaml: |
    scheduler:
      mode: ppo                    # ppo | round-robin | best-fit
      allocation:
        min_cores: 10              # 최소 SM Cores (%)
        max_cores: 100             # 최대 SM Cores (%)
        min_memory: 512            # 최소 VRAM (MB)
        max_memory: 48000          # 최대 VRAM (MB)
      ppo:
        state_dim: 5
        action_dim: 2
        hidden_dim: 64
        update_interval: 100       # 피드백 업데이트 주기
      grpc:
        port: 50051
        max_workers: 10
```

### 워크로드 YAML 예시

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: sm-test-workload
    workload: a
  annotations:
    nvidia.com/gpumem: "4000"
    nvidia.com/gpucores: "80"
spec:
  containers:
  - name: ai-inference
    image: keti/ai-workload:latest
    resources:
      limits:
        keti.re.kr/gpu: 1
```

Mutating Webhook이 자동으로 아래 항목을 주입합니다:

```yaml
env:
  - name: LD_PRELOAD
    value: /libvai_accelerator.so
annotations:
  nvidia.com/gpumem: "1500"          # PPO 에이전트 결정값
  nvidia.com/gpucores: "58"          # PPO 에이전트 결정값
  original/gpumem: "4000"            # 원본 요청값 보존
  original/gpucores: "80"            # 원본 요청값 보존
```

## API Reference

### gRPC APIs

| Method | Description |
|--------|------------|
| `POST /allocate` | 신규 워크로드에 SM Cores 및 VRAM 할당 요청 |
| `POST /redefine-allocate` | 실행 중인 워크로드의 파티션 및 어노테이션 재설정 |
| `GET /status` | 현재 자원 할당 상태 및 파티션 맵 조회 |
| `POST /feedback` | 워크로드 실행 결과 피드백 (PPO 온라인 업데이트) |

### Allocation Request/Response

```python
# Request
AllocationRequest(
    requested_cores=80,       # 요청 SM Cores (%)
    requested_memory=4000,    # 요청 VRAM (MB)
    workload_type="inference", # inference | training | preprocessing
    priority=1,               # 우선순위 (0: 최고)
    model_name="bert-base"
)

# Response
AllocationResponse(
    allocated_cores=58,       # PPO 결정 SM Cores (%)
    allocated_memory=1500,    # PPO 결정 VRAM (MB)
    confidence=0.87,          # 할당 신뢰도
    partition_id="sm-part-a"
)
```

## Project Structure

```
keti-resource-scheduler/
├── deploy/
│   ├── device-plugin-daemonset.yaml
│   ├── scheduler-daemonset.yaml
│   └── mutating-webhook-config.yaml
├── config/
│   └── scheduler-config.yaml
├── models/
│   ├── actor_weights.pth
│   ├── encoder.pth
│   └── critic_weights.pth
├── scheduler/
│   ├── __init__.py
│   ├── ppo_core/
│   │   ├── __init__.py          # PPO 할당 로직
│   │   ├── actor.py             # Actor 신경망 (state → SM/VRAM 결정)
│   │   ├── critic.py            # Critic 신경망 (가치 함수)
│   │   └── trainer.py           # Offline Training 루프
│   ├── resource_analyzer/
│   │   ├── fragmentation.py     # 자원 단편화 예측
│   │   ├── isolation_checker.py # 워크로드 간 격리 검증
│   │   └── monitor.py           # AI Resource Monitor 연동
│   ├── grpc_server.py           # gRPC 서버
│   ├── webhook_handler.py       # Mutating Admission Webhook
│   └── rate_limiter.py          # 토큰 기반 커널 실행 제어
├── tests/
├── Dockerfile
├── Makefile
└── README.md
```

## Usage

### 오프라인 학습

```bash
# PPO 에이전트 오프라인 학습
python -m scheduler.ppo_core.trainer \
  --episodes 10000 \
  --env-config config/scheduler-config.yaml \
  --output models/
```

### 스케줄러 단독 실행 (개발/테스트)

```bash
# gRPC 서버 기동
python -m scheduler.grpc_server \
  --config config/scheduler-config.yaml \
  --weights models/ \
  --port 50051
```

### 할당 테스트

```bash
# 테스트 워크로드 배포
kubectl apply -f examples/inference-workload.yaml

# 할당 결과 확인
kubectl get pod -o jsonpath='{.metadata.annotations}' sm-test-workload
```

## Integration with KETRIS

Resource Scheduler는 KETRIS 프레임워크의 다른 모듈과 다음과 같이 연동됩니다:

```
                    ┌──────────────────────┐
                    │  Container Sight     │
                    │  (Static/Dynamic     │
                    │   Profiling)         │
                    └──────────┬───────────┘
                               │ 워크로드 부하 데이터
                               ▼
┌────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│ Kueue Queue    │──►│  Resource Scheduler  │──►│ KETI AI Device     │
│ (Gang/Topo/    │   │  (PPO Agent)         │   │ Plugin             │
│  Priority)     │   │                      │   │ (POST /allocate)   │
└────────────────┘   └──────────┬───────────┘   └────────────────────┘
                               │ 할당 결정
                               ▼
                    ┌──────────────────────┐
                    │ Temporal-Spatio      │
                    │ Sharing Module       │
                    │ (SM 파티션/시간슬라이스)│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ AI Accelerator       │
                    │ Kernel Gating Module │
                    │ (커널 실행 허용/대기)  │
                    └──────────────────────┘
```

## Supported Workloads

| Workload Type | Example Models | Scheduling Policy |
|--------------|----------------|-------------------|
| Training | GPT-7B, GPT2-Large | 대규모 SM 독점 할당, 시간 중심 전략 |
| Inference | BERT-Base, GPT2-Small, GPT2-Medium, GPT-1B, LLaMA-7B | SM 분할 공유, 공간 중심 전략 |
| Preprocessing | CSD 오프로딩 대상 전처리 | Computational Storage Engine 연계 |

> KETRIS 시공간 다중화로 최대 **20개 추론 모델** 동시 배치 가능 (MIG-Only 대비)

## Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04.3 LTS / 24.04.2 LTS |
| GPU | NVIDIA RTX A6000 (48GB GDDR6) × 2 |
| NPU | FuriosaAI Warboy |
| CUDA | 11.x / 12.x |
| Kubernetes | v1.26+ |
| KubeEdge | v1.15+ |
| Kueue | v0.5+ |

## Acknowledgment

이 소프트웨어는 2025년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구 결과입니다.
(No. RS-2025-25441574, 엣지 AI 학습 및 지능의 동시 제공이 가능한 시스템 SW 기술 개발)

## License

Copyright © 2025 Korea Electronics Technology Institute (KETI). All rights reserved.
