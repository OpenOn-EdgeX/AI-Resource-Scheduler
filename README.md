# AI Resource Scheduler

PPO (Proximal Policy Optimization) based GPU resource scheduling agent for Kubernetes.

## Overview

Actor-Critic based reinforcement learning agent that optimizes GPU SM partition allocation for multi-tenant workloads.

## Directory Structure
```
AI-Resource-Scheduler/
└── keti-ppo-agent/
    ├── cmd/          # Entry point
    ├── pkg/          # Core packages (agent, api)
    ├── deploy/       # Kubernetes deployments
    ├── scripts/      # Utility scripts
    ├── Dockerfile
    └── requirements.txt
```

## Quick Start
```bash
cd keti-ppo-agent
docker build -t keti-ppo-agent .
kubectl apply -f deploy/
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/allocate` | POST | Request GPU SM allocation |
| `/partition` | GET | Get current partition info |
| `/health` | GET | Health check |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8080 | Agent API port |
| `CUDA_MPS_PIPE_DIRECTORY` | - | MPS pipe path |
| `GPU_UUID` | - | Target GPU UUID |
