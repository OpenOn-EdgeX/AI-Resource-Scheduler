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
