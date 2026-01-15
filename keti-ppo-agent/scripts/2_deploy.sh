#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "\033[0;32m==========================================\033[0m"
echo -e "\033[0;32m  KETI PPO Agent - Deploy                 \033[0m"
echo -e "\033[0;32m==========================================\033[0m"

# Create namespace if not exists
kubectl create namespace edge-system --dry-run=client -o yaml | kubectl apply -f -

# Deploy
echo "Deploying PPO Agent DaemonSet..."
kubectl apply -f "${PROJECT_DIR}/deploy/daemonset.yaml"

# Wait for rollout
echo ""
echo "Waiting for rollout..."
kubectl rollout status daemonset/keti-ppo-agent -n edge-system --timeout=120s || true

echo ""
echo -e "\033[0;32m==========================================\033[0m"
echo -e "\033[0;32m  Deploy Complete!                        \033[0m"
echo -e "\033[0;32m==========================================\033[0m"
echo ""
echo "Check status:"
echo "  kubectl get pods -n edge-system -l app=keti-ppo-agent"
echo "  kubectl logs -n edge-system -l app=keti-ppo-agent"
