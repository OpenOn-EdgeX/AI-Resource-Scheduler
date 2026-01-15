#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="ketidevit2/keti-ppo-agent"
IMAGE_TAG="${1:-latest}"

echo -e "\033[0;32m==========================================\033[0m"
echo -e "\033[0;32m  KETI PPO Agent - Build                  \033[0m"
echo -e "\033[0;32m==========================================\033[0m"

cd "$PROJECT_DIR"

echo -e "\033[1;33mBuilding image: ${IMAGE_NAME}:${IMAGE_TAG}\033[0m"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo -e "\033[1;33mPushing image to Docker Hub...\033[0m"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "\033[0;32m==========================================\033[0m"
echo -e "\033[0;32m  Build & Push Complete!                  \033[0m"
echo -e "\033[0;32m==========================================\033[0m"
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Next step:"
echo "  - Deploy: ./scripts/2_deploy.sh"
