#!/usr/bin/env python3
"""
KETI PPO Agent - Entry Point

각 Edge 노드에서 DaemonSet으로 실행됨
Webhook에서 호출하여 GPU 자원 할당 결정
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkg.config import LOG_LEVEL, NODE_NAME
from pkg.agent import PPOAgent
from pkg.api import PPOAgentAPI

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Entry point"""
    logger.info("KETI PPO Agent Starting...")
    logger.info(f"Node: {NODE_NAME}")

    agent = PPOAgent()
    api = PPOAgentAPI(agent)
    api.run()


if __name__ == "__main__":
    main()
