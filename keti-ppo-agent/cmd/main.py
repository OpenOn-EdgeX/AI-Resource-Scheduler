#!/usr/bin/env python3
"""
KETI PPO Agent - Entry Point

각 Edge 노드에서 DaemonSet으로 실행됨
Webhook에서 호출하여 GPU 자원 할당 결정
"""

import os
import sys
import signal
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkg.config import LOG_LEVEL, NODE_NAME, get_config_summary
from pkg.agent import PPOAgent
from pkg.api import PPOAgentAPI

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Application:
    """Main application class"""

    def __init__(self):
        self.running = True
        self.agent = None
        self.api = None

        # Signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.agent:
            self.agent.save_model()

    def run(self):
        """Main run loop"""
        logger.info("=" * 50)
        logger.info("KETI PPO Agent Starting...")
        logger.info("=" * 50)

        # Print config
        config = get_config_summary()
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        # Initialize PPO Agent
        logger.info("Initializing PPO Agent...")
        self.agent = PPOAgent()

        # Initialize API Server
        logger.info("Initializing API Server...")
        self.api = PPOAgentAPI(self.agent)

        logger.info("=" * 50)
        logger.info("KETI PPO Agent is running!")
        logger.info("=" * 50)
        logger.info(f"Node: {NODE_NAME}")
        logger.info(f"API Endpoint: http://{config['api_host']}:{config['api_port']}")
        logger.info("")
        logger.info("Endpoints:")
        logger.info("  POST /allocate  - Request resource allocation")
        logger.info("  GET  /health    - Health check")
        logger.info("  GET  /status    - Agent status")
        logger.info("  POST /feedback  - Learning feedback")

        # Run API server (blocking)
        self.api.run()

        # Cleanup
        self._shutdown()

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        if self.agent:
            self.agent.save_model()
        logger.info("KETI PPO Agent stopped.")


def main():
    """Entry point"""
    import traceback
    try:
        app = Application()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
