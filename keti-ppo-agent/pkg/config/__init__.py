"""
KETI PPO Agent Configuration
"""

import os

# API Server settings
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', '8080'))

# Node info
NODE_NAME = os.environ.get('NODE_NAME', 'unknown')
TOTAL_GPU_MEMORY_MB = int(os.environ.get('TOTAL_GPU_MEMORY_MB', '24576'))  # 24GB default

# Logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

