"""
KETI PPO Agent Configuration
"""

import os

# API Server settings
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', '8080'))

# PPO Model settings
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/ppo_model.pt')
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '3e-4'))
GAMMA = float(os.environ.get('GAMMA', '0.99'))  # Discount factor
CLIP_EPSILON = float(os.environ.get('CLIP_EPSILON', '0.2'))  # PPO clip range

# State space dimensions
# State: [requested_cores, requested_mem, node_gpu_util, node_mem_util, running_pods]
STATE_DIM = int(os.environ.get('STATE_DIM', '5'))

# Action space dimensions
# Action: [cores_percent, memory_mb] (continuous)
ACTION_DIM = int(os.environ.get('ACTION_DIM', '2'))

# Resource limits
MIN_CORES_PERCENT = int(os.environ.get('MIN_CORES_PERCENT', '10'))
MAX_CORES_PERCENT = int(os.environ.get('MAX_CORES_PERCENT', '100'))
MIN_MEMORY_MB = int(os.environ.get('MIN_MEMORY_MB', '512'))
MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', '32768'))

# Node info
NODE_NAME = os.environ.get('NODE_NAME', 'unknown')
TOTAL_GPU_MEMORY_MB = int(os.environ.get('TOTAL_GPU_MEMORY_MB', '24576'))  # 24GB default

# Logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


def get_config_summary():
    return {
        "api_host": API_HOST,
        "api_port": API_PORT,
        "model_path": MODEL_PATH,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "node_name": NODE_NAME,
        "cores_range": f"{MIN_CORES_PERCENT}-{MAX_CORES_PERCENT}%",
        "memory_range": f"{MIN_MEMORY_MB}-{MAX_MEMORY_MB}MB",
    }
