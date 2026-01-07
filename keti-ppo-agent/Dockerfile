# KETI PPO Agent Dockerfile
# For KubeEdge Edge Node AI-based Resource Allocation

FROM python:3.11-slim

LABEL maintainer="KETI"
LABEL description="KETI PPO Agent for AI-based GPU Resource Allocation"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: torch is large, consider using CPU-only version for edge nodes
RUN pip install --no-cache-dir flask numpy && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || \
    echo "PyTorch installation failed, will use heuristic mode"

# Copy application code
COPY cmd/ ./cmd/
COPY pkg/ ./pkg/

# Create models directory
RUN mkdir -p /app/models

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Entry point
ENTRYPOINT ["python", "-u", "cmd/main.py"]
