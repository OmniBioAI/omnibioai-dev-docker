#!/bin/bash
# ==========================================================
# Script Name: run_ai_dev.sh
# Description: Launch OmniBioAI Dev Environment
# Image:       ghcr.io/omnibioai/omnibioai-dev-env
# Requires:    NVIDIA GPU + nvidia-container-toolkit
# Usage:       bash run_ai_dev.sh [OPTIONS]
#   --jupyter    Start JupyterLab automatically
#   --ollama     Start Ollama server automatically
#   --build      Force rebuild image from Dockerfile
#   --help       Show this help message
# ==========================================================

IMAGE_NAME="ghcr.io/omnibioai/omnibioai-dev-env"
TAG="latest"
CONTAINER_NAME="omnibio_dev_foundry"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

# ── Parse arguments ───────────────────────────────────────
START_JUPYTER=false
START_OLLAMA=false
FORCE_BUILD=false

for arg in "$@"; do
  case $arg in
    --jupyter) START_JUPYTER=true ;;
    --ollama)  START_OLLAMA=true ;;
    --build)   FORCE_BUILD=true ;;
    --help)
      grep "^#" "$0" | grep -v "!/bin/bash" | sed 's/^# \?//'
      exit 0
      ;;
  esac
done

echo "=========================================================="
echo "  OmniBioAI Dev Environment"
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "=========================================================="

# ── 1. Check NVIDIA runtime ───────────────────────────────
echo ""
echo "▶ Checking GPU availability..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "❌ ERROR: nvidia-smi not found."
    echo "   Install NVIDIA drivers + nvidia-container-toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader 2>/dev/null | while IFS=',' read -r name mem driver; do
  echo "   ✅ GPU: $name | Memory: $mem | Driver: $driver"
done

# ── 2. Check Docker daemon ────────────────────────────────
if ! docker info &>/dev/null; then
    echo "❌ ERROR: Docker daemon not running."
    echo "   Start Docker: sudo systemctl start docker"
    exit 1
fi

# ── 3. Get or build image ─────────────────────────────────
echo ""
echo "▶ Checking image..."

if [[ "$FORCE_BUILD" == true ]]; then
    echo "   Force rebuild requested..."
    docker build -t ${IMAGE_NAME}:${TAG} .

elif [[ "$(docker images -q ${IMAGE_NAME}:${TAG} 2>/dev/null)" == "" ]]; then
    echo "   Image not found locally. Pulling from GHCR..."
    if ! docker pull ${IMAGE_NAME}:${TAG}; then
        echo "   Pull failed. Building from Dockerfile..."
        if [[ ! -f "Dockerfile" ]]; then
            echo "❌ ERROR: No Dockerfile found in $(pwd)"
            exit 1
        fi
        docker build -t ${IMAGE_NAME}:${TAG} .
    fi
else
    echo "   ✅ Image found: ${IMAGE_NAME}:${TAG}"
fi

# ── 4. Clean up existing container ───────────────────────
echo ""
echo "▶ Cleaning up existing container..."
docker rm -f ${CONTAINER_NAME} &>/dev/null && \
  echo "   Removed existing container." || \
  echo "   No existing container to remove."

# ── 5. Build startup command ──────────────────────────────
STARTUP_CMD="echo '========================================'; \
echo '  OmniBioAI Dev Environment Ready'; \
echo '========================================'; \
echo '  Workspace : /workspace'; \
echo '  JupyterLab: http://localhost:${JUPYTER_PORT}'; \
echo '  Ollama    : http://localhost:${OLLAMA_PORT}'; \
echo '========================================'"

if [[ "$START_OLLAMA" == true ]]; then
    STARTUP_CMD="${STARTUP_CMD}; ollama serve &"
    echo "   Ollama will start automatically on port ${OLLAMA_PORT}"
fi

if [[ "$START_JUPYTER" == true ]]; then
    STARTUP_CMD="${STARTUP_CMD}; jupyter lab \
        --ip=0.0.0.0 \
        --port=${JUPYTER_PORT} \
        --allow-root \
        --no-browser \
        --NotebookApp.token='' \
        --NotebookApp.password=''"
    echo "   JupyterLab will start on http://localhost:${JUPYTER_PORT}"
else
    STARTUP_CMD="${STARTUP_CMD}; bash"
fi

# ── 6. Launch container ───────────────────────────────────
echo ""
echo "▶ Launching container..."
echo "   Workspace : $(pwd) → /workspace"
echo "   JupyterLab: http://localhost:${JUPYTER_PORT}"
echo "   Ollama    : http://localhost:${OLLAMA_PORT}"
echo ""

docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name ${CONTAINER_NAME} \
  --hostname omnibioai-dev \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.ollama:/root/.ollama \
  -v "$(pwd)":/workspace \
  -v ~/.gitconfig:/root/.gitconfig:ro \
  -p ${JUPYTER_PORT}:${JUPYTER_PORT} \
  -p ${OLLAMA_PORT}:${OLLAMA_PORT} \
  -e HF_HOME=/root/.cache/huggingface \
  -e OLLAMA_HOST=0.0.0.0 \
  -e PYTHONUNBUFFERED=1 \
  -e TERM=xterm-256color \
  -it ${IMAGE_NAME}:${TAG} bash -c "${STARTUP_CMD}"

# ── 7. Exit message ───────────────────────────────────────
echo ""
echo "=========================================================="
echo "  Session ended. Work saved in $(pwd)"
echo "  To resume: docker start -ai ${CONTAINER_NAME}"
echo "=========================================================="
