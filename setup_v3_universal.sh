#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka V3 Benchmark Setup — vLLM 0.14.0 + gonka_poc (Universal)
# For: H100, H200, A100 and other NVIDIA GPUs (NOT Blackwell)
# Benchmark-only: no nginx, no mlnode, no tmux, no logrotate
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

VLLM_VERSION="0.14.0"
CUDA_VERSION="13.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#################################
# COLORS
#################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $(date '+%H:%M:%S') $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $(date '+%H:%M:%S') $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $(date '+%H:%M:%S') $1"; }
log_poc()     { echo -e "${CYAN}[PoC]${NC} $(date '+%H:%M:%S') $1"; }

#################################
# CHECK ROOT
#################################
if [[ $EUID -ne 0 ]]; then
    log_error "Run as root: sudo bash $0"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Gonka V3 Benchmark Setup — vLLM ${VLLM_VERSION} + CUDA ${CUDA_VERSION}         ║"
echo "║     UNIVERSAL: H100/H200/A100 (benchmark only)                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

#################################
# 1. BASE PACKAGES
#################################
log_info "Installing base packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    ca-certificates curl git jq tar wget \
    lsof software-properties-common

#################################
# 2. CUDA TOOLKIT
#################################
if [ ! -d "/usr/local/cuda-13.0" ]; then
    log_warning "CUDA 13.0 toolkit not found. Installing..."
    if ! dpkg -l | grep -q cuda-keyring; then
        log_info "Adding NVIDIA repository..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update
    fi
    log_info "Installing CUDA 13.0 toolkit..."
    apt-get install -y cuda-toolkit-13-0
    log_success "CUDA 13.0 toolkit installed"
fi

# Set CUDA_HOME
if [ -d "/usr/local/cuda-13.0" ]; then
    CUDA_HOME="/usr/local/cuda-13.0"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_HOME="/usr/local/cuda-12.6"
    log_warning "Using CUDA 12.6"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
else
    log_error "CUDA not found"
    exit 1
fi
log_info "CUDA_HOME: $CUDA_HOME"

#################################
# 3. PYTHON 3.12 + UV
#################################
log_info "Installing Python 3.12..."
add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
apt-get update -qq
apt-get install -y python3.12 python3.12-venv python3.12-dev

log_info "Installing uv package manager..."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if command -v uv &> /dev/null; then
    log_success "uv: $(uv --version)"
else
    log_warning "uv failed, using pip"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - --break-system-packages
fi

#################################
# 4. CHECK GPU
#################################
log_info "Checking GPU..."
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# Block Blackwell GPUs
if echo "$GPU_NAME" | grep -qiE '(B200|B100|GB200|B300)'; then
    log_error "Blackwell GPU detected ($GPU_NAME)! Use setup_v3_blackwell.sh instead."
    exit 1
fi

DRIVER_VERSION_FULL=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER_VERSION_FULL" | cut -d'.' -f1)
log_info "NVIDIA Driver: $DRIVER_VERSION_FULL"

MIN_DRIVER_MAJOR=535
if [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ]; then
    log_error "Driver version $DRIVER_VERSION_FULL is too old! Minimum: $MIN_DRIVER_MAJOR"
    exit 1
fi
log_success "Driver version OK"

#################################
# 5. ENVIRONMENT
#################################
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

#################################
# 6. REMOVE CONFLICTING PACKAGES
#################################
log_info "Removing conflicting packages..."
apt-get remove -y python3-torch python3-triton 2>/dev/null || true
rm -rf /usr/lib/python3/dist-packages/torch* 2>/dev/null || true
rm -rf /usr/lib/python3/dist-packages/triton* 2>/dev/null || true

#################################
# 7. INSTALL PYTORCH + VLLM
#################################
log_info "Installing vLLM ${VLLM_VERSION} with CUDA ${CUDA_VERSION}..."
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match \
    vllm==${VLLM_VERSION} \
    --extra-index-url https://wheels.vllm.ai/${VLLM_VERSION}/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

log_info "Installing dependencies..."
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match \
    uvicorn fastapi starlette pydantic \
    "httpx[http2]" toml fire nvidia-ml-py \
    accelerate tiktoken transformers \
    openai aiohttp \
    grpcio grpcio-tools protobuf

apt-get remove -y python3-scipy 2>/dev/null || true
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match scipy

#################################
# 8. FLASHINFER + TRITON
#################################
log_info "Installing FlashInfer 0.5.3..."
uv pip uninstall --python python3.12 --system --break-system-packages flashinfer-python 2>/dev/null || true
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match \
    flashinfer-python==0.5.3 \
    --extra-index-url https://flashinfer.ai/whl/cu130/torch2.5/

apt-get remove -y python3-optree 2>/dev/null || true
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match optree
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match triton 2>/dev/null || true

rm -rf ~/.cache/flashinfer 2>/dev/null || true

#################################
# 9. VERIFY INSTALLATION
#################################
log_info "Verifying installation..."
echo "  PyTorch: $(python3.12 -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python3.12 -c 'import torch; print(torch.version.cuda)')"
echo "  vLLM: $(python3.12 -c 'import vllm; print(vllm.__version__)')"
python3.12 -c "import flashinfer; print('  FlashInfer: installed')" 2>/dev/null || echo "  FlashInfer: not installed"
python3.12 -c "import triton; print(f'  Triton: {triton.__version__}')" 2>/dev/null || echo "  Triton: not installed"

#################################
# 10. INSTALL GONKA PoC MODULE
#################################
log_poc "Installing gonka_poc module..."

VLLM_PATH=$(python3.12 -c "import vllm; print(vllm.__path__[0])")
GONKA_POC_DEST="$VLLM_PATH/gonka_poc"

# Find gonka_poc directory
GONKA_POC_SOURCE=""
for CHECK_DIR in "$SCRIPT_DIR/gonka_poc" "./gonka_poc" "$(dirname "$SCRIPT_DIR")/gonka_poc"; do
    if [ -d "$CHECK_DIR" ] && [ -f "$CHECK_DIR/__init__.py" ] && [ -f "$CHECK_DIR/routes.py" ]; then
        GONKA_POC_SOURCE="$CHECK_DIR"
        break
    fi
done

if [ -n "$GONKA_POC_SOURCE" ]; then
    log_poc "Using local gonka_poc from: $GONKA_POC_SOURCE"
    rm -rf "$GONKA_POC_DEST"
    cp -r "$GONKA_POC_SOURCE" "$VLLM_PATH/"
    log_success "gonka_poc installed from local source"
else
    log_error "gonka_poc directory not found! Please ensure it's in the same directory as this script."
    exit 1
fi

# Verify
if python3.12 -c "from vllm.gonka_poc import PoCManagerV1" 2>/dev/null; then
    log_success "gonka_poc module installed successfully"
else
    log_error "gonka_poc module installation failed!"
    exit 1
fi

#################################
# 11. PATCH api_server.py
#################################
log_poc "Patching api_server.py for gonka_poc support..."

API_SERVER_PATH=$(python3.12 -c "import vllm.entrypoints.openai; import os; print(os.path.dirname(vllm.entrypoints.openai.__file__))")
API_SERVER_FILE="$API_SERVER_PATH/api_server.py"

if ! grep -q "from vllm.gonka_poc.routes import router as gonka_poc_router" "$API_SERVER_FILE"; then
    cp "$API_SERVER_FILE" "$API_SERVER_FILE.bak"
    LAST_IMPORT_LINE=$(grep -n "^from " "$API_SERVER_FILE" | tail -1 | cut -d: -f1)
    sed -i "${LAST_IMPORT_LINE}a from vllm.gonka_poc.routes import router as gonka_poc_router" "$API_SERVER_FILE"
    sed -i "/app.include_router(router)/a\    app.include_router(gonka_poc_router)" "$API_SERVER_FILE"
    log_success "api_server.py patched for gonka_poc"
else
    log_info "api_server.py already patched"
fi

#################################
# DONE
#################################
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     V3 SETUP COMPLETE — UNIVERSAL (H100/H200/A100)              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  vLLM: ${VLLM_VERSION} + gonka_poc                                     ║"
echo "║  Ready for: bash run.sh --mode v3                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
log_success "Setup complete!"
