#!/usr/bin/env bash
set -euo pipefail

# Bootstrap runtime for the unified repo:
# - core/streamdiffusionv2 (vendored upstream core)
# - apps/backend (realtime service)

ENV_NAME="${ENV_NAME:-cam2inf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

# Model toggles
DOWNLOAD_WAN_13B="${DOWNLOAD_WAN_13B:-1}"
DOWNLOAD_WAN_14B="${DOWNLOAD_WAN_14B:-0}"
DOWNLOAD_DMD_CKPT="${DOWNLOAD_DMD_CKPT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORE_DIR="$REPO_DIR/core/streamdiffusionv2"
BACKEND_DIR="$REPO_DIR/apps/backend"

INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

log() {
  echo "[$(date +'%H:%M:%S')] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

log "[1/10] Preflight checks"
require_cmd git
require_cmd wget
require_cmd curl

if [[ ! -d "$CORE_DIR" || ! -f "$CORE_DIR/CORE_MANIFEST.md" ]]; then
  echo "Core directory not found at: $CORE_DIR" >&2
  echo "Run from inside the unified repository." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This host is not GPU-ready." >&2
  exit 1
fi

nvidia-smi || true

log "[2/10] Install Miniconda if missing"
if [[ ! -d "$MINICONDA_DIR" ]]; then
  cd /tmp
  wget -q "$INSTALLER_URL" -O "$INSTALLER"
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
fi

# shellcheck disable=SC1091
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
"$MINICONDA_DIR/bin/conda" config --set auto_activate_base false >/dev/null 2>&1 || true

log "[3/10] Accept conda ToS (idempotent)"
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

log "[4/10] Create/activate env: $ENV_NAME (python $PYTHON_VERSION)"
if ! "$MINICONDA_DIR/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$MINICONDA_DIR/bin/conda" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

log "[5/10] Install base Python tooling"
python -m pip install -U pip setuptools wheel

log "[6/10] Install PyTorch CUDA wheels"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

log "[7/10] Install core runtime dependencies"
pip install "huggingface_hub[cli]"
pip install psutil
if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  pip install --no-build-isolation --no-deps "flash_attn==${FLASH_ATTN_VERSION}"
fi
pip install -r "$CORE_DIR/requirements-runtime.txt"

log "[8/10] Install backend package"
pip install -e "$BACKEND_DIR"

log "[9/10] Download model weights/checkpoints"
mkdir -p "$CORE_DIR/wan_models" "$CORE_DIR/ckpts"

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

if [[ "$DOWNLOAD_WAN_13B" == "1" ]]; then
  huggingface-cli download --resume-download \
    Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir "$CORE_DIR/wan_models/Wan2.1-T2V-1.3B"
fi

if [[ "$DOWNLOAD_WAN_14B" == "1" ]]; then
  huggingface-cli download --resume-download \
    Wan-AI/Wan2.1-T2V-14B \
    --local-dir "$CORE_DIR/wan_models/Wan2.1-T2V-14B"
fi

if [[ "$DOWNLOAD_DMD_CKPT" == "1" ]]; then
  huggingface-cli download --resume-download \
    jerryfeng/StreamDiffusionV2 \
    --local-dir "$CORE_DIR/ckpts" \
    --include "wan_causal_dmd_v2v/*"
fi

log "[10/10] Verify runtime"
export STREAMDIFFUSIONV2_PATH="$CORE_DIR"

python - <<PYEOF
import os
import sys
import torch

core_dir = os.environ["STREAMDIFFUSIONV2_PATH"]
sys.path.insert(0, core_dir)

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline  # noqa: F401
print("core import: ok")
print("STREAMDIFFUSIONV2_PATH=", core_dir)
PYEOF

cat <<MSG

Bootstrap completed.

Next shell commands:
  conda activate $ENV_NAME
  export STREAMDIFFUSIONV2_PATH="$CORE_DIR"
  cd "$BACKEND_DIR"
  python -m app.main

MSG
