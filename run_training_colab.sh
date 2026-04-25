#!/usr/bin/env bash
set -e

# =========================================================
# MoCha + ROCm (Polaris RX 560) Auto Fix for NixOS
# - Auto detect ROCm libs from /nix/store
# - Auto replace broken PyTorch bundled libs
# - Auto create venv if missing
# - Auto test ROCm
# =========================================================

cd /home/jay/ai/MoCha

echo "================================"
echo "Starting MoCha ROCm environment"
echo "================================"

# =========================================================
# Polaris compatibility
# =========================================================

export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export HSA_ENABLE_SDMA=0
export HSA_NO_SCRATCH_RECLAIM=1

# =========================================================
# HIP kernel compatibility for older GPUs
# =========================================================

export AMD_SERIALIZE_KERNEL=1
export TORCH_USE_HIP_DSA=1
export HIP_DEVICE_WAIT_MODE=0
export GPU_DEVICE_ORDINAL=0
export HIPFORT_EMIT_LINE_DIRECTIVES=0

# Optional Vulkan ICD
export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json"

# =========================================================
# Auto detect ROCm library path
# =========================================================

ROCM_LIB="$(dirname "$(find /nix/store -name libamdhip64.so 2>/dev/null | grep -v site-packages | head -1)")"

if [ -z "$ROCM_LIB" ]; then
    echo "ERROR: libamdhip64.so not found in /nix/store"
    exit 1
fi

echo "Detected ROCm lib path:"
echo "$ROCM_LIB"

# =========================================================
# Clean LD_LIBRARY_PATH
# =========================================================

export LD_LIBRARY_PATH="/run/current-system/sw/lib:/run/opengl-driver/lib:${ROCM_LIB}:${LD_LIBRARY_PATH:-}"

echo
echo "LD_LIBRARY_PATH="
echo "$LD_LIBRARY_PATH"
echo

# =========================================================
# Create venv if missing
# =========================================================

if [ ! -d ".env-rocm" ]; then
    echo "Creating Python virtual environment..."

    nix-shell -p python311 --run '
        cd /home/jay/ai/MoCha
        python -m venv .env-rocm --system-site-packages
    '
fi

# =========================================================
# Activate venv
# =========================================================

source .env-rocm/bin/activate

TORCH_LIB="$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib"

# =========================================================
# Install dependencies if torch missing
# =========================================================

if ! python -c "import torch" >/dev/null 2>&1; then
    echo "Installing PyTorch ROCm packages..."

    pip install -r requirements-amd.txt \
        --index-url https://download.pytorch.org/whl/rocm5.7

    pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/rocm5.7
fi

# =========================================================
# Auto fix broken PyTorch ROCm libraries
# =========================================================

echo
echo "Fixing PyTorch bundled ROCm libraries..."

for lib in libamdhip64.so libhiprtc.so; do
    SYSTEM_LIB="${ROCM_LIB}/${lib}"
    TORCH_TARGET="${TORCH_LIB}/${lib}"

    if [ -f "$SYSTEM_LIB" ]; then
        echo "Replacing: $lib"

        rm -f "$TORCH_TARGET"
        ln -sf "$SYSTEM_LIB" "$TORCH_TARGET"
    else
        echo "WARNING: $SYSTEM_LIB not found"
    fi
done

echo

# =========================================================
# ROCm verification
# =========================================================

echo "Testing PyTorch + ROCm..."

python - << EOF
import torch

print("================================")
print("Torch version:", torch.__version__)
print("ROCm available:", torch.cuda.is_available())
print("HIP Version:", torch.version.hip)
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")
print("================================")
EOF

echo
echo "Environment ready."
echo
cd /home/jay/ai/MoCha
# /home/jay/ai/MoCha/.env-rocm/bin/python train_mocha_colab.py \
#     --data_path ./data/train_data.csv \
#     --output_dir ./checkpoints \
#     --num_epochs 1 \
#     --max_steps 2 \
#     --learning_rate 1e-4 \
#     --use_1_3b \
#     --num_frames 24 \
#     --batch_size 1

/home/jay/ai/MoCha/.env-rocm/bin/python cache_latents.py