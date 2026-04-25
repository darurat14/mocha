#!/bin/bash

# Agar GPU Polaris (RX 560) dikenali ROCm
export HSA_OVERRIDE_GFX_VERSION=8.0.3

# NixOS-specific library paths
export LD_LIBRARY_PATH="/run/current-system/sw/lib:/run/opengl-driver/lib:$(nix eval --raw nixpkgs#openvino)/runtime/lib/intel64:$NIX_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json"

echo "================================"
cd /home/jay/ai/MoCha

if [ ! -d ".env-cpu" ]; then
    python -m venv .env-cpu --system-site-packages
fi

source .env-cpu/bin/activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-amd.txt --extra-index-url https://download.pytorch.org/whl/cpu
# pip install transformers==4.41.2 accelerate==0.30.1
