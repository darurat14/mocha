#!/bin/bash

# Agar GPU Polaris (RX 560) dikenali ROCm
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export HSA_ENABLE_SDMA=0
export HSA_NO_SCRATCH_RECLAIM=1


# NixOS-specific library paths
export LD_LIBRARY_PATH="/run/current-system/sw/lib:/run/opengl-driver/lib:$(nix eval --raw nixpkgs#openvino)/runtime/lib/intel64:$NIX_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json"

echo "================================"
cd /home/jay/ai/MoCha
/home/jay/ai/MoCha/.env-cpu/bin/python train_mocha_lora.py \
    --data_path ./data/train_data.csv \
    --output_dir ./checkpoints \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --max_steps 5 \
    --num_workers 2
