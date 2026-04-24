#!/bin/bash

# Agar GPU Polaris (RX 560) dikenali ROCm
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export HSA_ENABLE_SDMA=0
export HSA_NO_SCRATCH_RECLAIM=1


# NixOS-specific library paths
export LD_LIBRARY_PATH="/run/current-system/sw/lib:/run/opengl-driver/lib:$(nix eval --raw nixpkgs#openvino)/runtime/lib/intel64:$NIX_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json"

nix-shell -p python311 --run '


echo "================================"
cd /home/jay/ai/MoCha

if [ ! -d ".env-rocm" ]; then
    python -m venv .env-rocm --system-site-packages


    source .env-rocm/bin/activate

    # Agar GPU Polaris (RX 560) dikenali ROCm
    export ROC_ENABLE_PRE_VEGA=1
    export HSA_OVERRIDE_GFX_VERSION=8.0.3
    export HSA_ENABLE_SDMA=0
    export HSA_NO_SCRATCH_RECLAIM=1


    # NixOS-specific library paths
    export LD_LIBRARY_PATH="/run/current-system/sw/lib:/run/opengl-driver/lib:$(nix eval --raw nixpkgs#openvino)/runtime/lib/intel64:$NIX_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
    export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json"


    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/rocm5.7
    #pip install -r requirements-amd.txt --index-url https://download.pytorch.org/whl/rocm5.7
fi
'
echo LD_LIBRARY_PATH
source .env-rocm/bin/activate

#python -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.0

python -c "import torch; print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"


python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

python -c "import torch; print(f'HIP Version: {torch.version.hip}')"
