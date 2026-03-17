#!/usr/bin/env bash
# Patches MASt3R-SLAM and its dependencies for compatibility with:
#   - CUDA 13.x (dropped old compute capabilities, API changes)
#   - PyTorch 2.10+ (deprecated .type() → .scalar_type(), namespace changes)
#   - Python 3.14 (numpy version, missing wheels)

set -euo pipefail

MAST3R_DIR="${1:-vendor/MASt3R-SLAM}"

if [ ! -d "$MAST3R_DIR" ]; then
    echo "ERROR: MASt3R-SLAM not found at $MAST3R_DIR"
    exit 1
fi

echo "Patching MASt3R-SLAM..."

# ---------------------------------------------------------------------------
# 1. Disable torch's CUDA version check (system CUDA 13.x vs torch cu128)
# ---------------------------------------------------------------------------
CPP_EXT="$(uv run python -c 'import torch.utils.cpp_extension as e; print(e.__file__)' 2>/dev/null || true)"
if [ -n "$CPP_EXT" ] && [ -f "$CPP_EXT" ]; then
    if grep -q "CUDA check disabled" "$CPP_EXT"; then
        echo "  [1/5] torch CUDA check already disabled"
    else
        sed -i '/^def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:/{
            a\    # CUDA check disabled by patch-mast3r-slam.sh
            a\    return
        }' "$CPP_EXT"
        echo "  [1/5] Disabled torch CUDA version check"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Fix curope: use TORCH_CUDA_ARCH_LIST and fix .type() deprecation
# ---------------------------------------------------------------------------
CUROPE_DIR="$MAST3R_DIR/thirdparty/mast3r/dust3r/croco/models/curope"
if [ -f "$CUROPE_DIR/kernels.cu" ]; then
    sed -i 's/\.type(), "/\.scalar_type(), "/g' "$CUROPE_DIR/kernels.cu"
    echo "  [2/5] Fixed curope kernels.cu (.type() → .scalar_type())"
fi

if [ -f "$CUROPE_DIR/setup.py" ]; then
    # Replace arch detection with TORCH_CUDA_ARCH_LIST support
    python3 -c "
p = '$CUROPE_DIR/setup.py'
with open(p) as f:
    src = f.read()
if 'TORCH_CUDA_ARCH_LIST' not in src:
    old = \"all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()\"
    new = '''import os
if os.environ.get('TORCH_CUDA_ARCH_LIST'):
    archs = os.environ['TORCH_CUDA_ARCH_LIST'].replace(' ', ';').split(';')
    all_cuda_archs = []
    for a in archs:
        a = a.strip().replace('.', '')
        if a:
            all_cuda_archs += ['-gencode', f'arch=compute_{a},code=sm_{a}']
else:
    all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()'''
    src = src.replace(old, new)
    with open(p, 'w') as f:
        f.write(src)
    print('  [2/5] Fixed curope setup.py (TORCH_CUDA_ARCH_LIST support)')
else:
    print('  [2/5] curope setup.py already patched')
"
fi

# ---------------------------------------------------------------------------
# 3. Fix MASt3R-SLAM backend: CUDA API compatibility
# ---------------------------------------------------------------------------
GN_KERNELS="$MAST3R_DIR/mast3r_slam/backend/src/gn_kernels.cu"
if [ -f "$GN_KERNELS" ]; then
    # torch::linalg::linalg_norm → at::linalg_norm
    sed -i 's/torch::linalg::linalg_norm/at::linalg_norm/g' "$GN_KERNELS"
    echo "  [3/5] Fixed gn_kernels.cu (linalg namespace)"
fi

MATCH_KERNELS="$MAST3R_DIR/mast3r_slam/backend/src/matching_kernels.cu"
if [ -f "$MATCH_KERNELS" ]; then
    sed -i 's/\.type(), "/\.scalar_type(), "/g' "$MATCH_KERNELS"
    echo "  [3/5] Fixed matching_kernels.cu (.type() → .scalar_type())"
fi

# ---------------------------------------------------------------------------
# 4. Fix MASt3R-SLAM setup.py: arch list + cleanup bad patches
# ---------------------------------------------------------------------------
SETUP_PY="$MAST3R_DIR/setup.py"
if [ -f "$SETUP_PY" ]; then
    python3 -c "
p = '$SETUP_PY'
with open(p) as f:
    src = f.read()
if 'TORCH_CUDA_ARCH_LIST' not in src:
    old = '''    extra_compile_args[\"nvcc\"] = [
        \"-O3\",
        \"-gencode=arch=compute_60,code=sm_60\",
        \"-gencode=arch=compute_61,code=sm_61\",
        \"-gencode=arch=compute_70,code=sm_70\",
        \"-gencode=arch=compute_75,code=sm_75\",
        \"-gencode=arch=compute_80,code=sm_80\",
        \"-gencode=arch=compute_86,code=sm_86\",
    ]'''
    new = '''    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.9').replace(' ', ';').split(';')
    gencode_flags = []
    for a in arch_list:
        a = a.strip().replace('.', '')
        if a:
            gencode_flags.append(f'-gencode=arch=compute_{a},code=sm_{a}')
    extra_compile_args[\"nvcc\"] = [\"-O3\"] + gencode_flags'''
    src = src.replace(old, new)
    with open(p, 'w') as f:
        f.write(src)
    print('  [4/5] Fixed MASt3R-SLAM setup.py (arch list)')
else:
    print('  [4/5] MASt3R-SLAM setup.py already patched')
"
fi

# ---------------------------------------------------------------------------
# 5. Relax dependency pins (Python 3.14 compat)
# ---------------------------------------------------------------------------
PYPROJECT="$MAST3R_DIR/pyproject.toml"
if [ -f "$PYPROJECT" ]; then
    sed -i 's/"numpy==1.26.4"/"numpy>=1.26"/g' "$PYPROJECT"
    # Remove deps that aren't needed for our use case
    sed -i '/"pyrealsense2"/d' "$PYPROJECT"
    sed -i '/"evo"/d' "$PYPROJECT"
    echo "  [5/5] Relaxed dependency pins"
fi

echo "MASt3R-SLAM patched successfully."
