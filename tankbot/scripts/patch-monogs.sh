#!/usr/bin/env bash
# Patches vendored MonoGS for compatibility with:
#   - CUDA 13.x (missing <cstdint> in C++ headers)
#   - System CUDA / PyTorch CUDA version mismatch
#   - Python 3.14 (no open3d wheels available)
#
# Run after cloning MonoGS into vendor/MonoGS.

set -euo pipefail

MONOGS_DIR="${1:-vendor/MonoGS}"

if [ ! -d "$MONOGS_DIR" ]; then
    echo "ERROR: MonoGS not found at $MONOGS_DIR"
    exit 1
fi

echo "Patching MonoGS at $MONOGS_DIR..."

# ---------------------------------------------------------------------------
# 1. Fix CUDA 13.x C++20 compilation — add missing <cstdint> includes
# ---------------------------------------------------------------------------
RASTERIZER="$MONOGS_DIR/submodules/diff-gaussian-rasterization"

# rasterizer.h — root header used everywhere
sed -i '/#include <vector>/i #include <cstdint>' \
    "$RASTERIZER/cuda_rasterizer/rasterizer.h"

# auxiliary.h — included by all .cu files
sed -i '/#include "stdio.h"/a #include <cstdint>' \
    "$RASTERIZER/cuda_rasterizer/auxiliary.h"

# rasterize_points.h — pybind bridge
sed -i '/#pragma once/a #include <cstdint>' \
    "$RASTERIZER/rasterize_points.h"

# ---------------------------------------------------------------------------
# 2. Bypass CUDA version check in setup.py (system CUDA != torch CUDA)
# ---------------------------------------------------------------------------
for setup_py in \
    "$RASTERIZER/setup.py" \
    "$MONOGS_DIR/submodules/simple-knn/setup.py"; do

    if grep -q "_check_cuda_version" "$setup_py"; then
        echo "  Already patched: $setup_py"
    else
        sed -i '/^from setuptools import setup/a \
import torch.utils.cpp_extension\n\
# Allow building with a newer system CUDA than the one torch was compiled with\n\
torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None' \
            "$setup_py"
    fi
done

# ---------------------------------------------------------------------------
# 3. Remove open3d dependency — replace with numpy unprojection
#    (open3d has no Python 3.14 wheels)
# ---------------------------------------------------------------------------
GAUSSIAN_MODEL="$MONOGS_DIR/gaussian_splatting/scene/gaussian_model.py"

# Remove the open3d import line
sed -i '/^import open3d as o3d$/d' "$GAUSSIAN_MODEL"

# Replace create_pcd_from_image to pass raw arrays instead of o3d.geometry.Image
python3 -c "
import re, sys

path = '$GAUSSIAN_MODEL'
with open(path, 'r') as f:
    src = f.read()

# Replace create_pcd_from_image: pass raw numpy arrays instead of o3d.Image
old_img_block = '''            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))'''
new_img_block = '''            rgb = rgb_raw.astype(np.uint8)
            depth = depthmap.astype(np.float32)'''
src = src.replace(old_img_block, new_img_block)

old_img_block2 = '''            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))'''
new_img_block2 = '''            rgb = rgb_raw.astype(np.uint8)
            depth = depth_raw.astype(np.float32)'''
src = src.replace(old_img_block2, new_img_block2)

# Replace create_pcd_from_image_and_depth: numpy unprojection instead of o3d
old_pcd = '''        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)'''

new_pcd = '''        # Unproject RGBD to point cloud using numpy (replaces open3d)
        H, W = depth.shape[:2]
        depth_arr = np.asarray(depth, dtype=np.float32)
        rgb_arr = np.asarray(rgb, dtype=np.float32)
        if rgb_arr.max() > 1.0:
            rgb_arr = rgb_arr / 255.0

        valid = (depth_arr > 0) & (depth_arr < 100.0)
        vs, us = np.where(valid)

        n_total = len(vs)
        n_keep = max(1, int(n_total / downsample_factor))
        if n_keep < n_total:
            indices = np.random.choice(n_total, n_keep, replace=False)
            vs, us = vs[indices], us[indices]

        zs = depth_arr[vs, us]
        xs = (us.astype(np.float32) - cam.cx) * zs / cam.fx
        ys = (vs.astype(np.float32) - cam.cy) * zs / cam.fy
        pts_cam = np.stack([xs, ys, zs], axis=1)

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        C2W = np.linalg.inv(W2C)
        pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
        pts_world = (C2W @ pts_h.T).T[:, :3]

        new_xyz = pts_world
        new_rgb = rgb_arr[vs, us]'''

src = src.replace(old_pcd, new_pcd)

with open(path, 'w') as f:
    f.write(src)

print('  Patched gaussian_model.py: removed open3d, added numpy unprojection')
"

echo "MonoGS patched successfully."
