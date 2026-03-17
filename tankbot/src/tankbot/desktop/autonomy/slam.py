"""3D Gaussian Splatting SLAM adapter — wraps MonoGS for online monocular operation.

Provides a clean frame-by-frame API on top of MonoGS's multi-process SLAM system.
The adapter manages the FrontEnd (tracking) and BackEnd (mapping) processes,
feeds decoded camera frames, and exposes depth rendering + PLY export.

Requires:
  - MonoGS cloned into vendor/MonoGS (see `task slam:setup`)
  - CUDA extensions built (diff-gaussian-rasterization, simple-knn)
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# Frame resolution (must match camera config)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Depth strip regions (same as vision.py)
STRIP_LEFT = (0, FRAME_WIDTH // 3)
STRIP_CENTER = (FRAME_WIDTH // 3, 2 * FRAME_WIDTH // 3)
STRIP_RIGHT = (2 * FRAME_WIDTH // 3, FRAME_WIDTH)


@dataclass
class CameraIntrinsics:
    """Pinhole camera model — Pi Camera Module 3 defaults at 640x480."""

    fx: float = 530.0
    fy: float = 530.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT


@dataclass
class SLAMResult:
    """Output of a single SLAM frame processing step."""

    depth_map: np.ndarray  # H×W float32 meters
    camera_pose: np.ndarray  # 4×4 float64 (world-to-camera)
    tracking_quality: float  # 0..1 (1 = perfect tracking)
    num_gaussians: int


def _find_monogs_root() -> Path:
    """Locate the vendored MonoGS directory."""
    candidates = [
        Path(__file__).resolve().parents[5] / "vendor" / "MonoGS",
        Path.cwd() / "vendor" / "MonoGS",
    ]
    for p in candidates:
        if (p / "slam.py").exists():
            return p
    raise ImportError(
        "MonoGS not found. Run `task slam:setup` to clone it into vendor/MonoGS."
    )


def _ensure_monogs_on_path() -> Path:
    """Add MonoGS and its submodules to sys.path if not already present."""
    root = _find_monogs_root()
    paths = [str(root), str(root / "gaussian_splatting")]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


class SplatSLAM:
    """Online monocular SLAM via 3D Gaussian Splatting (MonoGS).

    Usage::

        slam = SplatSLAM(CameraIntrinsics())
        slam.load()

        for jpeg_bytes in frame_stream:
            result = slam.process_frame(jpeg_bytes, time.monotonic())
            if result is not None:
                depth_strips = slam.get_strip_distances(result.depth_map)
                # ... use for navigation
    """

    # Minimum frames before depth output is trusted
    WARMUP_FRAMES = 30

    # Max Gaussians before pruning (keeps PLY < ~5 MB)
    MAX_GAUSSIANS = 50_000

    def __init__(self, intrinsics: CameraIntrinsics | None = None) -> None:
        self._intrinsics = intrinsics or CameraIntrinsics()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._monogs_root: Path | None = None

        # MonoGS components (set in load())
        self._gaussians = None
        self._frontend = None
        self._backend = None
        self._pipeline_params = None
        self._background = None
        self._projection_matrix = None

        # State
        self._frame_idx = 0
        self._initialized = False
        self._prev_pose: torch.Tensor | None = None
        self._prev_prev_pose: torch.Tensor | None = None
        self._keyframes: list = []
        self._current_window: list[int] = []
        self._last_quality = 0.0

        # Queues for frontend/backend communication (single-threaded mode)
        self._pending_keyframes: list = []

    def load(self) -> None:
        """Initialize MonoGS components and CUDA extensions."""
        self._monogs_root = _ensure_monogs_on_path()

        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov

        # Build config dict directly (avoids MonoGS's YAML inheritance chain)
        self._config = {
            "Results": {
                "save_results": False,
                "use_gui": False,
                "eval_rendering": False,
                "use_wandb": False,
            },
            "Dataset": {
                "type": "custom",
                "sensor_type": "monocular",
                "pcd_downsample": 64,
                "pcd_downsample_init": 32,
                "adaptive_pointsize": True,
                "point_size": 0.01,
                "Calibration": {
                    "fx": self._intrinsics.fx,
                    "fy": self._intrinsics.fy,
                    "cx": self._intrinsics.cx,
                    "cy": self._intrinsics.cy,
                    "width": self._intrinsics.width,
                    "height": self._intrinsics.height,
                },
            },
            "Training": {
                "init_itr_num": 500,
                "init_gaussian_update": 100,
                "init_gaussian_reset": 500,
                "init_gaussian_th": 0.005,
                "init_gaussian_extent": 30,
                "tracking_itr_num": 60,
                "mapping_itr_num": 80,
                "gaussian_update_every": 150,
                "gaussian_update_offset": 50,
                "gaussian_th": 0.7,
                "gaussian_extent": 1.0,
                "gaussian_reset": 2001,
                "size_threshold": 20,
                "kf_interval": 5,
                "window_size": 8,
                "pose_window": 3,
                "edge_threshold": 1.1,
                "rgb_boundary_threshold": 0.01,
                "kf_translation": 0.08,
                "kf_min_translation": 0.05,
                "kf_overlap": 0.9,
                "kf_cutoff": 0.3,
                "prune_mode": "slam",
                "single_thread": True,
                "spherical_harmonics": False,
                "lr": {
                    "cam_rot_delta": 0.003,
                    "cam_trans_delta": 0.001,
                },
            },
            "opt_params": {
                "position_lr_init": 0.0016,
                "position_lr_final": 0.0000016,
                "position_lr_delay_mult": 0.01,
                "position_lr_max_steps": 30000,
                "feature_lr": 0.0025,
                "opacity_lr": 0.05,
                "scaling_lr": 0.001,
                "rotation_lr": 0.001,
                "percent_dense": 0.01,
                "lambda_dssim": 0.2,
                "densification_interval": 100,
                "opacity_reset_interval": 3000,
                "densify_from_iter": 500,
                "densify_until_iter": 15000,
                "densify_grad_threshold": 0.0002,
            },
            "model_params": {
                "sh_degree": 0,
            },
            "pipeline_params": {
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
            },
        }

        # Initialize Gaussian model
        sh_degree = 3 if self._config["Training"].get("spherical_harmonics", False) else 0
        self._gaussians = GaussianModel(sh_degree, config=self._config)
        self._gaussians.init_lr(6.0)

        # Set up optimizer on empty model (MonoGS pattern: setup first, then extend)
        from argparse import Namespace
        opt = Namespace(**self._config["opt_params"])
        self._gaussians.training_setup(opt)

        # Pipeline params
        from argparse import Namespace
        self._pipeline_params = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
        )

        # Background color
        self._background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self._device)

        # Projection matrix from intrinsics
        W = self._intrinsics.width
        H = self._intrinsics.height
        fx = self._intrinsics.fx
        fy = self._intrinsics.fy
        cx = self._intrinsics.cx
        cy = self._intrinsics.cy

        fovx = focal2fov(fx, W)
        fovy = focal2fov(fy, H)

        self._projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=fx, fy=fy, cx=cx, cy=cy,
            W=W, H=H,
        ).transpose(0, 1).to(self._device)

        self._fovx = fovx
        self._fovy = fovy

        log.info(
            "SplatSLAM loaded on %s — %dx%d, fx=%.1f fy=%.1f",
            self._device, W, H, fx, fy,
        )

    def process_frame(self, jpeg_bytes: bytes, timestamp: float) -> SLAMResult | None:
        """Process a single JPEG frame through the SLAM pipeline.

        Returns None during warmup or if tracking fails badly.
        """
        # Decode JPEG
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to torch tensor [3, H, W] float32 on CUDA, range [0, 1]
        img_tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(self._device)
        )

        from utils.camera_utils import Camera
        from gaussian_splatting.gaussian_renderer import render

        # Create camera viewpoint
        viewpoint = Camera.init_from_gui(
            uid=self._frame_idx,
            T=torch.eye(4, dtype=torch.float64, device=self._device),
            FoVx=self._fovx,
            FoVy=self._fovy,
            fx=self._intrinsics.fx,
            fy=self._intrinsics.fy,
            cx=self._intrinsics.cx,
            cy=self._intrinsics.cy,
            H=self._intrinsics.height,
            W=self._intrinsics.width,
        )
        viewpoint.original_image = img_tensor

        # Initialize pose from previous frame
        if self._prev_pose is not None:
            if self._prev_prev_pose is not None and self._frame_idx > 200:
                # Constant velocity motion model
                velocity = self._prev_pose @ torch.inverse(self._prev_prev_pose)
                predicted_pose = velocity @ self._prev_pose
                viewpoint.update_RT(predicted_pose[:3, :3], predicted_pose[:3, 3])
            else:
                viewpoint.update_RT(self._prev_pose[:3, :3], self._prev_pose[:3, 3])

        self._frame_idx += 1

        # First frame: initialize the map
        if not self._initialized:
            self._initialize(viewpoint)
            self._initialized = True
            if self._frame_idx < self.WARMUP_FRAMES:
                return None

        # Track: optimize camera pose against current Gaussians
        tracking_quality = self._track(viewpoint)
        self._last_quality = tracking_quality

        # Store pose history for velocity model
        current_pose = self._get_camera_pose_matrix(viewpoint)
        self._prev_prev_pose = self._prev_pose
        self._prev_pose = current_pose.clone()

        # Check if this is a keyframe
        if self._is_keyframe(viewpoint):
            self._add_keyframe(viewpoint)

        # Render depth at current pose
        with torch.no_grad():
            render_pkg = render(viewpoint, self._gaussians, self._pipeline_params, self._background)
            depth_tensor = render_pkg["depth"]  # [1, H, W]

        depth_map = depth_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # Prune if too many Gaussians
        if self._gaussians.get_xyz.shape[0] > self.MAX_GAUSSIANS:
            self._prune_gaussians()

        # During warmup, return None (robot stays still)
        if self._frame_idx < self.WARMUP_FRAMES:
            return None

        return SLAMResult(
            depth_map=depth_map,
            camera_pose=current_pose.detach().cpu().numpy(),
            tracking_quality=tracking_quality,
            num_gaussians=self._gaussians.get_xyz.shape[0],
        )

    def _points_to_gaussians(self, points: torch.Tensor, colors: torch.Tensor,
                             kf_id: int = -1) -> None:
        """Add 3D points+colors as new Gaussians to the scene.

        Args:
            points: [N, 3] float CUDA tensor — world-space positions
            colors: [N, 3] float CUDA tensor — RGB in 0..1
        """
        from gaussian_splatting.utils.sh_utils import RGB2SH
        from simple_knn._C import distCUDA2
        from gaussian_splatting.utils.general_utils import inverse_sigmoid

        # Ensure float32 — distCUDA2 requires it
        points = points.float()
        colors = colors.float()

        fused_color = RGB2SH(colors)
        features = torch.zeros(
            (colors.shape[0], 3, (self._gaussians.max_sh_degree + 1) ** 2),
            dtype=torch.float32, device=self._device,
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        point_size = self._config["Dataset"].get("point_size", 0.01)
        dist2 = torch.clamp_min(distCUDA2(points), 1e-7) * point_size
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self._gaussians.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((points.shape[0], 4), device=self._device)
        rots[:, 0] = 1.0

        opacities = inverse_sigmoid(
            0.5 * torch.ones((points.shape[0], 1), dtype=torch.float32, device=self._device)
        )

        self._gaussians.extend_from_pcd(points, features, scales, rots, opacities, kf_id)

    def _initialize(self, viewpoint) -> None:
        """Initialize the Gaussian map from the first frame."""
        from gaussian_splatting.gaussian_renderer import render

        log.info("Initializing SLAM from first frame...")

        # Set first pose to identity
        viewpoint.update_RT(torch.eye(3), torch.zeros(3))

        # Create initial point cloud from the frame
        # For monocular, we create sparse points at random depths
        H, W = self._intrinsics.height, self._intrinsics.width
        num_init_points = 2000
        rng = np.random.default_rng(42)

        us = rng.integers(0, W, size=num_init_points)
        vs = rng.integers(0, H, size=num_init_points)
        depths = rng.uniform(0.5, 5.0, size=num_init_points).astype(np.float32)

        fx, fy = self._intrinsics.fx, self._intrinsics.fy
        cx, cy = self._intrinsics.cx, self._intrinsics.cy
        xs = (us - cx) * depths / fx
        ys = (vs - cy) * depths / fy

        points = np.stack([xs, ys, depths], axis=1)
        img = viewpoint.original_image.cpu().permute(1, 2, 0).numpy()
        colors = img[vs, us]

        points_t = torch.from_numpy(points).float().to(self._device)
        colors_t = torch.from_numpy(colors).float().to(self._device)

        self._points_to_gaussians(points_t, colors_t, kf_id=0)

        # Run initialization optimization
        init_itr = self._config["Training"]["init_itr_num"]
        for i in range(init_itr):
            render_pkg = render(viewpoint, self._gaussians, self._pipeline_params, self._background)
            image = render_pkg["render"]
            loss = torch.nn.functional.l1_loss(image, viewpoint.original_image)
            loss.backward()

            if i % 50 == 0 and i > 0:
                self._gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self._gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]],
                )
                self._gaussians.add_densification_stats(
                    render_pkg["viewspace_points"], render_pkg["visibility_filter"]
                )

                if i > 100:
                    self._gaussians.densify_and_prune(
                        self._config["opt_params"]["densify_grad_threshold"],
                        0.005,
                        30.0,  # scene_extent (init_gaussian_extent)
                        self.MAX_GAUSSIANS,
                    )

            self._gaussians.optimizer.step()
            self._gaussians.optimizer.zero_grad(set_to_none=True)

        self._keyframes.append(viewpoint)
        self._current_window = [0]

        log.info(
            "SLAM initialized: %d Gaussians from first frame",
            self._gaussians.get_xyz.shape[0],
        )

    def _track(self, viewpoint) -> float:
        """Optimize camera pose for the current frame. Returns tracking quality 0..1."""
        from gaussian_splatting.gaussian_renderer import render

        # Create optimizer for pose parameters
        opt_params = [
            {"params": [viewpoint.cam_rot_delta], "lr": 0.003},
            {"params": [viewpoint.cam_trans_delta], "lr": 0.001},
        ]
        if hasattr(viewpoint, "exposure_a"):
            opt_params.extend([
                {"params": [viewpoint.exposure_a], "lr": 0.01},
                {"params": [viewpoint.exposure_b], "lr": 0.01},
            ])
        pose_optimizer = torch.optim.Adam(opt_params)

        tracking_itr = self._config["Training"]["tracking_itr_num"]
        best_loss = float("inf")
        converged_count = 0

        for i in range(tracking_itr):
            render_pkg = render(viewpoint, self._gaussians, self._pipeline_params, self._background)
            image = render_pkg["render"]

            loss = torch.nn.functional.l1_loss(image, viewpoint.original_image)
            loss_val = loss.item()

            loss.backward()
            pose_optimizer.step()
            pose_optimizer.zero_grad()

            # Apply pose update
            from utils.pose_utils import update_pose
            update_pose(viewpoint)

            # Check convergence
            if loss_val < best_loss:
                best_loss = loss_val
                converged_count = 0
            else:
                converged_count += 1

            if converged_count > 10:
                break

            # Check for very small updates
            delta_norm = viewpoint.cam_rot_delta.norm() + viewpoint.cam_trans_delta.norm()
            if delta_norm.item() < 1e-4:
                break

        # Quality metric: inverse of photometric loss, normalized to 0..1
        # With monocular init from random depths, early losses are high (~0.3-0.5)
        # so we use a generous scale. As the map improves, losses drop to ~0.02-0.1.
        quality = max(0.0, min(1.0, 1.0 - best_loss / 0.5))
        return round(quality, 3)

    def _is_keyframe(self, viewpoint) -> bool:
        """Decide if the current frame should be added as a keyframe."""
        if not self._keyframes:
            return True

        kf_interval = self._config["Training"].get("kf_interval", 4)
        if self._frame_idx % kf_interval != 0:
            return False

        # Translation-based trigger
        current_pose = self._get_camera_pose_matrix(viewpoint)
        last_kf_pose = self._get_camera_pose_matrix(self._keyframes[-1])
        translation = (current_pose[:3, 3] - last_kf_pose[:3, 3]).norm().item()

        kf_translation = self._config["Training"].get("kf_translation", 0.04)
        return translation > kf_translation

    def _add_keyframe(self, viewpoint) -> None:
        """Add a keyframe and run mapping optimization."""
        from gaussian_splatting.gaussian_renderer import render

        self._keyframes.append(viewpoint)
        kf_idx = len(self._keyframes) - 1

        # Update sliding window
        window_size = self._config["Training"]["window_size"]
        self._current_window.append(kf_idx)
        if len(self._current_window) > window_size:
            self._current_window = self._current_window[-window_size:]

        # Try to add new Gaussians from this keyframe
        self._extend_gaussians_from_frame(viewpoint)

        # Mapping: optimize Gaussians using keyframes in the window
        mapping_itr = self._config["Training"]["mapping_itr_num"]
        for i in range(mapping_itr):
            # Pick a random keyframe from the window
            kf = self._keyframes[self._current_window[i % len(self._current_window)]]

            render_pkg = render(kf, self._gaussians, self._pipeline_params, self._background)
            image = render_pkg["render"]
            loss = torch.nn.functional.l1_loss(image, kf.original_image)
            loss.backward()

            if i % 30 == 0 and i > 0:
                self._gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self._gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]],
                )
                self._gaussians.add_densification_stats(
                    render_pkg["viewspace_points"], render_pkg["visibility_filter"]
                )
                self._gaussians.densify_and_prune(
                    self._config["opt_params"]["densify_grad_threshold"],
                    0.005,
                    1.0,  # scene_extent for keyframes
                    self.MAX_GAUSSIANS,
                )

            self._gaussians.optimizer.step()
            self._gaussians.optimizer.zero_grad()

        log.debug(
            "Keyframe %d added, window=%s, gaussians=%d",
            kf_idx, self._current_window, self._gaussians.get_xyz.shape[0],
        )

    def _extend_gaussians_from_frame(self, viewpoint) -> None:
        """Create new Gaussians from under-reconstructed regions of the frame."""
        from gaussian_splatting.gaussian_renderer import render

        with torch.no_grad():
            render_pkg = render(viewpoint, self._gaussians, self._pipeline_params, self._background)
            opacity = render_pkg["opacity"].squeeze(0)  # [H, W]
            depth = render_pkg["depth"].squeeze(0)  # [H, W]

        # Find pixels with low opacity (not well covered)
        mask = opacity < 0.5
        if mask.sum() < 100:
            return

        # Sample sparse points from uncovered regions
        ys, xs = torch.where(mask)
        if len(xs) > 500:
            indices = torch.randperm(len(xs))[:500]
            xs, ys = xs[indices], ys[indices]

        # Unproject at estimated depths (use rendered depth where available, else random)
        fx, fy = self._intrinsics.fx, self._intrinsics.fy
        cx, cy = self._intrinsics.cx, self._intrinsics.cy

        depths = depth[ys, xs]
        # Replace zero/invalid depths with random indoor-range depths
        invalid = depths < 0.1
        depths[invalid] = torch.FloatTensor(invalid.sum().item()).uniform_(0.5, 3.0).to(self._device)

        xs_f = xs.float()
        ys_f = ys.float()
        zs = depths
        x3d = (xs_f - cx) * zs / fx
        y3d = (ys_f - cy) * zs / fy

        points_cam = torch.stack([x3d, y3d, zs], dim=1).float()  # [N, 3]

        # Transform to world coordinates
        pose = self._get_camera_pose_matrix(viewpoint).float()  # world-to-camera
        R = pose[:3, :3]
        t = pose[:3, 3]
        # camera-to-world: p_world = R^T @ (p_cam - t)
        points_world = (points_cam - t.unsqueeze(0)) @ R  # [N, 3]

        # Get colors
        img = viewpoint.original_image  # [3, H, W]
        colors = img[:, ys.long(), xs.long()].permute(1, 0).float()  # [N, 3]

        self._points_to_gaussians(points_world, colors, kf_id=len(self._keyframes))

    def _prune_gaussians(self) -> None:
        """Remove low-opacity Gaussians to keep the scene manageable."""
        prune_mask = (self._gaussians.get_opacity < 0.1).squeeze()
        if prune_mask.sum() > 0:
            self._gaussians.prune_points(prune_mask)
            log.debug("Pruned %d low-opacity Gaussians", prune_mask.sum().item())

    def _get_camera_pose_matrix(self, viewpoint) -> torch.Tensor:
        """Extract the 4x4 world-to-camera matrix from a Camera viewpoint."""
        R = viewpoint.R  # [3, 3]
        T = viewpoint.T  # [3] translation
        pose = torch.eye(4, dtype=torch.float64, device=self._device)
        pose[:3, :3] = R.to(dtype=torch.float64)
        pose[:3, 3] = T.to(dtype=torch.float64)
        return pose

    def export_ply(self, path: str) -> bool:
        """Export current Gaussian scene to PLY file (atomic write)."""
        if self._gaussians is None or self._gaussians.get_xyz.shape[0] == 0:
            return False

        try:
            # Write to temp file then atomically rename
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=dir_path)
            os.close(fd)

            self._gaussians.save_ply(tmp_path)
            os.replace(tmp_path, path)

            log.debug("Exported PLY: %d Gaussians → %s", self._gaussians.get_xyz.shape[0], path)
            return True
        except Exception:
            log.exception("Failed to export PLY")
            return False

    def get_pose(self) -> np.ndarray:
        """Return current camera pose as 4x4 numpy array."""
        if self._prev_pose is not None:
            return self._prev_pose.detach().cpu().numpy()
        return np.eye(4)

    @property
    def num_gaussians(self) -> int:
        if self._gaussians is None:
            return 0
        return self._gaussians.get_xyz.shape[0]

    @property
    def tracking_quality(self) -> float:
        return self._last_quality

    # ------------------------------------------------------------------
    # Depth strip helpers (same logic as the old DepthEstimator)
    # ------------------------------------------------------------------
    @staticmethod
    def get_strip_distances(depth_map: np.ndarray) -> tuple[float, float, float]:
        """Nearest obstacle distance (m) in left, center, right strips.

        Uses the middle band (rows 30%-80%) to ignore floor/ceiling.
        Returns the 10th percentile per strip.
        """
        h = depth_map.shape[0]
        roi = depth_map[int(h * 0.3):int(h * 0.8), :]

        left = float(np.percentile(roi[:, STRIP_LEFT[0]:STRIP_LEFT[1]], 10))
        center = float(np.percentile(roi[:, STRIP_CENTER[0]:STRIP_CENTER[1]], 10))
        right = float(np.percentile(roi[:, STRIP_RIGHT[0]:STRIP_RIGHT[1]], 10))
        return round(left, 2), round(center, 2), round(right, 2)

    @staticmethod
    def get_center_distance(depth_map: np.ndarray) -> float:
        """Nearest obstacle distance (m) in the central region."""
        h, w = depth_map.shape
        roi = depth_map[int(h * 0.3):int(h * 0.8), int(w * 0.3):int(w * 0.7)]
        return round(float(np.percentile(roi, 10)), 2)
