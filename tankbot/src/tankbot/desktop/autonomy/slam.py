"""MASt3R-SLAM adapter — dense monocular SLAM without calibration.

Wraps MASt3R-SLAM for online frame-by-frame operation. Produces dense
colored point clouds and camera poses from a monocular RGB stream.

No camera calibration required — MASt3R estimates geometry from learned
priors (Sim3 poses with relative scale).

Requires:
  - MASt3R-SLAM cloned into vendor/MASt3R-SLAM (see `task slam:setup`)
  - CUDA extensions built
  - Model checkpoints downloaded (~2.3 GB)
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
class SLAMResult:
    """Output of a single SLAM frame processing step."""

    depth_map: np.ndarray        # H×W float32 meters (at model resolution)
    camera_pose: np.ndarray      # 4×4 float64 (world-from-camera)
    tracking_quality: float      # 0..1
    num_points: int              # total points in map


def _find_mast3r_root() -> Path:
    """Locate the vendored MASt3R-SLAM directory."""
    candidates = [
        Path(__file__).resolve().parents[5] / "vendor" / "MASt3R-SLAM",
        Path.cwd() / "vendor" / "MASt3R-SLAM",
    ]
    for p in candidates:
        if (p / "main.py").exists():
            return p
    raise ImportError(
        "MASt3R-SLAM not found. Run `task slam:setup` to install it."
    )


def _ensure_on_path() -> Path:
    """Add MASt3R-SLAM to sys.path."""
    root = _find_mast3r_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _run_backend_wrapper(cfg, model, states, keyframes, mast3r_root):
    """Backend process entry point — runs global optimization + loop closure."""
    import os, sys
    os.chdir(str(mast3r_root))  # Checkpoints use relative paths
    # Add MASt3R-SLAM to path directly (can't use _find_mast3r_root after chdir)
    if str(mast3r_root) not in sys.path:
        sys.path.insert(0, str(mast3r_root))

    from mast3r_slam.config import set_global_config, config
    from mast3r_slam.global_opt import FactorGraph
    from mast3r_slam.mast3r_utils import load_retriever
    from mast3r_slam.frame import Mode

    set_global_config(cfg)
    device = keyframes.device

    factor_graph = FactorGraph(model, keyframes, keyframes.K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue

        if mode == Mode.RELOC:
            # Import relocalization from main.py
            states.dequeue_reloc()
            continue

        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        try:
            # Graph construction — connect to previous + retrieved keyframes
            kf_idx = []
            n_consec = 1
            for j in range(min(n_consec, idx)):
                kf_idx.append(idx - 1 - j)

            frame = keyframes[idx]
            retrieval_inds = retrieval_database.update(
                frame, add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds

            lc_inds = set(retrieval_inds)
            lc_inds.discard(idx - 1)
            if len(lc_inds) > 0:
                log.info("Loop closure detected at keyframe %d: %s", idx, lc_inds)

            kf_idx = list(set(kf_idx) - {idx})
            frame_idx = [idx] * len(kf_idx)
            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )

            with states.lock:
                states.edges_ii[:] = factor_graph.ii.cpu().tolist()
                states.edges_jj[:] = factor_graph.jj.cpu().tolist()

            # Solve — no calibration mode uses ray-based optimization
            factor_graph.solve_GN_rays()

            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    states.global_optimizer_tasks.pop(0)
        except Exception as e:
            log.error("Backend optimization error: %s", e)
            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    states.global_optimizer_tasks.pop(0)


class SplatSLAM:
    """Online monocular SLAM via MASt3R-SLAM.

    Usage::

        slam = SplatSLAM()
        slam.load()

        for jpeg_bytes in frame_stream:
            result = slam.process_frame(jpeg_bytes, time.monotonic())
            if result is not None:
                depth_strips = slam.get_strip_distances(result.depth_map)
    """

    WARMUP_FRAMES = 5
    MAX_KEYFRAMES = 200

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mast3r_root: Path | None = None

        # MASt3R-SLAM components (set in load())
        self._model = None
        self._tracker = None
        self._keyframes = None
        self._backend_process = None
        self._backend_queue = None
        self._last_frame = None

        # State
        self._frame_idx = 0
        self._last_quality = 0.0
        self._last_pose = np.eye(4)
        self._initialized = False
        self._img_size = 512  # MASt3R supports 224 or 512 only

    def load(self) -> None:
        """Initialize MASt3R-SLAM model and components."""
        self._mast3r_root = _ensure_on_path()

        from mast3r_slam.config import load_config, config
        from mast3r_slam.mast3r_utils import load_mast3r
        from mast3r_slam.frame import SharedKeyframes
        from mast3r_slam.tracker import FrameTracker

        # Load config — use base (no calibration, Sim3 mode)
        config_path = self._mast3r_root / "config" / "base.yaml"
        load_config(str(config_path))

        # Load the MASt3R model (use full path to checkpoint)
        weights_path = str(self._mast3r_root / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        self._model = load_mast3r(path=weights_path, device=self._device)
        log.info("MASt3R model loaded on %s", self._device)

        # Compute model image size (512px on longer side, divisible by 16)
        h, w = FRAME_HEIGHT, FRAME_WIDTH
        scale = self._img_size / max(h, w)
        self._model_h = (int(h * scale) // 16) * 16
        self._model_w = (int(w * scale) // 16) * 16

        # Shared keyframe buffer (needs multiprocessing manager)
        import multiprocessing as mp
        self._mp_manager = mp.Manager()
        self._keyframes = SharedKeyframes(
            self._mp_manager,
            h=self._model_h,
            w=self._model_w,
            buffer=self.MAX_KEYFRAMES,
        )

        # Shared states for frontend/backend communication
        from mast3r_slam.frame import SharedStates, Mode
        self._states = SharedStates(
            self._mp_manager,
            h=self._model_h,
            w=self._model_w,
        )
        self._states.set_mode(Mode.INIT)

        # Tracker (frontend)
        self._tracker = FrameTracker(self._model, self._keyframes, self._device)

        # Backend process — global optimization + loop closure
        import torch.multiprocessing as mp
        from mast3r_slam.config import config, set_global_config

        self._model.share_memory()
        self._backend_process = mp.Process(
            target=_run_backend_wrapper,
            args=(config, self._model, self._states, self._keyframes, self._mast3r_root),
            daemon=True,
        )
        self._backend_process.start()
        log.info("Backend optimization process started")

        log.info("MASt3R-SLAM loaded — no calibration mode (Sim3)")

    def process_frame(self, jpeg_bytes: bytes, timestamp: float) -> SLAMResult | None:
        """Process a single JPEG frame through MASt3R-SLAM.

        Returns None during warmup or if tracking fails.
        """
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # MASt3R expects float32 in [0.0, 1.0] — resize_img does np.uint8(img * 255)
        # so passing uint8 directly causes overflow and corrupted images
        img_float = img_rgb.astype(np.float32) / 255.0

        self._frame_idx += 1
        t0 = time.monotonic()

        from mast3r_slam.frame import create_frame, Mode
        from mast3r_slam.mast3r_utils import mast3r_inference_mono
        import lietorch

        try:
            # Create initial pose — identity for first frame, last pose for subsequent
            if not self._initialized:
                T_WC = lietorch.Sim3.Identity(1, device=self._device)
            else:
                T_WC = self._last_frame.T_WC

            # create_frame expects float32 [0,1] RGB numpy array
            frame = create_frame(
                self._frame_idx, img_float, T_WC,
                img_size=self._img_size, device=str(self._device),
            )

            if not self._initialized:
                # First frame: monocular inference to bootstrap point map
                X_init, C_init = mast3r_inference_mono(self._model, frame)
                frame.update_pointmap(X_init, C_init)
                self._keyframes.append(frame)
                self._states.queue_global_optimization(len(self._keyframes) - 1)
                self._states.set_mode(Mode.TRACKING)
                self._states.set_frame(frame)
                self._initialized = True
                self._last_frame = frame
                log.info("MASt3R-SLAM initialized from first frame")
            else:
                # Track against keyframes
                add_new_kf, match_info, try_reloc = self._tracker.track(frame)

                if try_reloc:
                    if not hasattr(self, '_reloc_count'):
                        self._reloc_count = 0
                    self._reloc_count += 1
                    if self._reloc_count == 1 or self._reloc_count % 30 == 0:
                        log.warning(
                            "Tracking lost (frame %d, %d consecutive). Move camera toward known area.",
                            self._frame_idx, self._reloc_count,
                        )
                    self._states.set_mode(Mode.RELOC)
                else:
                    self._reloc_count = 0

                self._states.set_frame(frame)

                # Log tracking diagnostics every 10 frames
                if self._frame_idx % 10 == 0:
                    pose = frame.T_WC.matrix().detach().cpu().numpy()
                    if pose.ndim == 3:
                        pose = pose[0]
                    tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
                    # Extract rotation angle from rotation matrix
                    import math
                    trace = pose[0, 0] + pose[1, 1] + pose[2, 2]
                    angle = math.degrees(math.acos(max(-1, min(1, (trace - 1) / 2))))
                    log.info(
                        "Track %d: pos=[%.3f,%.3f,%.3f] rot=%.1f° kf=%s reloc=%s",
                        self._frame_idx, tx, ty, tz, angle, add_new_kf, try_reloc,
                    )

                if add_new_kf:
                    self._keyframes.append(frame)
                    self._states.queue_global_optimization(len(self._keyframes) - 1)
                    log.info("New keyframe added (total: %d)", len(self._keyframes))

                self._last_frame = frame

        except Exception:
            log.exception("MASt3R-SLAM frame processing failed")
            return None

        elapsed = time.monotonic() - t0
        if self._frame_idx % 10 == 0:
            log.info("Frame %d: %.0fms (%.1f FPS)", self._frame_idx, elapsed * 1000, 1.0 / elapsed)

        # Extract pose (world-from-camera, 4x4)
        try:
            pose_matrix = frame.T_WC.matrix().detach().cpu().numpy().astype(np.float64)
            if pose_matrix.ndim == 3:
                pose_matrix = pose_matrix[0]  # Remove batch dim
            self._last_pose = pose_matrix
        except Exception:
            pose_matrix = self._last_pose

        # Extract depth from the dense point map
        try:
            X = frame.X_canon  # [H*W, 3] points in camera frame
            if X is not None:
                depths = X[:, 2].detach().cpu().numpy()  # Z = depth
                h, w = self._model_h, self._model_w
                depth_map = depths.reshape(h, w).astype(np.float32)
                depth_map = np.clip(depth_map, 0.01, 50.0)
            else:
                depth_map = np.ones((self._model_h, self._model_w), dtype=np.float32) * 99.0
        except Exception:
            depth_map = np.ones((self._model_h, self._model_w), dtype=np.float32) * 99.0

        # Quality from confidence map
        try:
            C = frame.C
            if C is not None:
                quality = float(C.mean().detach().cpu())
                quality = max(0.0, min(1.0, quality))
            else:
                quality = 0.5
        except Exception:
            quality = 0.5

        self._last_quality = quality

        # Count total points
        num_points = 0
        try:
            with self._keyframes.lock:
                num_kf = self._keyframes.n_size.value
            num_points = num_kf * self._model_h * self._model_w
        except Exception:
            pass

        if self._frame_idx < self.WARMUP_FRAMES:
            return None

        return SLAMResult(
            depth_map=depth_map,
            camera_pose=pose_matrix,
            tracking_quality=quality,
            num_points=num_points,
        )

    def export_ply(self, path: str) -> bool:
        """Export current map as a colored PLY point cloud."""
        if self._keyframes is None:
            log.warning("export_ply: no keyframes")
            return False

        try:
            with self._keyframes.lock:
                n = self._keyframes.n_size.value
            if n == 0:
                log.warning("export_ply: 0 keyframes stored")
                return False
            self._export_ply_manual(path)
            log.info("PLY exported: %s", path)
            return True
        except Exception:
            log.exception("Failed to export PLY")
            return False

    def _export_ply_manual(self, path: str) -> None:
        """Export point cloud from SharedKeyframes as colored PLY."""
        import lietorch
        from plyfile import PlyData, PlyElement

        kf = self._keyframes
        all_points = []
        all_colors = []

        with kf.lock:
            n = kf.n_size.value

        for i in range(n):
            try:
                with kf.lock:
                    T_WC = lietorch.Sim3(kf.T_WC[i])  # [1, 8] → Sim3
                    X = kf.X[i]         # [H*W, 3] points in camera frame
                    C = kf.C[i]         # [H*W, 1] confidence
                    uimg = kf.uimg[i]   # [H, W, 3] unnormalized image (CPU, 0-1)

                # Filter by confidence
                conf = C.squeeze(-1)
                mask = conf > 1.0  # MASt3R confidence is exp-scaled, >1 is decent

                if mask.sum() < 10:
                    continue

                # Transform to world frame
                X_world = T_WC.act(X[mask])
                pts = X_world.detach().cpu().numpy()
                if pts.ndim == 3:
                    pts = pts[0]  # Remove batch dim

                # Extract colors from uimg (float32 0-1 range, RGB order)
                mask_cpu = mask.cpu()
                img_flat = uimg.reshape(-1, 3)
                colors = (img_flat[mask_cpu].numpy() * 255).clip(0, 255).astype(np.uint8)

                all_points.append(pts)
                all_colors.append(colors)
            except Exception as e:
                log.debug("PLY export: skipped keyframe %d: %s", i, e)
                continue

        if not all_points:
            return

        pts = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        # Downsample if too many points (keep PLY manageable)
        max_points = 500_000
        if len(pts) > max_points:
            indices = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[indices]
            colors = colors[indices]

        # Atomic write
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=dir_path)
        os.close(fd)

        vertices = np.empty(len(pts), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ])
        # Camera convention: Y down, Z forward → Three.js: Y up, Z backward
        vertices['x'] = pts[:, 0]
        vertices['y'] = -pts[:, 1]  # Flip Y (camera Y-down → Three.js Y-up)
        vertices['z'] = -pts[:, 2]  # Flip Z (camera Z-forward → Three.js Z-backward)
        vertices['red'] = colors[:, 0]
        vertices['green'] = colors[:, 1]
        vertices['blue'] = colors[:, 2]

        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el], text=False).write(tmp_path)
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)

    def get_pose(self) -> np.ndarray:
        return self._last_pose.copy()

    @property
    def num_gaussians(self) -> int:
        """Compatibility alias — returns total point count."""
        try:
            n = self._keyframes.num_keyframes if self._keyframes and hasattr(self._keyframes, 'num_keyframes') else 0
            return n * self._img_size * self._img_size  # approximate
        except Exception:
            return 0

    @property
    def tracking_quality(self) -> float:
        return self._last_quality

    # ------------------------------------------------------------------
    # Depth strip helpers (same logic as vision.py)
    # ------------------------------------------------------------------
    @staticmethod
    def get_strip_distances(depth_map: np.ndarray) -> tuple[float, float, float]:
        """Nearest obstacle distance (m) in left, center, right strips."""
        h, w = depth_map.shape[:2]
        roi = depth_map[int(h * 0.3):int(h * 0.8), :]

        strip_w = w // 3
        left = float(np.percentile(roi[:, :strip_w], 10))
        center = float(np.percentile(roi[:, strip_w:2*strip_w], 10))
        right = float(np.percentile(roi[:, 2*strip_w:], 10))
        return round(left, 2), round(center, 2), round(right, 2)

    @staticmethod
    def get_center_distance(depth_map: np.ndarray) -> float:
        """Nearest obstacle distance (m) in the central region."""
        h, w = depth_map.shape[:2]
        roi = depth_map[int(h * 0.3):int(h * 0.8), int(w * 0.3):int(w * 0.7)]
        return round(float(np.percentile(roi, 10)), 2)
