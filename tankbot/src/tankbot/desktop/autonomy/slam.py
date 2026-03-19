"""MASt3R-SLAM adapter — dense monocular SLAM with optional calibration.

Wraps MASt3R-SLAM for online frame-by-frame operation. Produces dense
colored point clouds and camera poses from a monocular RGB stream.

Camera calibration is optional. Without intrinsics, MASt3R estimates
geometry from learned priors (Sim3 poses with relative scale). With
intrinsics enabled, the backend switches to the calibrated optimizer.

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
from typing import Any

import cv2
import numpy as np
import torch
import yaml

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
    num_keyframes: int = 0       # total keyframes currently stored
    tracking_lost: bool = False  # True when tracker requests relocalization
    new_keyframe: bool = False   # True when a new keyframe was just added


@dataclass
class CameraCalibration:
    """Undistortion maps plus resized-frame intrinsics for calibrated mode."""

    K_frame: np.ndarray
    mapx: np.ndarray
    mapy: np.ndarray

    def remap(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)


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


def _attempt_relocalization(states, keyframes, factor_graph, retrieval_database, cfg: dict) -> bool:
    """Mirror upstream retrieval-based relocalization for lost tracking."""
    frame = states.get_frame()
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=cfg["retrieval"]["k"],
            min_thresh=cfg["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False

        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print(f"[Backend] RELOCALIZING against KF {n_kf - 1} and {kf_idx}", flush=True)

            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                cfg["reloc"]["min_match_frac"],
                is_reloc=cfg["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=cfg["retrieval"]["k"],
                    min_thresh=cfg["retrieval"]["min_thresh"],
                )
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
                successful_loop_closure = True
                print("[Backend] Relocalization succeeded", flush=True)
            else:
                keyframes.pop_last()
                print("[Backend] Relocalization failed", flush=True)

        if successful_loop_closure:
            if cfg["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()

        return successful_loop_closure


def _run_backend_wrapper(cfg, model, states, keyframes, K, mast3r_root):
    """Backend process entry point — runs global optimization + loop closure."""
    import os, sys, traceback
    os.chdir(str(mast3r_root))  # Checkpoints use relative paths
    if str(mast3r_root) not in sys.path:
        sys.path.insert(0, str(mast3r_root))

    try:
        from mast3r_slam.config import set_global_config, config
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_retriever
        from mast3r_slam.frame import Mode

        set_global_config(cfg)
        device = keyframes.device

        factor_graph = FactorGraph(model, keyframes, K, device)
        retrieval_database = load_retriever(model)
        print("[Backend] Initialized successfully", flush=True)
    except Exception as e:
        print(f"[Backend] FATAL: Failed to initialize: {e}", flush=True)
        traceback.print_exc()
        return

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue

        if mode == Mode.RELOC:
            with states.lock:
                pending_reloc = states.reloc_sem.value
            if pending_reloc == 0:
                time.sleep(0.01)
                continue
            success = _attempt_relocalization(
                states, keyframes, factor_graph, retrieval_database, config
            )
            if success:
                states.set_mode(Mode.TRACKING)
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
                print(f"[Backend] Loop closure at KF {idx} → matched KFs {lc_inds}", flush=True)

            kf_idx = list(set(kf_idx) - {idx})
            frame_idx = [idx] * len(kf_idx)

            n_factors_before = len(factor_graph.ii) if hasattr(factor_graph, 'ii') else 0

            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )

            n_factors_after = len(factor_graph.ii) if hasattr(factor_graph, 'ii') else 0

            with states.lock:
                states.edges_ii[:] = factor_graph.ii.cpu().tolist()
                states.edges_jj[:] = factor_graph.jj.cpu().tolist()

            if config["use_calib"]:
                factor_graph.solve_GN_calib()
                solve_mode = "calib"
            else:
                factor_graph.solve_GN_rays()
                solve_mode = "rays"

            print(
                f"[Backend] KF {idx}: {n_factors_after} factors, retrieval={retrieval_inds} "
                f"(solve={solve_mode})",
                flush=True,
            )

            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    states.global_optimizer_tasks.pop(0)
        except Exception as e:
            import traceback
            print(f"[Backend] ERROR optimizing KF {idx}: {e}", flush=True)
            traceback.print_exc()
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

    def __init__(self, calibration_path: str = "") -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mast3r_root: Path | None = None
        self._calibration_path = calibration_path or os.environ.get("ROBOT_CAMERA_CALIB", "")

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
        self._use_calib = False
        self._intrinsics = None

    def _resolve_calibration_path(self) -> Path | None:
        if not self._calibration_path:
            return None

        candidates = [Path(self._calibration_path)]
        if self._mast3r_root is not None:
            candidates.append(self._mast3r_root / self._calibration_path)

        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Calibration file not found: {self._calibration_path}"
        )

    def _load_calibration(self) -> tuple[Any, torch.Tensor] | tuple[None, None]:
        calib_path = self._resolve_calibration_path()
        if calib_path is None:
            return None, None
        from mast3r_slam.mast3r_utils import resize_img

        with calib_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        width = int(data["width"])
        height = int(data["height"])
        calib = data["calibration"]
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:], dtype=np.float32)

        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        center = True
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (width, height), 0, (width, height), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (width, height), cv2.CV_32FC1
        )

        dummy = np.zeros((height, width, 3), dtype=np.float32)
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            dummy, self._img_size, return_transformation=True
        )
        K_frame = K_opt.copy()
        K_frame[0, 0] = K_opt[0, 0] / scale_w
        K_frame[1, 1] = K_opt[1, 1] / scale_h
        K_frame[0, 2] = K_opt[0, 2] / scale_w - half_crop_w
        K_frame[1, 2] = K_opt[1, 2] / scale_h - half_crop_h

        intrinsics = CameraCalibration(K_frame=K_frame, mapx=mapx, mapy=mapy)

        K = torch.from_numpy(intrinsics.K_frame).to(self._device, dtype=torch.float32)
        log.info("Calibration mode enabled using %s", calib_path)
        return intrinsics, K

    def load(self) -> None:
        """Initialize MASt3R-SLAM model and components."""
        self._mast3r_root = _ensure_on_path()

        from mast3r_slam.config import load_config, config
        from mast3r_slam.mast3r_utils import load_mast3r
        from mast3r_slam.frame import SharedKeyframes
        from mast3r_slam.tracker import FrameTracker

        calib_path = self._resolve_calibration_path()
        config_name = "calib.yaml" if calib_path is not None else "base.yaml"
        config_path = self._mast3r_root / "config" / config_name
        prev_cwd = Path.cwd()
        try:
            os.chdir(self._mast3r_root)
            load_config(f"config/{config_name}")
        finally:
            os.chdir(prev_cwd)
        self._use_calib = bool(config.get("use_calib", False))

        # Upstream defaults are conservative for offline datasets. Our live robot stream
        # benefits from broader retrieval and less strict relocalization acceptance.
        config["retrieval"]["k"] = max(int(config["retrieval"]["k"]), 5)
        config["retrieval"]["min_thresh"] = min(float(config["retrieval"]["min_thresh"]), 1e-3)
        config["reloc"]["min_match_frac"] = min(float(config["reloc"]["min_match_frac"]), 0.15)
        config["reloc"]["strict"] = False
        log.info(
            "Live relocalization tuning: retrieval.k=%d min_thresh=%.4g reloc.min_match_frac=%.2f strict=%s",
            int(config["retrieval"]["k"]),
            float(config["retrieval"]["min_thresh"]),
            float(config["reloc"]["min_match_frac"]),
            bool(config["reloc"]["strict"]),
        )

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

        K = None
        if self._use_calib:
            self._intrinsics, K = self._load_calibration()
            self._keyframes.set_intrinsics(K)

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
            args=(config, self._model, self._states, self._keyframes, K, self._mast3r_root),
            daemon=True,
        )
        self._backend_process.start()
        log.info("Backend optimization process started")

        if self._use_calib:
            log.info("MASt3R-SLAM loaded — calibration mode active")
        else:
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
        if self._intrinsics is not None:
            img_rgb = self._intrinsics.remap(img_rgb)

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
                mode = self._states.get_mode()

                if mode == Mode.RELOC:
                    X, C = mast3r_inference_mono(self._model, frame)
                    frame.update_pointmap(X, C)
                    self._states.set_frame(frame)
                    self._states.queue_reloc()
                    add_new_kf = False
                    try_reloc = True
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
                        if getattr(self, '_reloc_count', 0) > 0:
                            # Recovered from reloc — re-enable backend optimization
                            self._states.set_mode(Mode.TRACKING)
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

                self._new_kf_this_frame = False
                if add_new_kf:
                    self._keyframes.append(frame)
                    self._states.queue_global_optimization(len(self._keyframes) - 1)
                    self._new_kf_this_frame = True
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
        # MASt3R Sim3 mode: depths are relative, not metric.
        # depth_scale is dynamically calibrated against ultrasonic in vision.py
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
        num_kf = 0
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
            num_keyframes=num_kf,
            tracking_lost=getattr(self, '_reloc_count', 0) > 0,
            new_keyframe=getattr(self, '_new_kf_this_frame', False),
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

                conf = C.squeeze(-1)
                mask = conf > 1.0

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
