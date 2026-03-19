"""Interactive camera calibration tool for the robot video stream.

Captures checkerboard views from the robot over WebSocket, estimates camera
intrinsics with OpenCV, and writes a MASt3R-SLAM-compatible YAML file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tankbot.desktop.autonomy.robot_client import RobotClient


@dataclass
class Capture:
    image_points: np.ndarray
    frame_bgr: np.ndarray
    accepted_at: float


class FrameStream:
    def __init__(self, robot_url: str) -> None:
        self._client = RobotClient(robot_url)
        self.latest_frame: bytes | None = None

    async def _on_frame(self, jpeg: bytes) -> None:
        self.latest_frame = jpeg

    async def connect(self) -> None:
        self._client.on_frame(self._on_frame)
        self._client.on_telemetry(lambda _: None)
        await self._client.connect()

    async def listen(self) -> None:
        await self._client.listen()

    async def close(self) -> None:
        await self._client.close()


def _board_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_mm
    return objp


def _detect_checkerboard(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if found and corners is not None:
        return True, corners.astype(np.float32)

    legacy_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=legacy_flags)
    if not found or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined.astype(np.float32)


def _draw_overlay(frame_bgr: np.ndarray, pattern_size: tuple[int, int], found: bool, corners: np.ndarray | None) -> np.ndarray:
    overlay = frame_bgr.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(overlay, pattern_size, corners, found)
        status = "checkerboard detected"
        color = (0, 220, 0)
    else:
        status = "checkerboard not found"
        color = (0, 0, 255)

    cv2.putText(overlay, status, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "space=save  c=calibrate  q=quit",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _write_calibration_yaml(path: Path, width: int, height: int, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    dist = dist_coeffs.reshape(-1).tolist()
    calibration = [fx, fy, cx, cy] + [float(x) for x in dist]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"width: {width}\n")
        f.write(f"height: {height}\n")
        f.write("# calibration: [fx, fy, cx, cy, k1, k2, p1, p2, k3]\n")
        f.write("calibration: [" + ", ".join(f"{x:.8f}" for x in calibration) + "]\n")


def _print_result(rms: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, output: Path) -> None:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    dist = dist_coeffs.reshape(-1).tolist()
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")
    print("distortion=" + ", ".join(f"{x:.6f}" for x in dist))
    print(f"Wrote calibration YAML: {output}")


async def run(args: argparse.Namespace) -> int:
    pattern_size = (args.cols, args.rows)
    output = Path(args.output)
    captures_dir = Path(args.captures_dir)
    captures_dir.mkdir(parents=True, exist_ok=True)

    stream = FrameStream(args.robot)
    await stream.connect()
    listener = asyncio.create_task(stream.listen())

    captures: list[Capture] = []
    objp = _board_object_points(args.cols, args.rows, args.square_mm)
    image_size: tuple[int, int] | None = None
    last_save = 0.0

    try:
        while True:
            jpeg = stream.latest_frame
            if jpeg is None:
                await asyncio.sleep(0.02)
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await asyncio.sleep(0.02)
                continue

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = _detect_checkerboard(gray, pattern_size)
            overlay = _draw_overlay(frame, pattern_size, found, corners)
            cv2.putText(
                overlay,
                f"captures={len(captures)} pattern={args.cols}x{args.rows} square={args.square_mm:g}mm",
                (18, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Tankbot Camera Calibration", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and found and corners is not None:
                now = time.monotonic()
                if now - last_save < 0.5:
                    continue
                captures.append(Capture(corners.copy(), frame.copy(), now))
                idx = len(captures)
                cv2.imwrite(str(captures_dir / f"capture_{idx:03d}.png"), frame)
                print(f"Saved capture {idx}")
                last_save = now

            if key == ord("c"):
                if len(captures) < args.min_captures:
                    print(f"Need at least {args.min_captures} captures, have {len(captures)}")
                    continue
                if image_size is None:
                    print("No valid image size yet")
                    continue

                object_points = [objp.copy() for _ in captures]
                image_points = [cap.image_points for cap in captures]
                rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                    object_points,
                    image_points,
                    image_size,
                    None,
                    None,
                )
                _write_calibration_yaml(output, image_size[0], image_size[1], camera_matrix, dist_coeffs)
                _print_result(rms, camera_matrix, dist_coeffs, output)
                return 0

            await asyncio.sleep(0.01)
    finally:
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener
        await stream.close()
        cv2.destroyAllWindows()

    if len(captures) >= args.min_captures and image_size is not None:
        object_points = [objp.copy() for _ in captures]
        image_points = [cap.image_points for cap in captures]
        rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        _write_calibration_yaml(output, image_size[0], image_size[1], camera_matrix, dist_coeffs)
        _print_result(rms, camera_matrix, dist_coeffs, output)
        return 0

    print(f"Not enough captures to calibrate. Need {args.min_captures}, got {len(captures)}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the robot camera from the live video stream")
    parser.add_argument("--robot", default=os.environ.get("ROBOT_URL", "ws://raspberrypi.local:9000"))
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners down")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Checkerboard square size in millimeters")
    parser.add_argument("--min-captures", type=int, default=15, help="Minimum saved views before calibration")
    parser.add_argument("--output", default="calibration/camera_intrinsics.yaml")
    parser.add_argument("--captures-dir", default="calibration/captures")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
