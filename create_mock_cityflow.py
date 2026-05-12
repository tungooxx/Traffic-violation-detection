"""
create_mock_cityflow.py
-----------------------
Generates a minimal mock CityFlow-formatted dataset for testing the MTMC
pipeline without needing the official dataset.

Creates the following structure:
    mock_cityflow/
        scene_001/
            camera_0001/
                video.mp4          (synthetic 30s video with moving coloured boxes)
                calibration.txt    (simple homography matrix)
            camera_0002/
                video.mp4
                calibration.txt
            transition_matrix.json

Each "vehicle" is a coloured rectangle that moves across the frame.
The same vehicle ID appears in camera_0001 first, then in camera_0002
a few seconds later — mimicking a real cross-camera scenario.

Ground truth is saved to:
    mock_cityflow/scene_001/ground_truth.csv
    Columns: frame, camera_id, vehicle_id, x1, y1, x2, y2

Usage:
    python create_mock_cityflow.py --output mock_cityflow --num-vehicles 10
"""

import argparse
import csv
import json
import math
import os
import random
import struct
import zlib

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_color(seed: int) -> tuple:
    rng = random.Random(seed)
    return (rng.randint(30, 255), rng.randint(30, 255), rng.randint(30, 255))


def _draw_vehicle(frame: np.ndarray,
                  x: int, y: int,
                  w: int, h: int,
                  color: tuple,
                  vid: int) -> None:
    """Draw a simple rectangular 'vehicle' with an ID label."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.putText(frame, f"V{vid:02d}",
                (x + 4, y + h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _write_calibration(path: str, tx: float, ty: float) -> None:
    """Write a minimal homography matrix calibration file."""
    H = [
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ]
    with open(path, "w") as f:
        for row in H:
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")


def _write_transition_matrix(path: str,
                               cam_ids: list,
                               min_sec: float = 5.0,
                               max_sec: float = 30.0) -> None:
    matrix = {}
    for src in cam_ids:
        matrix[src] = {}
        for dst in cam_ids:
            if src != dst:
                matrix[src][dst] = {"min_sec": min_sec, "max_sec": max_sec}
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"  Wrote transition matrix → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Video generator
# ─────────────────────────────────────────────────────────────────────────────

class VehicleSpec:
    """Defines one vehicle's appearance and trajectory."""

    def __init__(self, vid: int,
                 cam_entry_frame: int,
                 cam_exit_frame: int,
                 start_x: int, end_x: int,
                 y: int, w: int = 80, h: int = 45):
        self.vid             = vid
        self.cam_entry_frame = cam_entry_frame
        self.cam_exit_frame  = cam_exit_frame
        self.start_x         = start_x
        self.end_x           = end_x
        self.y               = y
        self.w               = w
        self.h               = h
        self.color           = _random_color(vid)

    def position_at(self, frame_idx: int) -> tuple:
        """Return (x, y) at a given frame, or None if not in frame."""
        if frame_idx < self.cam_entry_frame or frame_idx > self.cam_exit_frame:
            return None
        t = (frame_idx - self.cam_entry_frame) / max(
            self.cam_exit_frame - self.cam_entry_frame, 1)
        x = int(self.start_x + t * (self.end_x - self.start_x))
        return (x, self.y)


def generate_camera_video(output_path: str,
                           vehicles: list,
                           n_frames: int = 900,
                           fps: int = 30,
                           width: int = 1280,
                           height: int = 720) -> list:
    """
    Generate a synthetic camera video and return ground truth rows.

    Returns
    -------
    List of (frame_idx, vehicle_id, x1, y1, x2, y2) tuples.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    gt_rows = []
    bg_color = (60, 80, 60)   # dark green road background

    for f in range(n_frames):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Draw road markings
        cv2.line(frame, (0, height // 2), (width, height // 2),
                 (200, 200, 200), 2)

        for v in vehicles:
            pos = v.position_at(f)
            if pos is None:
                continue
            x, y = pos
            _draw_vehicle(frame, x, y, v.w, v.h, v.color, v.vid)
            gt_rows.append((f, v.vid, x, y, x + v.w, y + v.h))

        # Frame counter overlay
        cv2.putText(frame, f"Frame {f:04d}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (220, 220, 220), 1)

        writer.write(frame)

    writer.release()
    return gt_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a mock CityFlow dataset for MTMC testing.")
    parser.add_argument("--output",       default="mock_cityflow",
                        help="Output root directory")
    parser.add_argument("--num-vehicles", type=int, default=10,
                        help="Number of vehicles to simulate")
    parser.add_argument("--fps",          type=int, default=30)
    parser.add_argument("--duration",     type=int, default=30,
                        help="Video duration in seconds")
    args = parser.parse_args()

    fps      = args.fps
    duration = args.duration
    n_frames = fps * duration
    n_veh    = args.num_vehicles

    scene_dir = os.path.join(args.output, "scene_001")
    cam0_dir  = os.path.join(scene_dir, "camera_0001")
    cam1_dir  = os.path.join(scene_dir, "camera_0002")

    for d in (cam0_dir, cam1_dir):
        os.makedirs(d, exist_ok=True)

    print(f"[MockCityFlow] Generating {n_veh} vehicles, "
          f"{duration}s @ {fps}fps …")

    # ── Define vehicle trajectories ───────────────────────────────────────────
    # Camera 0: vehicles travel left-to-right in the first half of the video.
    # Camera 1: the same vehicles appear right-to-left in the second half,
    #           simulating them passing through an intersection and entering
    #           the next camera's field of view.

    cam0_vehicles = []
    cam1_vehicles = []
    all_gt = []   # (camera_id, frame, vehicle_id, x1, y1, x2, y2)

    rng = random.Random(42)
    height = 720
    width  = 1280

    for i in range(n_veh):
        vid = i + 1
        y   = rng.randint(height // 4, 3 * height // 4 - 45)

        # Camera 0: vehicle enters between frame 0 and n_frames//3
        c0_entry = rng.randint(0, n_frames // 3)
        c0_exit  = c0_entry + rng.randint(fps * 3, fps * 8)
        c0_exit  = min(c0_exit, n_frames - 1)

        cam0_vehicles.append(VehicleSpec(
            vid=vid,
            cam_entry_frame=c0_entry,
            cam_exit_frame=c0_exit,
            start_x=-80, end_x=width + 10,
            y=y,
        ))

        # Camera 1: same vehicle appears ~10-20 seconds later
        delay_frames = rng.randint(fps * 10, fps * 20)
        c1_entry = c0_exit + delay_frames
        c1_exit  = c1_entry + rng.randint(fps * 3, fps * 8)
        c1_exit  = min(c1_exit, n_frames - 1)

        if c1_entry < n_frames:
            cam1_vehicles.append(VehicleSpec(
                vid=vid,
                cam_entry_frame=c1_entry,
                cam_exit_frame=c1_exit,
                start_x=width + 10, end_x=-80,   # right-to-left
                y=y,
            ))

    # ── Generate videos ───────────────────────────────────────────────────────
    cam0_video = os.path.join(cam0_dir, "video.mp4")
    cam1_video = os.path.join(cam1_dir, "video.mp4")

    print(f"  Generating camera_0001 video → {cam0_video}")
    gt0 = generate_camera_video(cam0_video, cam0_vehicles,
                                 n_frames=n_frames, fps=fps,
                                 width=width, height=height)

    print(f"  Generating camera_0002 video → {cam1_video}")
    gt1 = generate_camera_video(cam1_video, cam1_vehicles,
                                 n_frames=n_frames, fps=fps,
                                 width=width, height=height)

    # ── Write calibration files ───────────────────────────────────────────────
    _write_calibration(os.path.join(cam0_dir, "calibration.txt"),
                       tx=0.0, ty=0.0)
    _write_calibration(os.path.join(cam1_dir, "calibration.txt"),
                       tx=500.0, ty=0.0)
    print("  Wrote calibration files.")

    # ── Write transition matrix ───────────────────────────────────────────────
    cam_ids = ["scene_001/camera_0001", "scene_001/camera_0002"]
    _write_transition_matrix(
        os.path.join(scene_dir, "transition_matrix.json"),
        cam_ids=cam_ids,
        min_sec=8.0,
        max_sec=25.0,
    )

    # ── Write ground truth CSV ────────────────────────────────────────────────
    gt_path = os.path.join(scene_dir, "ground_truth.csv")
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "frame", "vehicle_id",
                         "x1", "y1", "x2", "y2"])
        for frame, vid, x1, y1, x2, y2 in gt0:
            writer.writerow(["camera_0001", frame, vid, x1, y1, x2, y2])
        for frame, vid, x1, y1, x2, y2 in gt1:
            writer.writerow(["camera_0002", frame, vid, x1, y1, x2, y2])
    print(f"  Wrote ground truth → {gt_path}")

    print(f"\n[MockCityFlow] Done! Dataset at: {os.path.abspath(args.output)}")
    print(f"  Vehicles: {n_veh}")
    print(f"  Scene:    scene_001")
    print(f"  Cameras:  camera_0001, camera_0002")
    print(f"\nTo test the adapter:")
    print(f"  from mtmc.cityflow_adapter import CityFlowAdapter")
    print(f"  adapter = CityFlowAdapter('{args.output}')")
    print(f"  print(adapter.summary())")


if __name__ == "__main__":
    main()
