from pathlib import Path
import os
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtmc.cityflow_adapter import CityFlowAdapter
from mtmc.global_tracker import GlobalTracker
from mtmc.reid import VehicleReID
from mtmc.tracklet import Tracklet


DATA_ROOT = ROOT / "AICity22_Track1_MTMC_Tracking"
SMALL_ROOT = ROOT / "cityflow_2022_small"
OUT_DIR = ROOT / "eval_small"
PRED_PATH = OUT_DIR / "pred_globaltracker_c001_c002_f001_120.txt"

CAMERAS = [(1, "c001"), (2, "c002")]
MAX_FRAME = 120


def load_mtsc_tracks(cam_name: str):
    path = (
        DATA_ROOT
        / "train"
        / "S01"
        / cam_name
        / "mtsc"
        / "mtsc_deepsort_mask_rcnn.txt"
    )
    tracks = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            if frame > MAX_FRAME:
                continue
            local_id = int(float(parts[1]))
            x, y, w, h = [int(round(float(v))) for v in parts[2:6]]
            tracks.setdefault(local_id, []).append((frame, x, y, w, h))
    return tracks


def load_video_frames(cam_name: str):
    path = DATA_ROOT / "train" / "S01" / cam_name / "vdo.avi"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    frames = {}
    frame_id = 1
    while frame_id <= MAX_FRAME:
        ok, frame = cap.read()
        if not ok:
            break
        frames[frame_id] = frame
        frame_id += 1
    cap.release()
    return frames


def make_tracklet(cam_id: int, cam_name: str, local_id: int, rows, frames):
    camera_id = f"S01/{cam_name}"
    rows = sorted(rows, key=lambda row: row[0])
    tracklet = Tracklet(
        local_id=local_id,
        camera_id=camera_id,
        entry_time=rows[0][0] / 10.0,
        exit_time=rows[-1][0] / 10.0,
    )
    tracklet.source_camera_num = cam_id
    tracklet.source_rows = rows

    sample_rows = rows[:: max(1, len(rows) // 5)][:5]
    for frame_id, x, y, w, h in sample_rows:
        frame = frames.get(frame_id)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width - 1, x + w))
        y2 = max(0, min(height - 1, y + h))
        tracklet.update(frame, (x1, y1, x2, y2))
    return tracklet


def main():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    os.environ.setdefault("HOME", str(ROOT))
    os.environ.setdefault("USERPROFILE", str(ROOT))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

    adapter = CityFlowAdapter(str(SMALL_ROOT))
    scene = adapter.scenes[0]
    configs = adapter.get_camera_configs(scene)
    transition = adapter.get_transition_matrix(scene)
    positions = {cfg["id"]: cfg["position"] for cfg in configs}

    reid = VehicleReID(device="cpu")
    tracker = GlobalTracker(
        reid_model=reid,
        camera_positions=positions,
        transition_matrix=transition,
        match_threshold=0.45,
        gallery_ttl=3600.0,
        allow_same_camera_match=False,
    )

    tracklets = []
    for cam_id, cam_name in CAMERAS:
        frames = load_video_frames(cam_name)
        tracks = load_mtsc_tracks(cam_name)
        for local_id, rows in tracks.items():
            if len(rows) < 5:
                continue
            tracklets.append(make_tracklet(cam_id, cam_name, local_id, rows, frames))

    tracklets.sort(key=lambda t: (t.entry_time, t.camera_id, t.local_id))
    for tracklet in tracklets:
        tracker.associate(tracklet)

    pred_rows = []
    for tracklet in tracklets:
        global_id = int(tracklet.global_id)
        cam_id = int(tracklet.source_camera_num)
        for frame_id, x, y, w, h in tracklet.source_rows:
            pred_rows.append((cam_id, global_id, frame_id, x, y, w, h))

    pred_rows.sort()
    with PRED_PATH.open("w") as f:
        for cam_id, global_id, frame_id, x, y, w, h in pred_rows:
            f.write(f"{cam_id} {global_id} {frame_id} {x} {y} {w} {h} -1 -1\n")

    print(f"tracklets: {len(tracklets)}")
    print(f"global ids: {len(tracker.get_all_trajectories())}")
    print(f"prediction: {PRED_PATH}")


if __name__ == "__main__":
    main()
