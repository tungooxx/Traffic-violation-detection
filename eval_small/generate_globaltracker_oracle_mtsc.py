from pathlib import Path
import os
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtmc.cityflow_adapter import CityFlowAdapter
from mtmc.global_tracker import GlobalTracker
from mtmc.reid import VehicleReID
from mtmc.tracklet import Tracklet


DATA_ROOT = ROOT / "AICity22_Track1_MTMC_Tracking"
SMALL_ROOT = ROOT / "cityflow_2022_small"
OUT_DIR = ROOT / "eval_small"
GT_PATH = OUT_DIR / "gt_train_c001_c002_f001_120.txt"
PRED_PATH = OUT_DIR / "pred_globaltracker_oracle_mtsc_c001_c002_f001_120.txt"

CAM_NAMES = {1: "c001", 2: "c002"}


def load_video_frames(cam_name: str, max_frame: int):
    path = DATA_ROOT / "train" / "S01" / cam_name / "vdo.avi"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = {}
    frame_id = 1
    while frame_id <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frames[frame_id] = frame
        frame_id += 1
    cap.release()
    return frames


def main():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    os.environ.setdefault("HOME", str(ROOT))
    os.environ.setdefault("USERPROFILE", str(ROOT))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

    grouped = {}
    max_frame = 0
    for line in GT_PATH.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cam_id = int(parts[0])
        source_id = int(parts[1])
        frame_id = int(parts[2])
        x, y, w, h = [int(round(float(v))) for v in parts[3:7]]
        max_frame = max(max_frame, frame_id)
        grouped.setdefault((cam_id, source_id), []).append((frame_id, x, y, w, h))

    frames_by_cam = {
        cam_id: load_video_frames(cam_name, max_frame)
        for cam_id, cam_name in CAM_NAMES.items()
    }

    adapter = CityFlowAdapter(str(SMALL_ROOT))
    scene = adapter.scenes[0]
    configs = adapter.get_camera_configs(scene)
    transition = adapter.get_transition_matrix(scene)
    positions = {cfg["id"]: cfg["position"] for cfg in configs}

    reid = VehicleReID(device="cpu")
    match_threshold = float(os.environ.get("MTMC_MATCH_THRESHOLD", "0.45"))
    tracker = GlobalTracker(
        reid_model=reid,
        camera_positions=positions,
        transition_matrix=transition,
        match_threshold=match_threshold,
        allow_same_camera_match=False,
        gallery_ttl=3600.0,
    )

    tracklets = []
    for (cam_id, source_id), rows in grouped.items():
        rows = sorted(rows)
        cam_name = CAM_NAMES[cam_id]
        tracklet = Tracklet(
            local_id=source_id,
            camera_id=f"S01/{cam_name}",
            entry_time=rows[0][0] / 10.0,
            exit_time=rows[-1][0] / 10.0,
        )
        tracklet.source_camera_num = cam_id
        tracklet.source_rows = rows

        for frame_id, x, y, w, h in rows[:: max(1, len(rows) // 5)][:5]:
            frame = frames_by_cam[cam_id].get(frame_id)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(0, min(width - 1, x + w))
            y2 = max(0, min(height - 1, y + h))
            tracklet.update(frame, (x1, y1, x2, y2))
        tracklets.append(tracklet)

    tracklets.sort(key=lambda t: (t.entry_time, t.camera_id, t.local_id))
    for tracklet in tracklets:
        tracker.associate(tracklet)

    pred_rows = []
    for tracklet in tracklets:
        for frame_id, x, y, w, h in tracklet.source_rows:
            pred_rows.append((
                int(tracklet.source_camera_num),
                int(tracklet.global_id),
                frame_id,
                x,
                y,
                w,
                h,
            ))

    pred_rows.sort()
    with PRED_PATH.open("w") as f:
        for cam_id, global_id, frame_id, x, y, w, h in pred_rows:
            f.write(f"{cam_id} {global_id} {frame_id} {x} {y} {w} {h} -1 -1\n")

    print(f"threshold: {match_threshold}")
    print(f"oracle local tracklets: {len(tracklets)}")
    print(f"global ids: {len(tracker.get_all_trajectories())}")
    print(f"prediction: {PRED_PATH}")


if __name__ == "__main__":
    main()
