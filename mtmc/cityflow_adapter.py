"""
mtmc/cityflow_adapter.py
------------------------
Adapter for the CityFlow / AI City Challenge dataset format.

CityFlow directory structure expected:
    <dataset_root>/
        scene_001/
            camera_0001/
                video.mp4          (or a sequence of frames: 00001.jpg, ...)
                calibration.txt    (optional)
                roi.jpg            (optional region-of-interest mask)
            camera_0002/
                ...
        scene_002/
            ...
        transition_matrix.json     (optional; auto-generated if missing)

transition_matrix.json format:
    {
        "camera_0001": {
            "camera_0002": {"min_sec": 8,  "max_sec": 45},
            "camera_0003": {"min_sec": 20, "max_sec": 90}
        },
        ...
    }

Usage:
    from mtmc.cityflow_adapter import CityFlowAdapter

    adapter = CityFlowAdapter("path/to/dataset_root")
    for scene in adapter.scenes:
        camera_cfgs = adapter.get_camera_configs(scene)
        transition  = adapter.get_transition_matrix(scene)
        # Pass camera_cfgs to CameraWorker threads
        # Pass transition to GlobalTracker
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Transition matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TRANSITION_WINDOW = {"min_sec": 5.0, "max_sec": 300.0}


def build_default_transition_matrix(camera_ids: List[str]) -> Dict:
    """
    Build a permissive all-pairs transition matrix when no calibration data
    is available.  Every camera pair is assigned the DEFAULT_TRANSITION_WINDOW.
    """
    matrix: Dict[str, Dict] = {}
    for src in camera_ids:
        matrix[src] = {}
        for dst in camera_ids:
            if src != dst:
                matrix[src][dst] = dict(DEFAULT_TRANSITION_WINDOW)
    return matrix


def load_transition_matrix(path: str) -> Dict:
    """Load a transition matrix from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_transition_matrix(matrix: Dict, path: str) -> None:
    """Persist a transition matrix to a JSON file."""
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Camera config builder
# ─────────────────────────────────────────────────────────────────────────────

def _find_video(camera_dir: Path) -> Optional[str]:
    """
    Find the video source for a camera directory.
    Accepts: video.mp4, video.avi, video.mov, or a frame sequence (*.jpg).
    """
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = list(camera_dir.glob(ext))
        if matches:
            return str(matches[0])

    # Frame sequence: return the directory path so cv2.VideoCapture can
    # use glob pattern (caller must handle this case)
    jpgs = sorted(camera_dir.glob("*.jpg"))
    if jpgs:
        return str(camera_dir / "%05d.jpg")   # printf-style for cv2

    return None


def _find_mask(camera_dir: Path) -> Optional[str]:
    """Look for an ROI mask image in the camera directory."""
    for name in ("roi.jpg", "roi.png", "mask.jpg", "mask.png"):
        p = camera_dir / name
        if p.exists():
            return str(p)
    return None


def _parse_calibration(calib_path: Path) -> Tuple[float, float]:
    """
    Parse a CityFlow-style calibration.txt to extract the approximate
    GPS / world position of the camera.

    CityFlow calibration.txt format (simplified):
        Homography matrix (3x3) mapping image pixels to ground plane (metres).
        We extract the translation column (last column) as the camera position.

    Returns (x_metres, y_metres).  Falls back to (0.0, 0.0) on parse failure.
    """
    try:
        with open(calib_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Expect 3 lines of 3 space-separated floats (homography matrix)
        H = []
        for line in lines[:3]:
            row = [float(v) for v in line.split()]
            H.append(row)

        # Translation is H[0][2], H[1][2] (in homogeneous coordinates)
        x = H[0][2] / H[2][2] if H[2][2] != 0 else H[0][2]
        y = H[1][2] / H[2][2] if H[2][2] != 0 else H[1][2]
        return float(x), float(y)
    except Exception:
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CityFlowAdapter
# ─────────────────────────────────────────────────────────────────────────────

class CityFlowAdapter:
    """
    Reads a CityFlow-formatted dataset directory and exposes camera
    configurations and transition matrices compatible with the MTMC pipeline.

    Parameters
    ----------
    dataset_root : str
        Path to the root directory of the CityFlow dataset.
    default_line : list[int]
        Default violation trigger line [x1, y1, x2, y2] used when no
        per-camera line is configured.  Defaults to a horizontal mid-frame
        line at y=400.
    """

    def __init__(self,
                 dataset_root: str,
                 default_line: Optional[List[int]] = None):
        self.root = Path(dataset_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {dataset_root}")

        self.default_line = default_line or [0, 400, 1920, 400]
        self._scenes: Optional[List[str]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def scenes(self) -> List[str]:
        """Return a sorted list of scene names found in the dataset root."""
        if self._scenes is None:
            self._scenes = sorted(
                d.name for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        return self._scenes

    def get_camera_configs(self, scene: str) -> List[dict]:
        """
        Build a list of camera config dicts for a given scene.
        Each dict is compatible with the `camera_cfgs` format used by
        `CameraWorker` and `yolo-detect.py`.

        Returns
        -------
        List of dicts with keys:
            id, video, mask, line, position
        """
        scene_dir = self.root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene not found: {scene_dir}")

        configs = []
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir():
                continue

            cam_id = f"{scene}/{cam_dir.name}"
            video  = _find_video(cam_dir)
            if video is None:
                # No video found — skip this directory
                continue

            mask = _find_mask(cam_dir)

            # Camera position from calibration file
            calib = cam_dir / "calibration.txt"
            position = _parse_calibration(calib) if calib.exists() else (0.0, 0.0)

            configs.append({
                "id":       cam_id,
                "video":    video,
                "mask":     mask,
                "line":     list(self.default_line),
                "position": position,
            })

        return configs

    def get_transition_matrix(self, scene: str) -> Dict:
        """
        Load or auto-generate the transition matrix for a scene.

        Looks for `<scene>/transition_matrix.json`.  If not found, generates
        a default permissive matrix and saves it for future use.

        Returns
        -------
        Dict mapping source camera ID → {dest camera ID → {min_sec, max_sec}}
        """
        scene_dir = self.root / scene
        matrix_path = scene_dir / "transition_matrix.json"

        if matrix_path.exists():
            return load_transition_matrix(str(matrix_path))

        # Auto-generate from camera list
        cam_ids = [
            f"{scene}/{d.name}"
            for d in sorted(scene_dir.iterdir())
            if d.is_dir()
        ]
        matrix = build_default_transition_matrix(cam_ids)
        save_transition_matrix(matrix, str(matrix_path))
        print(f"[CityFlowAdapter] Auto-generated transition matrix → "
              f"{matrix_path}")
        return matrix

    def get_all_configs(self) -> Dict[str, dict]:
        """
        Convenience method: returns configs for all scenes.

        Returns
        -------
        Dict mapping scene name → {"cameras": [...], "transition": {...}}
        """
        result = {}
        for scene in self.scenes:
            result[scene] = {
                "cameras":    self.get_camera_configs(scene),
                "transition": self.get_transition_matrix(scene),
            }
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [f"CityFlow Dataset: {self.root}",
                 f"  Scenes: {len(self.scenes)}"]
        for scene in self.scenes:
            try:
                cams = self.get_camera_configs(scene)
                lines.append(f"  {scene}: {len(cams)} camera(s)")
            except Exception as e:
                lines.append(f"  {scene}: ERROR ({e})")
        return "\n".join(lines)
