"""
mtmc/tracklet.py
----------------
Defines the Tracklet data structure and the SingleCameraTracker wrapper.

A Tracklet represents a stable, continuous observation of a single vehicle
within one camera's field of view. It accumulates bounding-box crops and
metadata that are later consumed by the Global Tracking Layer for cross-camera
Re-ID and association.

SingleCameraTracker wraps BoT-SORT (via the boxmot library) and manages the
lifecycle of Tracklet objects: creating them when a new ID appears, updating
them each frame, and finalising them when a vehicle leaves the scene.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── BoT-SORT import guard ─────────────────────────────────────────────────────
try:
    from boxmot import BoTSORT
    _BOTSORT_AVAILABLE = True
except ImportError:
    _BOTSORT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Tracklet
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tracklet:
    """
    Represents a single vehicle's continuous track within one camera.

    Attributes
    ----------
    local_id : int
        The tracker-assigned ID within this camera session.
    camera_id : str
        Identifier of the camera that produced this tracklet (e.g. "cam_0").
    entry_time : float
        Unix timestamp when the vehicle first appeared.
    exit_time : Optional[float]
        Unix timestamp when the vehicle was last seen (None if still active).
    crops : List[np.ndarray]
        BGR image crops of the vehicle, collected across frames.
    bboxes : List[Tuple[int,int,int,int]]
        Corresponding bounding boxes (x1, y1, x2, y2) for each crop.
    plate_text : str
        Best OCR string extracted for this tracklet (empty if not yet read).
    plate_confidence : float
        Confidence score [0, 1] of the plate_text OCR result.
    global_id : Optional[int]
        Assigned by GlobalTracker after cross-camera association.
    reid_embedding : Optional[np.ndarray]
        Feature vector produced by the Re-ID model (set after extraction).
    """

    local_id:          int
    camera_id:         str
    entry_time:        float          = field(default_factory=time.time)
    exit_time:         Optional[float] = None
    crops:             List[np.ndarray] = field(default_factory=list)
    bboxes:            List[Tuple[int, int, int, int]] = field(default_factory=list)
    plate_text:        str            = ""
    plate_confidence:  float          = 0.0
    global_id:         Optional[int]  = None
    reid_embedding:    Optional[np.ndarray] = None

    # ── Convenience helpers ───────────────────────────────────────────────────

    def update(self,
               frame: np.ndarray,
               bbox: Tuple[int, int, int, int],
               max_crops: int = 30) -> None:
        """
        Append a new crop from the current frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR camera frame.
        bbox : (x1, y1, x2, y2)
            Bounding box of the vehicle in this frame.
        max_crops : int
            Maximum number of crops to retain (keeps the most recent ones).
        """
        x1, y1, x2, y2 = bbox
        # Guard against degenerate boxes
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        self.crops.append(crop.copy())
        self.bboxes.append(bbox)
        if len(self.crops) > max_crops:
            self.crops.pop(0)
            self.bboxes.pop(0)

    def finalise(self) -> None:
        """Mark the tracklet as complete (vehicle left the camera view)."""
        self.exit_time = time.time()

    def is_active(self) -> bool:
        """Return True if the vehicle is still being tracked."""
        return self.exit_time is None

    def best_crops(self, n: int = 5) -> List[np.ndarray]:
        """
        Return up to *n* crops that are most suitable for Re-ID inference.
        Selection strategy: evenly spaced across the tracklet to maximise
        temporal diversity and avoid redundancy.
        """
        if not self.crops:
            return []
        if len(self.crops) <= n:
            return self.crops
        indices = np.linspace(0, len(self.crops) - 1, n, dtype=int)
        return [self.crops[i] for i in indices]

    def duration(self) -> float:
        """Return the duration in seconds (0 if not yet finalised)."""
        if self.exit_time is None:
            return time.time() - self.entry_time
        return self.exit_time - self.entry_time

    def __repr__(self) -> str:
        status = "active" if self.is_active() else f"done({self.duration():.1f}s)"
        return (f"Tracklet(local_id={self.local_id}, cam={self.camera_id}, "
                f"global_id={self.global_id}, crops={len(self.crops)}, "
                f"plate='{self.plate_text}'({self.plate_confidence:.2f}), "
                f"status={status})")


# ─────────────────────────────────────────────────────────────────────────────
# SingleCameraTracker
# ─────────────────────────────────────────────────────────────────────────────

class SingleCameraTracker:
    """
    Wraps BoT-SORT to manage Tracklet lifecycles for a single camera stream.

    When a new local ID appears, a Tracklet is created.
    Each frame, active tracklets are updated with new crops.
    When a local ID disappears for more than `max_age` frames, the
    corresponding Tracklet is finalised and added to `finished_tracklets`.

    Falls back to a lightweight SORT-compatible stub if boxmot is not installed.

    Parameters
    ----------
    camera_id : str
        Unique identifier for this camera (e.g. "cam_0").
    reid_weights : str
        Path to the Re-ID model weights used internally by BoT-SORT.
        Pass an empty string to disable BoT-SORT's built-in Re-ID.
    max_age : int
        Number of frames a track can be absent before it is finalised.
    min_hits : int
        Minimum consecutive detections before a track is confirmed.
    iou_threshold : float
        IoU threshold for detection-to-track association.
    device : str
        Torch device string, e.g. "cpu" or "cuda:0".
    """

    def __init__(self,
                 camera_id: str,
                 reid_weights: str = "",
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.5,
                 device: str = "cpu"):

        self.camera_id = camera_id
        self.max_age   = max_age

        if _BOTSORT_AVAILABLE:
            self._tracker = BoTSORT(
                reid_weights   = reid_weights if reid_weights else None,
                device         = device,
                half           = False,
                per_class      = False,
            )
        else:
            # Fallback: use the repository's existing SORT implementation
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from sort import Sort
            self._tracker     = Sort(max_age=max_age,
                                     min_hits=min_hits,
                                     iou_threshold=iou_threshold)
            self._use_fallback = True

        self._use_fallback: bool = not _BOTSORT_AVAILABLE

        # Active tracklets keyed by local track ID
        self._active: Dict[int, Tracklet] = {}
        # Finalised tracklets ready for cross-camera association
        self.finished_tracklets: List[Tracklet] = []
        # Track which IDs were seen in the previous frame
        self._prev_ids: set = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self,
               detections: np.ndarray,
               frame: np.ndarray) -> List[Tracklet]:
        """
        Process one frame of detections and update all active tracklets.

        Parameters
        ----------
        detections : np.ndarray, shape (N, 5)
            Each row: [x1, y1, x2, y2, confidence_score].
        frame : np.ndarray
            Full BGR camera frame (used to crop vehicle images).

        Returns
        -------
        List[Tracklet]
            All currently active tracklets after this update.
        """
        if self._use_fallback:
            tracked = self._tracker.update(detections)   # (N, 5): x1,y1,x2,y2,id
        else:
            # BoT-SORT expects (N,6): x1,y1,x2,y2,conf,class
            dets_botsort = np.hstack([
                detections[:, :5],
                np.zeros((len(detections), 1))   # class = 0 (vehicle)
            ]) if len(detections) > 0 else np.empty((0, 6))
            tracked = self._tracker.update(dets_botsort, frame)

        current_ids: set = set()

        for row in tracked:
            x1, y1, x2, y2, track_id = (
                int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
            )
            current_ids.add(track_id)

            if track_id not in self._active:
                # New vehicle — create a fresh Tracklet
                self._active[track_id] = Tracklet(
                    local_id  = track_id,
                    camera_id = self.camera_id,
                )

            self._active[track_id].update(frame, (x1, y1, x2, y2))

        # Finalise tracklets whose IDs have disappeared
        lost_ids = self._prev_ids - current_ids
        for lost_id in lost_ids:
            if lost_id in self._active:
                t = self._active.pop(lost_id)
                t.finalise()
                self.finished_tracklets.append(t)

        self._prev_ids = current_ids
        return list(self._active.values())

    def flush(self) -> List[Tracklet]:
        """
        Finalise all remaining active tracklets (call at end of video).

        Returns
        -------
        List[Tracklet]
            All tracklets that were still active and are now finalised.
        """
        flushed = []
        for t in list(self._active.values()):
            t.finalise()
            self.finished_tracklets.append(t)
            flushed.append(t)
        self._active.clear()
        return flushed

    def pop_finished(self) -> List[Tracklet]:
        """
        Return and clear the list of finished tracklets.
        Call this regularly to pass completed tracklets to the GlobalTracker.
        """
        done = list(self.finished_tracklets)
        self.finished_tracklets.clear()
        return done
