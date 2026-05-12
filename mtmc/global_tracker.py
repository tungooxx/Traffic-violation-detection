"""
mtmc/global_tracker.py
-----------------------
Cross-camera vehicle association using weighted fusion of three signals:

    S_total = w_reid · S_reid  +  w_lpr · S_lpr  +  w_st · S_st

where:
    S_reid  — cosine similarity between quality-weighted OSNet Re-ID embeddings
    S_lpr   — normalised Levenshtein similarity between OCR plate strings
    S_st    — zone-based spatio-temporal plausibility score using a
              camera transition matrix (min/max travel time windows)

Innovations for CityFlow / real-world data:
    1. Zone-Based Transition Matrix: replaces simple Euclidean distance with
       a per-camera-pair [min_sec, max_sec] travel time window derived from
       real topology.  Vehicles arriving outside the window score 0.0.
    2. Tracklet Quality Scoring: crops are weighted by bounding box area,
       YOLO detection confidence, and frame-centre proximity before computing
       the Re-ID embedding.  This suppresses occluded / far-away crops.
    3. Adaptive LPR Weighting: unchanged from original design.
    4. Hungarian bipartite matching: unchanged.

Usage:
    from mtmc.global_tracker import GlobalTracker
    from mtmc.reid import VehicleReID

    reid = VehicleReID()
    gt   = GlobalTracker(
        reid_model         = reid,
        camera_positions   = {"cam_0": (0.0, 0.0), "cam_1": (500.0, 0.0)},
        transition_matrix  = {
            "cam_0": {"cam_1": {"min_sec": 10, "max_sec": 60}},
        },
    )

    global_id = gt.associate(tracklet)
    print(gt.get_trajectory(global_id))
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from mtmc.reid import VehicleReID
from mtmc.tracklet import Tracklet

# ── Levenshtein distance ──────────────────────────────────────────────────────
try:
    from Levenshtein import distance as _lev_distance
except ImportError:
    def _lev_distance(a: str, b: str) -> int:
        """Pure-Python fallback for Levenshtein edit distance."""
        if not a:
            return len(b)
        if not b:
            return len(a)
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            new_dp = [i + 1]
            for j, cb in enumerate(b):
                new_dp.append(min(dp[j] + (ca != cb),
                                  dp[j + 1] + 1,
                                  new_dp[-1] + 1))
            dp = new_dp
        return dp[-1]


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryEntry — one camera sighting in a vehicle's global trajectory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryEntry:
    """Records a single camera sighting for a globally tracked vehicle."""
    global_id:        int
    camera_id:        str
    local_track_id:   int
    entry_time:       float
    exit_time:        Optional[float]
    plate_text:       str
    plate_confidence: float
    is_violation:     bool = False

    def to_dict(self) -> dict:
        return {
            "global_id":        self.global_id,
            "camera_id":        self.camera_id,
            "local_track_id":   self.local_track_id,
            "entry_time":       self.entry_time,
            "exit_time":        self.exit_time,
            "plate_text":       self.plate_text,
            "plate_confidence": self.plate_confidence,
            "is_violation":     self.is_violation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GalleryEntry — the accumulated representation of a globally tracked vehicle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    """
    Stores the aggregated Re-ID embedding and metadata for one global vehicle.
    """
    global_id:      int
    embedding:      np.ndarray              # running-average Re-ID vector
    plate_text:     str                     # best plate string seen so far
    plate_conf:     float                   # confidence of best plate
    last_camera:    str                     # camera where vehicle was last seen
    last_exit:      float                   # video timestamp of last camera exit
    last_exit_wall: float = field(default_factory=time.time)  # wall-clock time of last exit
    n_sightings:    int = 1                 # number of camera sightings

    def update_embedding(self, new_emb: np.ndarray) -> None:
        """Exponential moving average of Re-ID embeddings."""
        alpha = 0.3   # weight for the new observation
        self.embedding = alpha * new_emb + (1.0 - alpha) * self.embedding
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding /= norm

    def update_plate(self, text: str, conf: float) -> None:
        """Keep the plate reading with the highest confidence."""
        if conf > self.plate_conf and text not in ("", "Unknown"):
            self.plate_text = text
            self.plate_conf = conf


# ─────────────────────────────────────────────────────────────────────────────
# Tracklet Quality Scorer
# ─────────────────────────────────────────────────────────────────────────────

class TrackletQualityScorer:
    """
    Assigns a quality weight to each crop in a tracklet based on:
      - Bounding box area (larger = more detail)
      - YOLO detection confidence (stored in tracklet.det_confidences)
      - Distance of crop centre from the frame centre (closer = less distortion)

    The resulting weight vector is used to compute a quality-weighted average
    of the per-crop Re-ID embeddings.

    Parameters
    ----------
    frame_width  : int  — expected frame width  (default 1920 for CityFlow)
    frame_height : int  — expected frame height (default 1080 for CityFlow)
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.frame_cx = frame_width  / 2.0
        self.frame_cy = frame_height / 2.0
        self.frame_diag = math.sqrt(frame_width**2 + frame_height**2)

    def crop_weights(self, tracklet: Tracklet) -> np.ndarray:
        """
        Compute a normalised weight vector (shape: [n_crops]) for the tracklet.
        Returns uniform weights if bboxes are unavailable.
        """
        n = len(tracklet.crops)
        if n == 0:
            return np.array([], dtype=np.float32)

        bboxes = tracklet.bboxes[-n:] if len(tracklet.bboxes) >= n else tracklet.bboxes
        confs  = (tracklet.det_confidences[-n:]
                  if hasattr(tracklet, "det_confidences")
                  and len(tracklet.det_confidences) >= n
                  else [1.0] * n)

        weights = np.zeros(n, dtype=np.float32)
        for i, (bbox, conf) in enumerate(zip(bboxes, confs)):
            x1, y1, x2, y2 = bbox
            w = max(x2 - x1, 0)
            h = max(y2 - y1, 0)

            # Area score (normalised by frame area)
            area_score = (w * h) / max(
                self.frame_cx * 2 * self.frame_cy * 2, 1.0)

            # Centre proximity score
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = math.sqrt((cx - self.frame_cx)**2 + (cy - self.frame_cy)**2)
            centre_score = 1.0 - dist / max(self.frame_diag, 1.0)

            # Detection confidence score
            conf_score = float(conf)

            weights[i] = area_score * 0.4 + centre_score * 0.3 + conf_score * 0.3

        total = weights.sum()
        if total < 1e-6:
            return np.ones(n, dtype=np.float32) / n
        return weights / total


# ─────────────────────────────────────────────────────────────────────────────
# GlobalTracker
# ─────────────────────────────────────────────────────────────────────────────

class GlobalTracker:
    """
    Associates finalised Tracklets across cameras into persistent global IDs.

    Parameters
    ----------
    reid_model : VehicleReID
        Initialised Re-ID model.
    camera_positions : dict
        Mapping camera_id → (x, y) in metres.  Used as fallback when no
        transition_matrix is provided.
    transition_matrix : dict, optional
        Zone-based travel time windows:
            {src_cam: {dst_cam: {"min_sec": float, "max_sec": float}}}
        When provided, this replaces the Euclidean distance estimate.
    avg_speed_mps : float
        Fallback average speed (m/s) when no transition matrix entry exists.
    st_sigma : float
        Gaussian sigma (seconds) for the spatio-temporal score.
    lpr_confidence_threshold : float
        OCR confidence above which the plate signal dominates.
    match_threshold : float
        Minimum S_total to link a tracklet to an existing gallery entry.
    gallery_ttl : float
        Seconds before an inactive gallery entry is pruned.
    weights_plate_visible : tuple
        (w_reid, w_lpr, w_st) when plate is clearly visible.
    weights_plate_hidden : tuple
        (w_reid, w_lpr, w_st) when plate is hidden / low confidence.
    frame_width / frame_height : int
        Frame dimensions used by the quality scorer (default: CityFlow 1920×1080).
    """

    def __init__(self,
                 reid_model:               VehicleReID,
                 camera_positions:         Optional[Dict[str, Tuple[float, float]]] = None,
                 transition_matrix:        Optional[Dict] = None,
                 avg_speed_mps:            float = 14.0,
                 st_sigma:                 float = 30.0,
                 lpr_confidence_threshold: float = 0.8,
                 match_threshold:          float = 0.50,
                 gallery_ttl:              float = 3600.0,
                 allow_same_camera_match:  bool = True,
                 weights_plate_visible:    Tuple[float, float, float] = (0.30, 0.50, 0.20),
                 weights_plate_hidden:     Tuple[float, float, float] = (0.70, 0.00, 0.30),
                 frame_width:              int = 1920,
                 frame_height:             int = 1080):

        self.reid_model               = reid_model
        self.camera_positions         = camera_positions or {}
        self.transition_matrix        = transition_matrix or {}
        self.avg_speed_mps            = avg_speed_mps
        self.st_sigma                 = st_sigma
        self.lpr_confidence_threshold = lpr_confidence_threshold
        self.match_threshold          = match_threshold
        self.gallery_ttl              = gallery_ttl
        self.allow_same_camera_match  = allow_same_camera_match
        self.weights_plate_visible    = weights_plate_visible
        self.weights_plate_hidden     = weights_plate_hidden

        self._quality_scorer = TrackletQualityScorer(frame_width, frame_height)
        self._gallery:       Dict[int, GalleryEntry]          = {}
        self._trajectories:  Dict[int, List[TrajectoryEntry]] = {}
        self._next_global_id: int = 1

    # ── Public API ────────────────────────────────────────────────────────────

    def set_transition_matrix(self, matrix: Dict) -> None:
        """
        Replace the transition matrix at runtime.
        Useful when switching between CityFlow scenes.
        """
        self.transition_matrix = matrix

    def associate(self, tracklet: Tracklet) -> int:
        """
        Associate a finalised Tracklet with a global vehicle ID.

        Returns the assigned global ID.
        """
        self._prune_gallery()

        # Extract quality-weighted Re-ID embedding
        if tracklet.reid_embedding is None:
            self._extract_quality_weighted_embedding(tracklet)

        gallery_ids    = list(self._gallery.keys())
        best_global_id = None

        if gallery_ids:
            scores = np.array([
                self._compute_similarity(tracklet, self._gallery[gid])
                for gid in gallery_ids
            ])
            row_ind, col_ind = linear_sum_assignment(-scores.reshape(1, -1))
            best_score = scores[col_ind[0]]
            if best_score >= self.match_threshold:
                best_global_id = gallery_ids[col_ind[0]]

        if best_global_id is None:
            best_global_id = self._create_gallery_entry(tracklet)
        else:
            self._update_gallery_entry(best_global_id, tracklet)

        tracklet.global_id = best_global_id
        self._log_trajectory(tracklet)
        return best_global_id

    def associate_batch(self, tracklets: List[Tracklet]) -> List[int]:
        """
        Associate a batch of tracklets simultaneously using a full cost matrix.
        Prevents two tracklets from matching the same gallery entry.
        """
        self._prune_gallery()

        for t in tracklets:
            if t.reid_embedding is None:
                self._extract_quality_weighted_embedding(t)

        gallery_ids = list(self._gallery.keys())
        n_new = len(tracklets)
        n_gal = len(gallery_ids)

        assigned_ids: List[Optional[int]] = [None] * n_new

        if n_gal > 0:
            cost = np.zeros((n_new, n_gal))
            for i, t in enumerate(tracklets):
                for j, gid in enumerate(gallery_ids):
                    cost[i, j] = self._compute_similarity(t, self._gallery[gid])

            row_ind, col_ind = linear_sum_assignment(-cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= self.match_threshold:
                    assigned_ids[r] = gallery_ids[c]

        for i, t in enumerate(tracklets):
            if assigned_ids[i] is None:
                assigned_ids[i] = self._create_gallery_entry(t)
            else:
                self._update_gallery_entry(assigned_ids[i], t)
            t.global_id = assigned_ids[i]
            self._log_trajectory(t)

        return assigned_ids

    def get_trajectory(self, global_id: int) -> List[TrajectoryEntry]:
        """Return the full list of camera sightings for a global vehicle ID."""
        return list(self._trajectories.get(global_id, []))

    def get_all_trajectories(self) -> Dict[int, List[TrajectoryEntry]]:
        """Return all trajectories keyed by global ID."""
        return {gid: list(entries)
                for gid, entries in self._trajectories.items()}

    def is_violator(self, global_id: int) -> bool:
        """Return True if any sighting of this vehicle was flagged as a violation."""
        return any(
            e.is_violation
            for e in self._trajectories.get(global_id, [])
        )

    # ── Quality-Weighted Embedding Extraction ─────────────────────────────────

    def _extract_quality_weighted_embedding(self, tracklet: Tracklet) -> None:
        """
        Extract per-crop embeddings and combine them using quality weights.
        Falls back to simple mean aggregation if the model supports it directly.
        """
        crops = tracklet.best_crops(n=10)
        if not crops:
            tracklet.reid_embedding = np.zeros(512, dtype=np.float32)
            return

        # Get quality weights for the selected crops
        # We create a temporary tracklet-like view for the scorer
        weights = self._quality_scorer.crop_weights(tracklet)

        # Extract per-crop embeddings
        per_crop_embs = self.reid_model.extract_from_crops(
            crops, aggregation="none"
        )

        if per_crop_embs is None or len(per_crop_embs) == 0:
            # Fallback: use the model's own aggregation
            self.reid_model.extract(tracklet)
            return

        per_crop_embs = np.array(per_crop_embs)  # (n_crops, dim)

        # Align weights to the number of crops returned
        n = min(len(per_crop_embs), len(weights))
        if n == 0:
            tracklet.reid_embedding = np.zeros(512, dtype=np.float32)
            return

        w = weights[:n]
        w = w / (w.sum() + 1e-8)

        emb = (per_crop_embs[:n] * w[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm

        tracklet.reid_embedding = emb.astype(np.float32)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_similarity(self,
                             tracklet: Tracklet,
                             gallery: GalleryEntry) -> float:
        """Compute S_total for a (tracklet, gallery_entry) pair."""
        if (not self.allow_same_camera_match
                and tracklet.camera_id == gallery.last_camera):
            return 0.0

        s_reid = self._score_reid(tracklet, gallery)
        s_lpr  = self._score_lpr(tracklet, gallery)
        s_st   = self._score_st(tracklet, gallery)

        if tracklet.plate_confidence >= self.lpr_confidence_threshold:
            w_reid, w_lpr, w_st = self.weights_plate_visible
        else:
            w_reid, w_lpr, w_st = self.weights_plate_hidden

        return w_reid * s_reid + w_lpr * s_lpr + w_st * s_st

    def _score_reid(self,
                    tracklet: Tracklet,
                    gallery: GalleryEntry) -> float:
        """Cosine similarity between Re-ID embeddings, mapped to [0, 1]."""
        return VehicleReID.cosine_similarity(
            tracklet.reid_embedding, gallery.embedding
        )

    def _score_lpr(self,
                   tracklet: Tracklet,
                   gallery: GalleryEntry) -> float:
        """Normalised Levenshtein similarity between plate strings."""
        a = tracklet.plate_text.strip().upper()
        b = gallery.plate_text.strip().upper()
        if not a or not b or a == "UNKNOWN" or b == "UNKNOWN":
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        return 1.0 - _lev_distance(a, b) / max_len

    def _score_st(self,
                  tracklet: Tracklet,
                  gallery: GalleryEntry) -> float:
        """
        Zone-based spatio-temporal plausibility score.

        Priority:
          1. Use transition_matrix[src_cam][dst_cam] if available.
          2. Fall back to Euclidean distance + avg_speed_mps estimate.
          3. Return 1.0 if no position/timing info is available.
        """
        cam_a = gallery.last_camera
        cam_b = tracklet.camera_id

        if cam_a == cam_b:
            return 0.5   # same camera re-entry is plausible but not certain

        t_exit  = gallery.last_exit
        t_entry = tracklet.entry_time
        delta_t = t_entry - t_exit

        if delta_t < 0:
            return 0.0   # physically impossible

        # ── Zone-based transition matrix ──────────────────────────────────────
        tm_src = self.transition_matrix.get(cam_a, {})
        tm_dst = tm_src.get(cam_b)

        if tm_dst is not None:
            min_sec = float(tm_dst.get("min_sec", 0.0))
            max_sec = float(tm_dst.get("max_sec", float("inf")))

            if delta_t < min_sec or delta_t > max_sec:
                return 0.0   # outside the allowed time window

            # Score peaks at the midpoint of the window
            mid = (min_sec + max_sec) / 2.0
            diff = delta_t - mid
            sigma = max((max_sec - min_sec) / 4.0, 1.0)
            return float(math.exp(-(diff ** 2) / (2.0 * sigma ** 2)))

        # ── Euclidean distance fallback ───────────────────────────────────────
        pos_a = self.camera_positions.get(cam_a)
        pos_b = self.camera_positions.get(cam_b)

        if pos_a is None or pos_b is None:
            return 1.0   # no info → no penalty

        distance_m = math.sqrt(
            (pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2
        )
        expected_travel_s = distance_m / max(self.avg_speed_mps, 0.1)
        diff = delta_t - expected_travel_s
        return float(math.exp(-(diff ** 2) / (2.0 * self.st_sigma ** 2)))

    # ── Gallery management ────────────────────────────────────────────────────

    def _create_gallery_entry(self, tracklet: Tracklet) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        emb = (tracklet.reid_embedding
               if tracklet.reid_embedding is not None
               else np.zeros(512, dtype=np.float32))
        self._gallery[gid] = GalleryEntry(
            global_id      = gid,
            embedding      = emb.copy(),
            plate_text     = tracklet.plate_text,
            plate_conf     = tracklet.plate_confidence,
            last_camera    = tracklet.camera_id,
            last_exit      = tracklet.exit_time or time.time(),
            last_exit_wall = time.time(),
        )
        self._trajectories[gid] = []
        return gid

    def _update_gallery_entry(self,
                               global_id: int,
                               tracklet: Tracklet) -> None:
        entry = self._gallery[global_id]
        if tracklet.reid_embedding is not None:
            entry.update_embedding(tracklet.reid_embedding)
        entry.update_plate(tracklet.plate_text, tracklet.plate_confidence)
        entry.last_camera    = tracklet.camera_id
        entry.last_exit      = tracklet.exit_time or time.time()
        entry.last_exit_wall = time.time()
        entry.n_sightings   += 1

    def _log_trajectory(self, tracklet: Tracklet) -> None:
        gid = tracklet.global_id
        if gid not in self._trajectories:
            self._trajectories[gid] = []
        self._trajectories[gid].append(TrajectoryEntry(
            global_id        = gid,
            camera_id        = tracklet.camera_id,
            local_track_id   = tracklet.local_id,
            entry_time       = tracklet.entry_time,
            exit_time        = tracklet.exit_time,
            plate_text       = tracklet.plate_text,
            plate_confidence = tracklet.plate_confidence,
        ))

    def _prune_gallery(self) -> None:
        """
        Remove gallery entries inactive for more than gallery_ttl seconds.

        Uses wall-clock time (time.time()) for pruning.  Gallery entries store
        last_exit as a wall-clock timestamp so that pruning works correctly
        regardless of whether the pipeline is running in real-time or replaying
        recorded video.

        NOTE: When replaying video, exit_time values on Tracklets are video
        timestamps (seconds from video start), NOT wall-clock time.  The
        _create_gallery_entry and _update_gallery_entry methods therefore
        record time.time() as last_exit_wall alongside the video timestamp,
        and pruning uses last_exit_wall.
        """
        now     = time.time()
        to_drop = [
            gid for gid, entry in self._gallery.items()
            if (now - entry.last_exit_wall) > self.gallery_ttl
        ]
        for gid in to_drop:
            del self._gallery[gid]
