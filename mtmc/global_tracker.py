"""
mtmc/global_tracker.py
-----------------------
Cross-camera vehicle association using weighted fusion of three signals:

    S_total = w_reid · S_reid  +  w_lpr · S_lpr  +  w_st · S_st

where:
    S_reid  — cosine similarity between OSNet Re-ID embeddings
    S_lpr   — normalised Levenshtein similarity between OCR plate strings
    S_st    — spatio-temporal plausibility score based on camera positions
              and the time elapsed between tracklet exit / entry

Weights are adjusted dynamically:
    - When LPR confidence is high (≥ lpr_confidence_threshold), the plate
      signal is trusted more (default: w_lpr = 0.50).
    - When the plate is hidden or low-confidence, the system falls back to
      visual appearance (default: w_reid = 0.70).

Association is solved as a bipartite matching problem using the Hungarian
algorithm (scipy.optimize.linear_sum_assignment).

Usage:
    from mtmc.global_tracker import GlobalTracker
    from mtmc.reid import VehicleReID

    reid   = VehicleReID()
    gt     = GlobalTracker(reid_model=reid, camera_positions={
                "cam_0": (0.0,   0.0),
                "cam_1": (500.0, 0.0),
            })

    # After a tracklet is finalised by SingleCameraTracker:
    global_id = gt.associate(tracklet)
    print(f"Vehicle global ID: {global_id}")
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

    def to_dict(self) -> dict:
        return {
            "global_id":        self.global_id,
            "camera_id":        self.camera_id,
            "local_track_id":   self.local_track_id,
            "entry_time":       self.entry_time,
            "exit_time":        self.exit_time,
            "plate_text":       self.plate_text,
            "plate_confidence": self.plate_confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GalleryEntry — the accumulated representation of a globally tracked vehicle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    """
    Stores the aggregated Re-ID embedding and metadata for one global vehicle.
    """
    global_id:   int
    embedding:   np.ndarray              # running-average Re-ID vector
    plate_text:  str                     # best plate string seen so far
    plate_conf:  float                   # confidence of best plate
    last_camera: str                     # camera where vehicle was last seen
    last_exit:   float                   # timestamp of last camera exit
    n_sightings: int = 1                 # number of camera sightings

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
# GlobalTracker
# ─────────────────────────────────────────────────────────────────────────────

class GlobalTracker:
    """
    Associates finalised Tracklets across cameras into persistent global IDs.

    Parameters
    ----------
    reid_model : VehicleReID
        Initialised Re-ID model used to extract embeddings.
    camera_positions : dict
        Mapping of camera_id → (x, y) physical coordinates (metres or pixels).
        Used for spatio-temporal scoring.  Example:
            {"cam_0": (0.0, 0.0), "cam_1": (500.0, 0.0)}
    avg_speed_mps : float
        Expected average vehicle speed in metres per second (default 14 m/s ≈ 50 km/h).
        Used to compute the expected travel time between cameras.
    st_sigma : float
        Standard deviation (seconds) for the Gaussian spatio-temporal score.
        Smaller values make the score more sensitive to timing.
    lpr_confidence_threshold : float
        OCR confidence above which the plate signal is trusted (default 0.8).
    match_threshold : float
        Minimum S_total required to link a tracklet to an existing gallery
        entry (default 0.5).
    gallery_ttl : float
        Seconds after which an inactive gallery entry is pruned (default 3600).
    weights_plate_visible : tuple
        (w_reid, w_lpr, w_st) when plate is clearly visible.
    weights_plate_hidden : tuple
        (w_reid, w_lpr, w_st) when plate is hidden / low confidence.
    """

    def __init__(self,
                 reid_model: VehicleReID,
                 camera_positions: Optional[Dict[str, Tuple[float, float]]] = None,
                 avg_speed_mps: float = 14.0,
                 st_sigma: float = 30.0,
                 lpr_confidence_threshold: float = 0.8,
                 match_threshold: float = 0.50,
                 gallery_ttl: float = 3600.0,
                 weights_plate_visible: Tuple[float, float, float] = (0.30, 0.50, 0.20),
                 weights_plate_hidden:  Tuple[float, float, float] = (0.70, 0.00, 0.30)):

        self.reid_model               = reid_model
        self.camera_positions         = camera_positions or {}
        self.avg_speed_mps            = avg_speed_mps
        self.st_sigma                 = st_sigma
        self.lpr_confidence_threshold = lpr_confidence_threshold
        self.match_threshold          = match_threshold
        self.gallery_ttl              = gallery_ttl
        self.weights_plate_visible    = weights_plate_visible
        self.weights_plate_hidden     = weights_plate_hidden

        self._gallery: Dict[int, GalleryEntry] = {}
        self._trajectories: Dict[int, List[TrajectoryEntry]] = {}
        self._next_global_id: int = 1

    # ── Public API ────────────────────────────────────────────────────────────

    def associate(self, tracklet: Tracklet) -> int:
        """
        Associate a finalised Tracklet with a global vehicle ID.

        Steps:
        1. Extract (or reuse) the Re-ID embedding for the tracklet.
        2. Score the tracklet against every gallery entry.
        3. Run Hungarian matching; assign existing or new global ID.
        4. Update the gallery and trajectory log.

        Parameters
        ----------
        tracklet : Tracklet
            A finalised Tracklet (exit_time must be set).

        Returns
        -------
        int
            The assigned global vehicle ID.
        """
        self._prune_gallery()

        # 1. Ensure embedding is available
        if tracklet.reid_embedding is None:
            self.reid_model.extract(tracklet)

        # 2. Score against gallery
        gallery_ids   = list(self._gallery.keys())
        best_global_id = None

        if gallery_ids:
            scores = np.array([
                self._compute_similarity(tracklet, self._gallery[gid])
                for gid in gallery_ids
            ])  # shape (N,)

            # Hungarian on a single-row cost matrix
            row_ind, col_ind = linear_sum_assignment(-scores.reshape(1, -1))
            best_score = scores[col_ind[0]]
            if best_score >= self.match_threshold:
                best_global_id = gallery_ids[col_ind[0]]

        # 3. Assign global ID
        if best_global_id is None:
            best_global_id = self._create_gallery_entry(tracklet)
        else:
            self._update_gallery_entry(best_global_id, tracklet)

        tracklet.global_id = best_global_id

        # 4. Log trajectory
        self._log_trajectory(tracklet)

        return best_global_id

    def associate_batch(self, tracklets: List[Tracklet]) -> List[int]:
        """
        Associate a batch of tracklets simultaneously using a full cost matrix.

        This is more accurate than calling `associate` one-by-one because it
        prevents two tracklets from matching the same gallery entry.

        Parameters
        ----------
        tracklets : List[Tracklet]
            List of finalised Tracklets.

        Returns
        -------
        List[int]
            Global IDs in the same order as the input tracklets.
        """
        self._prune_gallery()

        # Extract embeddings
        for t in tracklets:
            if t.reid_embedding is None:
                self.reid_model.extract(t)

        gallery_ids = list(self._gallery.keys())
        n_new = len(tracklets)
        n_gal = len(gallery_ids)

        assigned_ids = [None] * n_new

        if n_gal > 0:
            # Build cost matrix (n_new × n_gal)
            cost = np.zeros((n_new, n_gal))
            for i, t in enumerate(tracklets):
                for j, gid in enumerate(gallery_ids):
                    cost[i, j] = self._compute_similarity(t, self._gallery[gid])

            row_ind, col_ind = linear_sum_assignment(-cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= self.match_threshold:
                    assigned_ids[r] = gallery_ids[c]

        # Create new entries for unmatched tracklets
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
            getattr(e, "is_violation", False)
            for e in self._trajectories.get(global_id, [])
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_similarity(self,
                             tracklet: Tracklet,
                             gallery: GalleryEntry) -> float:
        """
        Compute S_total for a (tracklet, gallery_entry) pair.
        """
        s_reid = self._score_reid(tracklet, gallery)
        s_lpr  = self._score_lpr(tracklet, gallery)
        s_st   = self._score_st(tracklet, gallery)

        # Dynamic weight selection
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
        """
        Normalised Levenshtein similarity between plate strings.
        Returns 0.0 if either string is empty or 'Unknown'.
        """
        a = tracklet.plate_text.strip().upper()
        b = gallery.plate_text.strip().upper()
        if not a or not b or a == "UNKNOWN" or b == "UNKNOWN":
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        edit_dist = _lev_distance(a, b)
        return 1.0 - edit_dist / max_len

    def _score_st(self,
                  tracklet: Tracklet,
                  gallery: GalleryEntry) -> float:
        """
        Spatio-temporal plausibility score.

        Models the probability that the vehicle could have travelled from
        `gallery.last_camera` to `tracklet.camera_id` in the observed time.

        Returns 1.0 if camera positions are unknown (no penalty).
        """
        cam_a = gallery.last_camera
        cam_b = tracklet.camera_id

        if cam_a == cam_b:
            # Same camera: high score only if re-entry is plausible (e.g. U-turn)
            return 0.5

        pos_a = self.camera_positions.get(cam_a)
        pos_b = self.camera_positions.get(cam_b)

        if pos_a is None or pos_b is None:
            return 1.0     # no position info → no penalty

        distance_m = math.sqrt(
            (pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2
        )
        expected_travel_s = distance_m / max(self.avg_speed_mps, 0.1)

        t_exit  = gallery.last_exit
        t_entry = tracklet.entry_time
        delta_t = t_entry - t_exit      # seconds between exit and entry

        if delta_t < 0:
            return 0.0     # entered before it left — impossible

        diff = delta_t - expected_travel_s
        score = math.exp(-(diff ** 2) / (2.0 * self.st_sigma ** 2))
        return float(score)

    # ── Gallery management ────────────────────────────────────────────────────

    def _create_gallery_entry(self, tracklet: Tracklet) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        emb = (tracklet.reid_embedding
               if tracklet.reid_embedding is not None
               else np.zeros(512, dtype=np.float32))
        self._gallery[gid] = GalleryEntry(
            global_id   = gid,
            embedding   = emb.copy(),
            plate_text  = tracklet.plate_text,
            plate_conf  = tracklet.plate_confidence,
            last_camera = tracklet.camera_id,
            last_exit   = tracklet.exit_time or time.time(),
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
        entry.last_camera = tracklet.camera_id
        entry.last_exit   = tracklet.exit_time or time.time()
        entry.n_sightings += 1

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
        """Remove gallery entries that have not been seen for gallery_ttl seconds."""
        now     = time.time()
        to_drop = [
            gid for gid, entry in self._gallery.items()
            if (now - entry.last_exit) > self.gallery_ttl
        ]
        for gid in to_drop:
            del self._gallery[gid]
