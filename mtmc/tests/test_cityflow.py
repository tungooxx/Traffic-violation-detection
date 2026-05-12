"""
mtmc/tests/test_cityflow.py
---------------------------
Tests for CityFlow-specific innovations:
  - Zone-based transition matrix scoring in GlobalTracker
  - TrackletQualityScorer crop weighting
  - CityFlowAdapter directory parsing
"""

import json
import math
import os
import tempfile
import time

import numpy as np
import pytest

from mtmc.cityflow_adapter import (
    CityFlowAdapter,
    build_default_transition_matrix,
    load_transition_matrix,
    save_transition_matrix,
)
from mtmc.global_tracker import GlobalTracker, TrackletQualityScorer
from mtmc.tracklet import Tracklet


# ─────────────────────────────────────────────────────────────────────────────
# Minimal stub for VehicleReID (avoids loading heavy model weights in tests)
# ─────────────────────────────────────────────────────────────────────────────

class _StubReID:
    """Returns a deterministic embedding based on the tracklet's camera_id."""

    def extract(self, tracklet: Tracklet) -> None:
        seed = abs(hash(tracklet.camera_id)) % 1000
        rng  = np.random.default_rng(seed)
        emb  = rng.random(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        tracklet.reid_embedding = emb

    def extract_from_crops(self, crops, aggregation="mean"):
        """Return per-crop embeddings (all identical for simplicity)."""
        emb = np.ones(512, dtype=np.float32) / math.sqrt(512)
        return np.stack([emb] * len(crops)) if crops else None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)


def _make_tracklet(camera_id: str,
                   entry_time: float,
                   exit_time: float,
                   plate: str = "ABC123",
                   plate_conf: float = 0.9,
                   emb: np.ndarray = None) -> Tracklet:
    t = Tracklet(local_id=1, camera_id=camera_id, entry_time=entry_time)
    t.plate_text       = plate
    t.plate_confidence = plate_conf
    t.exit_time        = exit_time
    dummy_crop         = np.zeros((64, 64, 3), dtype=np.uint8)
    t.crops.append(dummy_crop)
    t.bboxes.append((10, 10, 90, 55))
    if emb is not None:
        t.reid_embedding = emb
    return t


def _make_tracker(transition_matrix=None, camera_positions=None):
    return GlobalTracker(
        reid_model        = _StubReID(),
        camera_positions  = camera_positions or {},
        transition_matrix = transition_matrix or {},
        match_threshold   = 0.30,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Transition Matrix Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionMatrixScoring:

    def test_within_window_scores_positive(self):
        """A vehicle arriving within the allowed time window should score > 0."""
        tm = {
            "cam_A": {"cam_B": {"min_sec": 10.0, "max_sec": 30.0}}
        }
        tracker = _make_tracker(transition_matrix=tm)

        # Shared embedding so Re-ID score is 1.0
        emb = np.ones(512, dtype=np.float32) / math.sqrt(512)

        t1 = _make_tracklet("cam_A", entry_time=0.0, exit_time=5.0,
                             plate="XYZ999", plate_conf=0.95, emb=emb.copy())
        t2 = _make_tracklet("cam_B", entry_time=25.0, exit_time=35.0,
                             plate="XYZ999", plate_conf=0.95, emb=emb.copy())

        gid1 = tracker.associate(t1)
        gid2 = tracker.associate(t2)
        assert gid1 == gid2, "Same vehicle should receive the same global ID"

    def test_outside_window_creates_new_id(self):
        """
        A vehicle arriving outside the time window with an unknown plate
        (so only S_st contributes) should get a new ID because S_st=0 and
        S_reid alone is below the match threshold.
        """
        tm = {
            "cam_A": {"cam_B": {"min_sec": 10.0, "max_sec": 30.0}}
        }
        # Raise match_threshold so that Re-ID alone (without ST) is not enough
        tracker = _make_tracker(transition_matrix=tm)
        tracker.match_threshold = 0.80   # requires strong combined score

        # Use different embeddings (different cameras in StubReID) so Re-ID
        # score is not 1.0 — simulating a realistic scenario
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        emb1 = rng1.random(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = rng2.random(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        t1 = _make_tracklet("cam_A", entry_time=0.0, exit_time=5.0,
                             plate="Unknown", plate_conf=0.1, emb=emb1)
        # Arrives 200 seconds later — way outside the 10-30s window
        t2 = _make_tracklet("cam_B", entry_time=205.0, exit_time=215.0,
                             plate="Unknown", plate_conf=0.1, emb=emb2)

        gid1 = tracker.associate(t1)
        gid2 = tracker.associate(t2)
        assert gid1 != gid2, "Vehicle outside time window should get a new ID"

    def test_impossible_timing_scores_zero(self):
        """A vehicle that arrives before it left the previous camera scores 0."""
        tm = {
            "cam_A": {"cam_B": {"min_sec": 5.0, "max_sec": 60.0}}
        }
        tracker = _make_tracker(transition_matrix=tm)

        # Manually call the internal scorer
        from mtmc.global_tracker import GalleryEntry
        gallery = GalleryEntry(
            global_id=1,
            embedding=np.ones(512, dtype=np.float32) / math.sqrt(512),
            plate_text="ABC",
            plate_conf=0.9,
            last_camera="cam_A",
            last_exit=100.0,
        )
        tracklet = _make_tracklet("cam_B", entry_time=90.0, exit_time=110.0)
        tracklet.reid_embedding = gallery.embedding.copy()

        score = tracker._score_st(tracklet, gallery)
        assert score == 0.0, "Negative delta_t must yield 0.0"

    def test_fallback_to_euclidean_when_no_matrix(self):
        """When no transition matrix entry exists, fall back to Euclidean."""
        tracker = _make_tracker(
            camera_positions={"cam_A": (0.0, 0.0), "cam_B": (140.0, 0.0)},
        )
        from mtmc.global_tracker import GalleryEntry
        gallery = GalleryEntry(
            global_id=1,
            embedding=np.ones(512, dtype=np.float32) / math.sqrt(512),
            plate_text="ABC",
            plate_conf=0.9,
            last_camera="cam_A",
            last_exit=0.0,
        )
        # 140m at 14 m/s → expected 10s; arrive at 10s → perfect score
        tracklet = _make_tracklet("cam_B", entry_time=10.0, exit_time=20.0)
        tracklet.reid_embedding = gallery.embedding.copy()

        score = tracker._score_st(tracklet, gallery)
        assert score > 0.9, f"Expected high ST score, got {score:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tracklet Quality Scorer
# ─────────────────────────────────────────────────────────────────────────────

class TestTrackletQualityScorer:

    def test_weights_sum_to_one(self):
        scorer   = TrackletQualityScorer(frame_width=1280, frame_height=720)
        tracklet = _make_tracklet("cam_A", 0.0, 5.0)
        # Add a few more crops/bboxes
        for _ in range(4):
            tracklet.crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
            tracklet.bboxes.append((100, 100, 200, 150))

        weights = scorer.crop_weights(tracklet)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_larger_bbox_gets_higher_weight(self):
        scorer   = TrackletQualityScorer(frame_width=1280, frame_height=720)
        tracklet = Tracklet(local_id=1, camera_id="cam_A", entry_time=0.0)

        # Crop 0: small box far from centre
        tracklet.crops.append(np.zeros((20, 20, 3), dtype=np.uint8))
        tracklet.bboxes.append((1100, 600, 1120, 620))

        # Crop 1: large box near centre
        tracklet.crops.append(np.zeros((200, 200, 3), dtype=np.uint8))
        tracklet.bboxes.append((540, 260, 740, 460))

        weights = scorer.crop_weights(tracklet)
        assert weights[1] > weights[0], \
            "Larger, centred crop should have higher weight"

    def test_empty_tracklet_returns_empty_weights(self):
        scorer   = TrackletQualityScorer()
        tracklet = Tracklet(local_id=1, camera_id="cam_A", entry_time=0.0)
        weights  = scorer.crop_weights(tracklet)
        assert len(weights) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. CityFlowAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestCityFlowAdapter:

    def _make_mock_dataset(self, tmp_dir: str) -> str:
        """Create a minimal mock CityFlow directory structure."""
        scene_dir = os.path.join(tmp_dir, "scene_001")
        for cam in ("camera_0001", "camera_0002"):
            cam_dir = os.path.join(scene_dir, cam)
            os.makedirs(cam_dir, exist_ok=True)
            # Create a tiny dummy video file (not a valid mp4, but enough for
            # the adapter to detect it)
            open(os.path.join(cam_dir, "video.mp4"), "w").close()
            # Calibration
            with open(os.path.join(cam_dir, "calibration.txt"), "w") as f:
                f.write("1.0 0.0 100.0\n0.0 1.0 200.0\n0.0 0.0 1.0\n")
        return tmp_dir

    def test_scenes_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_mock_dataset(tmp)
            adapter = CityFlowAdapter(tmp)
            assert "scene_001" in adapter.scenes

    def test_camera_configs_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_mock_dataset(tmp)
            adapter = CityFlowAdapter(tmp)
            cams    = adapter.get_camera_configs("scene_001")
            assert len(cams) == 2
            ids = [c["id"] for c in cams]
            assert "scene_001/camera_0001" in ids
            assert "scene_001/camera_0002" in ids

    def test_calibration_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_mock_dataset(tmp)
            adapter = CityFlowAdapter(tmp)
            cams    = adapter.get_camera_configs("scene_001")
            # calibration.txt has tx=100, ty=200
            pos = {c["id"]: c["position"] for c in cams}
            x, y = pos["scene_001/camera_0001"]
            assert abs(x - 100.0) < 1.0
            assert abs(y - 200.0) < 1.0

    def test_transition_matrix_auto_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_mock_dataset(tmp)
            adapter = CityFlowAdapter(tmp)
            tm      = adapter.get_transition_matrix("scene_001")
            assert "scene_001/camera_0001" in tm
            assert "scene_001/camera_0002" in tm["scene_001/camera_0001"]

    def test_transition_matrix_loaded_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_mock_dataset(tmp)
            # Write a custom transition matrix
            custom_tm = {
                "scene_001/camera_0001": {
                    "scene_001/camera_0002": {"min_sec": 7.0, "max_sec": 42.0}
                }
            }
            tm_path = os.path.join(tmp, "scene_001", "transition_matrix.json")
            with open(tm_path, "w") as f:
                json.dump(custom_tm, f)

            adapter = CityFlowAdapter(tmp)
            tm      = adapter.get_transition_matrix("scene_001")
            entry   = tm["scene_001/camera_0001"]["scene_001/camera_0002"]
            assert entry["min_sec"] == 7.0
            assert entry["max_sec"] == 42.0

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            CityFlowAdapter("/nonexistent/path/to/dataset")

    def test_summary_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_mock_dataset(tmp)
            adapter  = CityFlowAdapter(tmp)
            summary  = adapter.summary()
            assert "scene_001" in summary
            assert "2 camera" in summary
