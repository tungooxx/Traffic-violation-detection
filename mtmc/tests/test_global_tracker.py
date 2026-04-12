"""
mtmc/tests/test_global_tracker.py
----------------------------------
Unit tests for the GlobalTracker weighted fusion and association logic.
"""

import time
import unittest

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mtmc.tracklet import Tracklet
from mtmc.reid import VehicleReID, EMBEDDING_DIM
from mtmc.global_tracker import GlobalTracker


# ── Stub Re-ID model (no GPU / model weights needed for tests) ────────────────

class StubReID(VehicleReID):
    """Returns a fixed embedding for each call, bypassing the real model."""

    def __init__(self):
        # Skip parent __init__ (avoids loading torchreid)
        self.device     = "cpu"
        self.model_name = "stub"
        self._model     = None

    def extract(self, tracklet, n_best_crops=5):
        # Return the embedding already stored on the tracklet, or a default
        if tracklet.reid_embedding is None:
            tracklet.reid_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
            tracklet.reid_embedding /= np.linalg.norm(tracklet.reid_embedding)
        return tracklet.reid_embedding

    def extract_from_crops(self, crops, aggregation="mean"):
        emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
        return emb / np.linalg.norm(emb)


def _make_tracklet(cam_id, local_id, plate="AB1234", plate_conf=0.9,
                   entry_offset=0.0, exit_offset=5.0,
                   embedding=None):
    """Helper: create a finalised Tracklet with controlled attributes."""
    now = time.time()
    t = Tracklet(local_id=local_id, camera_id=cam_id)
    t.entry_time       = now + entry_offset
    t.exit_time        = now + exit_offset
    t.plate_text       = plate
    t.plate_confidence = plate_conf
    if embedding is not None:
        t.reid_embedding = embedding.copy()
    return t


class TestGlobalTrackerAssociation(unittest.TestCase):

    def setUp(self):
        self.reid = StubReID()
        self.gt = GlobalTracker(
            reid_model       = self.reid,
            camera_positions = {
                "cam_0": (0.0,   0.0),
                "cam_1": (200.0, 0.0),
            },
            match_threshold          = 0.40,
            lpr_confidence_threshold = 0.80,
            avg_speed_mps            = 14.0,
            st_sigma                 = 30.0,
        )

    def test_new_vehicle_gets_global_id(self):
        t = _make_tracklet("cam_0", 1)
        gid = self.gt.associate(t)
        self.assertIsNotNone(gid)
        self.assertIsInstance(gid, int)
        self.assertEqual(t.global_id, gid)

    def test_same_vehicle_two_cameras_same_id(self):
        """Two tracklets with identical plate + embedding → same global ID."""
        emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
        emb /= np.linalg.norm(emb)

        t1 = _make_tracklet("cam_0", 1, plate="XY5678", plate_conf=0.95,
                             entry_offset=0.0, exit_offset=5.0, embedding=emb)
        t2 = _make_tracklet("cam_1", 2, plate="XY5678", plate_conf=0.95,
                             entry_offset=20.0, exit_offset=25.0, embedding=emb)

        gid1 = self.gt.associate(t1)
        gid2 = self.gt.associate(t2)
        self.assertEqual(gid1, gid2,
                         "Same vehicle should receive the same global ID.")

    def test_different_vehicles_different_ids(self):
        """Two tracklets with very different embeddings → different global IDs."""
        emb_a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_a[0] = 1.0

        emb_b = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_b[-1] = 1.0

        t1 = _make_tracklet("cam_0", 1, plate="AA0001", plate_conf=0.9,
                             embedding=emb_a)
        t2 = _make_tracklet("cam_0", 2, plate="BB9999", plate_conf=0.9,
                             embedding=emb_b)

        gid1 = self.gt.associate(t1)
        gid2 = self.gt.associate(t2)
        self.assertNotEqual(gid1, gid2)

    def test_trajectory_logged(self):
        t = _make_tracklet("cam_0", 1, plate="TT1111")
        gid = self.gt.associate(t)
        traj = self.gt.get_trajectory(gid)
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj[0].camera_id, "cam_0")
        self.assertEqual(traj[0].plate_text, "TT1111")

    def test_trajectory_grows_across_cameras(self):
        emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
        emb /= np.linalg.norm(emb)

        t1 = _make_tracklet("cam_0", 1, plate="CC2222", plate_conf=0.95,
                             entry_offset=0.0, exit_offset=5.0, embedding=emb)
        t2 = _make_tracklet("cam_1", 2, plate="CC2222", plate_conf=0.95,
                             entry_offset=20.0, exit_offset=25.0, embedding=emb)

        gid1 = self.gt.associate(t1)
        gid2 = self.gt.associate(t2)
        self.assertEqual(gid1, gid2)

        traj = self.gt.get_trajectory(gid1)
        self.assertEqual(len(traj), 2)
        cameras = [e.camera_id for e in traj]
        self.assertIn("cam_0", cameras)
        self.assertIn("cam_1", cameras)

    def test_batch_association_no_duplicates(self):
        """associate_batch should not assign the same gallery entry to two tracklets."""
        emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
        emb /= np.linalg.norm(emb)

        t1 = _make_tracklet("cam_0", 1, plate="DD3333", plate_conf=0.95,
                             embedding=emb)
        gid_seed = self.gt.associate(t1)

        # Two new tracklets that both look like t1
        t2 = _make_tracklet("cam_1", 2, plate="DD3333", plate_conf=0.95,
                             entry_offset=20.0, exit_offset=25.0, embedding=emb)
        t3 = _make_tracklet("cam_1", 3, plate="DD3333", plate_conf=0.95,
                             entry_offset=20.0, exit_offset=25.0, embedding=emb)

        ids = self.gt.associate_batch([t2, t3])
        # They cannot both match the same gallery entry
        self.assertNotEqual(ids[0], ids[1],
                            "Batch association must not assign the same ID twice.")


class TestScoringFunctions(unittest.TestCase):

    def setUp(self):
        self.reid = StubReID()
        self.gt = GlobalTracker(
            reid_model       = self.reid,
            camera_positions = {
                "cam_0": (0.0,   0.0),
                "cam_1": (140.0, 0.0),   # 140 m ≈ 10 s at 14 m/s
            },
            avg_speed_mps = 14.0,
            st_sigma      = 30.0,
        )

    def test_lpr_score_identical_plates(self):
        from mtmc.global_tracker import GalleryEntry
        import numpy as np

        t = _make_tracklet("cam_1", 1, plate="AB1234", plate_conf=0.9)
        t.reid_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        t.reid_embedding /= np.linalg.norm(t.reid_embedding)

        gallery = GalleryEntry(
            global_id   = 1,
            embedding   = t.reid_embedding.copy(),
            plate_text  = "AB1234",
            plate_conf  = 0.9,
            last_camera = "cam_0",
            last_exit   = time.time() - 10,
        )
        score = self.gt._score_lpr(t, gallery)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_lpr_score_completely_different(self):
        from mtmc.global_tracker import GalleryEntry
        import numpy as np

        t = _make_tracklet("cam_1", 1, plate="AB1234", plate_conf=0.9)
        t.reid_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        t.reid_embedding /= np.linalg.norm(t.reid_embedding)

        gallery = GalleryEntry(
            global_id   = 1,
            embedding   = t.reid_embedding.copy(),
            plate_text  = "ZZ9999",
            plate_conf  = 0.9,
            last_camera = "cam_0",
            last_exit   = time.time() - 10,
        )
        score = self.gt._score_lpr(t, gallery)
        self.assertLess(score, 0.5)

    def test_st_score_plausible_timing(self):
        """A vehicle that arrives ~10 s after leaving cam_0 should score near 1.0."""
        from mtmc.global_tracker import GalleryEntry
        import numpy as np

        now = time.time()
        t = Tracklet(local_id=1, camera_id="cam_1")
        t.entry_time       = now + 10.0   # arrived 10 s later
        t.exit_time        = now + 15.0
        t.plate_text       = "EE5555"
        t.plate_confidence = 0.9
        t.reid_embedding   = np.ones(EMBEDDING_DIM, dtype=np.float32)
        t.reid_embedding  /= np.linalg.norm(t.reid_embedding)

        gallery = GalleryEntry(
            global_id   = 1,
            embedding   = t.reid_embedding.copy(),
            plate_text  = "EE5555",
            plate_conf  = 0.9,
            last_camera = "cam_0",
            last_exit   = now,
        )
        score = self.gt._score_st(t, gallery)
        self.assertGreater(score, 0.8)

    def test_st_score_impossible_timing(self):
        """A vehicle that arrives before it left cam_0 should score 0.0."""
        from mtmc.global_tracker import GalleryEntry
        import numpy as np

        now = time.time()
        t = Tracklet(local_id=1, camera_id="cam_1")
        t.entry_time       = now - 5.0    # arrived BEFORE it left cam_0
        t.exit_time        = now
        t.plate_text       = "FF6666"
        t.plate_confidence = 0.9
        t.reid_embedding   = np.ones(EMBEDDING_DIM, dtype=np.float32)
        t.reid_embedding  /= np.linalg.norm(t.reid_embedding)

        gallery = GalleryEntry(
            global_id   = 1,
            embedding   = t.reid_embedding.copy(),
            plate_text  = "FF6666",
            plate_conf  = 0.9,
            last_camera = "cam_0",
            last_exit   = now,
        )
        score = self.gt._score_st(t, gallery)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
