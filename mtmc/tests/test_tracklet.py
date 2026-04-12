"""
mtmc/tests/test_tracklet.py
---------------------------
Unit tests for the Tracklet and SingleCameraTracker classes.
"""

import time
import unittest

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mtmc.tracklet import Tracklet


class TestTracklet(unittest.TestCase):

    def _make_frame(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_initial_state(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        self.assertEqual(t.local_id, 1)
        self.assertEqual(t.camera_id, "cam_0")
        self.assertIsNone(t.exit_time)
        self.assertTrue(t.is_active())
        self.assertEqual(len(t.crops), 0)

    def test_update_adds_crop(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        frame = self._make_frame()
        t.update(frame, (10, 10, 100, 100))
        self.assertEqual(len(t.crops), 1)
        self.assertEqual(len(t.bboxes), 1)

    def test_update_degenerate_box_ignored(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        frame = self._make_frame()
        t.update(frame, (100, 100, 50, 50))   # x2 < x1
        self.assertEqual(len(t.crops), 0)

    def test_max_crops_cap(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        frame = self._make_frame()
        for _ in range(40):
            t.update(frame, (10, 10, 100, 100), max_crops=30)
        self.assertEqual(len(t.crops), 30)

    def test_finalise(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        self.assertIsNone(t.exit_time)
        t.finalise()
        self.assertIsNotNone(t.exit_time)
        self.assertFalse(t.is_active())

    def test_best_crops_sampling(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        frame = self._make_frame()
        for _ in range(20):
            t.update(frame, (10, 10, 100, 100))
        best = t.best_crops(n=5)
        self.assertEqual(len(best), 5)

    def test_best_crops_fewer_than_n(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        frame = self._make_frame()
        for _ in range(3):
            t.update(frame, (10, 10, 100, 100))
        best = t.best_crops(n=5)
        self.assertEqual(len(best), 3)

    def test_duration(self):
        t = Tracklet(local_id=1, camera_id="cam_0")
        time.sleep(0.05)
        t.finalise()
        self.assertGreater(t.duration(), 0.0)

    def test_repr(self):
        t = Tracklet(local_id=7, camera_id="cam_1")
        r = repr(t)
        self.assertIn("local_id=7", r)
        self.assertIn("cam_1", r)


if __name__ == "__main__":
    unittest.main()
