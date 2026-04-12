"""
mtmc — Multi-Target Multi-Camera tracking package
===================================================
Provides:
    Tracklet              — vehicle observation within a single camera
    SingleCameraTracker   — BoT-SORT wrapper that produces Tracklets
    VehicleReID           — OSNet-based visual Re-ID feature extractor
    GlobalTracker         — cross-camera association with weighted fusion
"""

from mtmc.tracklet import Tracklet, SingleCameraTracker
from mtmc.reid import VehicleReID
from mtmc.global_tracker import GlobalTracker

__all__ = [
    "Tracklet",
    "SingleCameraTracker",
    "VehicleReID",
    "GlobalTracker",
]
