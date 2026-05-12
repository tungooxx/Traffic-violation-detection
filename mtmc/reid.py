"""
mtmc/reid.py
------------
Vehicle Re-Identification (Re-ID) module using OSNet via torchreid.

VehicleReID extracts a compact, discriminative feature embedding from a
list of vehicle image crops (a Tracklet). The embedding can then be compared
against embeddings from other cameras using cosine similarity to determine
whether two tracklets belong to the same physical vehicle.

Model: OSNet-x1.0 pre-trained on VehicleID / VeRi-776
    - OSNet (Omni-Scale Network) captures features at multiple scales
      simultaneously, making it robust to viewpoint and lighting changes.
    - Pre-trained weights are downloaded automatically on first use via
      torchreid's model zoo, or can be supplied manually.

Usage:
    from mtmc.reid import VehicleReID
    reid = VehicleReID(weights_path="weights/osnet_x1_0_vehicleid.pth")
    embedding = reid.extract(tracklet)          # np.ndarray, shape (512,)
    sim = reid.cosine_similarity(emb_a, emb_b)  # float in [-1, 1]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── torchreid import ──────────────────────────────────────────────────────────
try:
    import torchreid
    _TORCHREID_OK = True
except ImportError:
    _TORCHREID_OK = False
    warnings.warn(
        "torchreid is not installed. VehicleReID will return zero embeddings.\n"
        "  Install with: pip install torchreid",
        stacklevel=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image pre-processing transform (matches OSNet training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
_REID_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),          # standard Re-ID input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet statistics
        std =[0.229, 0.224, 0.225],
    ),
])

EMBEDDING_DIM = 512     # OSNet-x1.0 output dimension


# ─────────────────────────────────────────────────────────────────────────────
# VehicleReID
# ─────────────────────────────────────────────────────────────────────────────

class VehicleReID:
    """
    Extracts L2-normalised feature embeddings from vehicle image crops.

    Parameters
    ----------
    weights_path : str or None
        Path to a local `.pth` weights file.  If None (default), torchreid
        will attempt to download the model from its model zoo.
    model_name : str
        torchreid model architecture name (default: "osnet_x1_0").
    device : str
        Torch device string, e.g. "cpu" or "cuda:0".
    """

    def __init__(self,
                 weights_path: Optional[str] = None,
                 model_name: str = "osnet_x1_0",
                 device: str = "cpu"):

        self.device      = torch.device(device)
        self.model_name  = model_name
        self._model: Optional[torch.nn.Module] = None

        if _TORCHREID_OK:
            self._load_model(weights_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, weights_path: Optional[str]) -> None:
        """Build and load the OSNet model."""
        self._model = torchreid.models.build_model(
            name       = self.model_name,
            num_classes= 1000,          # placeholder; we only use the backbone
            pretrained = (weights_path is None),
        )
        if weights_path and Path(weights_path).exists():
            torchreid.utils.load_pretrained_weights(self._model, weights_path)
            print(f"[ReID] Loaded weights from '{weights_path}'")
        elif weights_path:
            warnings.warn(
                f"[ReID] Weights file '{weights_path}' not found. "
                "Using ImageNet-pretrained backbone instead.",
                stacklevel=2,
            )
        self._model.eval()
        self._model.to(self.device)
        print(f"[ReID] {self.model_name} ready on {self.device}")

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR crops to a batched tensor."""
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(_REID_TRANSFORM(rgb))
        if not tensors:
            return torch.zeros(1, 3, 256, 128)
        return torch.stack(tensors)     # (N, 3, H, W)

    @torch.no_grad()
    def _forward(self, tensor: torch.Tensor) -> np.ndarray:
        """Run the model and return an L2-normalised embedding per crop."""
        tensor = tensor.to(self.device)
        features = self._model(tensor)          # (N, EMBEDDING_DIM)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()

    def extract_from_crops(self,
                           crops: List[np.ndarray],
                           aggregation: str = "mean") -> np.ndarray:
        """
        Extract a single representative embedding from a list of image crops.

        Parameters
        ----------
        crops : List[np.ndarray]
            BGR vehicle image crops.
        aggregation : str
            How to combine per-crop embeddings: "mean" (default) or "max".

        Returns
        -------
        np.ndarray, shape (EMBEDDING_DIM,)
            L2-normalised aggregate embedding.
        """
        if not _TORCHREID_OK or self._model is None or not crops:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        tensor   = self._preprocess(crops)
        per_crop = self._forward(tensor)        # (N, EMBEDDING_DIM)

        if aggregation == "none":
            return per_crop.astype(np.float32)
        if aggregation == "max":
            embedding = per_crop.max(axis=0)
        else:
            embedding = per_crop.mean(axis=0)

        # Re-normalise the aggregate
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def extract(self, tracklet, n_best_crops: int = 5) -> np.ndarray:
        """
        Convenience method: extract embedding directly from a Tracklet object.

        Parameters
        ----------
        tracklet : Tracklet
            The tracklet whose best crops will be used.
        n_best_crops : int
            Number of crops to sample from the tracklet.

        Returns
        -------
        np.ndarray, shape (EMBEDDING_DIM,)
        """
        crops = tracklet.best_crops(n=n_best_crops)
        embedding = self.extract_from_crops(crops)
        tracklet.reid_embedding = embedding     # cache on the tracklet
        return embedding

    # ── Similarity ────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two L2-normalised embeddings.

        Returns a float in [0, 1] where 1.0 means identical.
        (Clipped from [-1, 1] to [0, 1] for use as a probability-like score.)
        """
        if emb_a is None or emb_b is None:
            return 0.0
        dot = float(np.dot(emb_a, emb_b))
        # Both embeddings are already L2-normalised, so dot == cosine similarity
        return float(np.clip((dot + 1.0) / 2.0, 0.0, 1.0))
