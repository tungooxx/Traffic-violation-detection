"""
process.py
----------
Vehicle image processing pipeline: plate detection and OCR.

Changes from original:
    - `number_plate_extract` now returns a (plate_text, confidence) tuple
      instead of writing only to a file.  The confidence score is the mean
      of the per-character detection confidences, giving downstream modules
      (e.g. GlobalTracker) a reliable signal for dynamic weight adjustment.
    - All original save-to-file behaviour is preserved.
"""

import cv2
import numpy as np
import yolov5
import os
from tool import convert_to_list, rotate_and_crop
from typing import Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Process:
    """
    Handles plate detection and OCR for a single vehicle crop.

    Parameters
    ----------
    save_path : str
        Directory where output images and text files are saved.
    """

    def __init__(self, save_path: str) -> None:
        self.save_path = save_path
        self.id = 1
        self.model_NPP = None   # plate detector
        self.model_NP  = None   # plate OCR

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_number_plate_picture(self, link: str) -> None:
        """Load the model that crops the licence plate region."""
        self.model_NPP = yolov5.load(link)

    def load_number_plate(self, link: str) -> None:
        """Load the model that reads characters from the plate crop."""
        self.model_NP = yolov5.load(link)

    # ── Image saving ──────────────────────────────────────────────────────────

    def save_image(self,
                   img: np.ndarray,
                   id: int,
                   x1: int, y1: int, x2: int, y2: int,
                   image_name: str = None) -> Optional[np.ndarray]:
        """
        Crop and save the vehicle region from the full frame.

        Returns the cropped image, or None on failure.
        """
        self.id = id
        alpha = 50
        if img is None:
            return None

        if image_name == "image":
            new_img = img[max(0, y1 - alpha):y2 + alpha,
                          max(0, x1 - alpha):x2 + alpha]
            des = os.path.join(self.save_path, f"image{self.id}.jpg")
        else:
            new_img = img[y1:y2 * 3, x1:x2]
            des = os.path.join(self.save_path, f"helmet{self.id}.jpg")

        if new_img.size == 0:
            return None
        cv2.imwrite(des, new_img)
        return new_img

    # ── Plate detection ───────────────────────────────────────────────────────

    def plate_detection(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop the licence plate region from a vehicle crop.

        Returns the plate crop image, or None if no plate is found.
        """
        if img is None or img.size == 0:
            return None

        rs = self.model_NPP(img)
        predictions = rs.pred[0]

        for p in predictions:
            x1, y1, x2, y2 = (int(v) for v in p[:4])
            img = img[y1:y2, x1:x2]

        if img is None or img.size == 0:
            print("[Process] plate_detection: empty crop after detection.")
            return None

        des = os.path.join(self.save_path, f"plate_image{int(self.id)}.jpg")
        cv2.imwrite(des, img)
        return img

    # ── OCR ───────────────────────────────────────────────────────────────────

    def number_plate_extract(self,
                             img: Optional[np.ndarray]
                             ) -> Tuple[str, float]:
        """
        Run OCR on a plate crop and return the plate text with a confidence
        score.

        Parameters
        ----------
        img : np.ndarray or None
            Plate crop image (BGR).

        Returns
        -------
        plate_text : str
            Recognised alphanumeric string, or "Unknown" on failure.
        confidence : float
            Mean per-character detection confidence in [0, 1].
            Returns 0.0 when the plate cannot be read.
        """
        plate_text  = "Unknown"
        confidence  = 0.0

        try:
            if img is None or img.size == 0:
                print("[Process] number_plate_extract: empty image.")
                return plate_text, confidence

            img = rotate_and_crop(img)

            class_names = [
                '1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','K',
                'L','M','N','P','S','T','U','V','X','Y','Z','0'
            ]

            if img.shape[0] > 100 and img.shape[1] > 100:
                img_resized = cv2.resize(img, (640, 640))
            else:
                img_resized = cv2.resize(img, (224, 224))

            rs          = self.model_NP(img_resized)
            predictions = rs.pred[0]

            if len(predictions) == 0:
                return plate_text, confidence

            char_list   = []
            confidences = []

            for p in predictions:
                score = float(p[4])
                x1, y1, x2, y2 = (int(v) for v in p[:4])
                name = class_names[int(p[5])]
                char_list.append([name, x1, y1, x2, y2])
                confidences.append(score)

            plate_text = convert_to_list(char_list)
            confidence = float(np.mean(confidences)) if confidences else 0.0

        except Exception as e:
            print(f"[Process] number_plate_extract error: {e}")
            plate_text = "Unknown"
            confidence = 0.0

        # ── Persist to file (original behaviour) ─────────────────────────────
        out_path = os.path.join(self.save_path, f"plate_number{self.id}.txt")
        with open(out_path, "a") as f:
            f.write(f"{plate_text}\n")

        return plate_text, confidence
