"""
yolo-detect.py
--------------
Multi-camera traffic violation detection and tracking entry point.

This script:
1. Reads all [camera_N] sections from project.ini and opens each video stream.
2. Runs per-camera YOLOv8 vehicle + helmet detection.
3. Feeds detections into per-camera SingleCameraTracker (BoT-SORT).
4. Passes finalised Tracklets through the LPR pipeline (plate detection + OCR).
5. Sends Tracklets to the GlobalTracker for cross-camera Re-ID association.
6. Persists trajectories to the database and sends violation email alerts.

Single-camera mode: define only [camera_0] in project.ini.
Multi-camera mode:  define [camera_0], [camera_1], … up to any number.

Usage:
    python yolo-detect.py
"""

import os
import sys
import threading
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yolov5
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mtmc.tracklet import SingleCameraTracker, Tracklet
from mtmc.reid import VehicleReID
from mtmc.global_tracker import GlobalTracker
from process import Process
from workWithDatabase import DatabaseConnector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_weights(s: str) -> Tuple[float, float, float]:
    """Parse a comma-separated weight string like '0.30, 0.50, 0.20'."""
    parts = [float(x.strip()) for x in s.split(",")]
    return tuple(parts[:3])


def _load_camera_configs(conf: ConfigParser) -> List[dict]:
    """
    Collect all [camera_N] sections from the config file.
    Returns a list of dicts with keys: id, video, mask, line, position.
    """
    cameras = []
    for section in conf.sections():
        if not section.startswith("camera_"):
            continue
        cam_id = section  # e.g. "camera_0"
        cameras.append({
            "id":       cam_id,
            "video":    conf.get(section, "video"),
            "mask":     conf.get(section, "mask"),
            "line":     [
                conf.getint(section, "linex1"),
                conf.getint(section, "liney1"),
                conf.getint(section, "linex2"),
                conf.getint(section, "liney2"),
            ],
            "position": (
                conf.getfloat(section, "position_x", fallback=0.0),
                conf.getfloat(section, "position_y", fallback=0.0),
            ),
        })
    cameras.sort(key=lambda c: c["id"])
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera processing thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    """
    Runs the detection + single-camera tracking loop for one camera in a
    dedicated thread.  Finished Tracklets are placed in `finished_queue` for
    the main thread to pick up and pass to the GlobalTracker.
    """

    def __init__(self,
                 cam_cfg: dict,
                 model_vehicle: yolov5.models.common.AutoShape,
                 model_helmet: YOLO,
                 processor: Process,
                 finished_queue: list,
                 queue_lock: threading.Lock,
                 show_vehicle: bool,
                 show_tracking: bool,
                 show_helmet: bool):
        super().__init__(daemon=True)
        self.cam_cfg       = cam_cfg
        self.model_vehicle = model_vehicle
        self.model_helmet  = model_helmet
        self.processor     = processor
        self.finished_queue = finished_queue
        self.queue_lock    = queue_lock
        self.show_vehicle  = show_vehicle
        self.show_tracking = show_tracking
        self.show_helmet   = show_helmet
        self.running       = True

    def run(self):
        cam_id  = self.cam_cfg["id"]
        line    = self.cam_cfg["line"]
        alpha   = 20    # pixel tolerance around the trigger line

        cap  = cv2.VideoCapture(self.cam_cfg["video"])
        mask = cv2.imread(self.cam_cfg["mask"])

        tracker = SingleCameraTracker(
            camera_id     = cam_id,
            max_age       = 30,
            min_hits      = 3,
            iou_threshold = 0.5,
        )

        class_names = ["car", "moto", "truck", "bus", "bicycle"]
        track_list  = []   # IDs already processed for violation

        print(f"[{cam_id}] Starting …")

        while self.running:
            success, img = cap.read()
            if not success:
                print(f"[{cam_id}] End of stream.")
                break

            if mask is not None:
                masked_img = cv2.bitwise_and(img, mask)
            else:
                masked_img = img

            # ── Vehicle detection ─────────────────────────────────────────────
            rs          = self.model_vehicle(masked_img)
            predictions = rs.pred[0]
            detections  = np.empty((0, 5))

            for p in predictions:
                score = round(float(p[4]), 2)
                x1, y1, x2, y2 = (int(v) for v in p[:4])
                if self.show_vehicle:
                    name = class_names[min(int(p[5]), len(class_names) - 1)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, f"{name}-{score}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                detections = np.vstack(
                    (detections, np.array([x1, y1, x2, y2, score]))
                )

            # Draw trigger line
            cv2.line(img,
                     (line[0], line[1]), (line[2], line[3]),
                     (255, 0, 0), 2)

            # ── BoT-SORT update ───────────────────────────────────────────────
            active_tracklets = tracker.update(detections, img)

            for t in active_tracklets:
                if not t.bboxes:
                    continue
                x1, y1, x2, y2 = t.bboxes[-1]
                xc = (x1 + x2) // 2
                yc = (y1 + y2) // 2

                if self.show_tracking:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = (f"G:{t.global_id}" if t.global_id
                             else f"L:{t.local_id}")
                    cv2.putText(img, label,
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)

                # ── Trigger line crossing ─────────────────────────────────────
                lx1, ly1, lx2, ly2 = line
                # Horizontal line: check y proximity
                # Vertical line: check x proximity
                if abs(lx1 - lx2) < abs(ly1 - ly2):
                    # Vertical trigger line
                    crossed = lx1 - alpha < xc < lx1 + alpha
                else:
                    # Horizontal trigger line
                    crossed = ly1 - alpha < yc < ly1 + alpha

                if crossed and t.local_id not in track_list:
                    track_list.append(t.local_id)
                    # Run LPR pipeline
                    new_img = self.processor.save_image(
                        img, t.local_id, x1, y1, x2, y2, "image"
                    )
                    if new_img is not None:
                        plate_img = self.processor.plate_detection(new_img)
                        if plate_img is not None:
                            text, conf = self.processor.number_plate_extract(
                                plate_img
                            )
                            t.plate_text       = text
                            t.plate_confidence = conf

            # ── Helmet detection ──────────────────────────────────────────────
            rs2 = self.model_helmet(masked_img)
            for result in rs2:
                boxes      = result.boxes.cpu().numpy()
                conf_arr   = result.boxes.conf.numpy()
                class_arr  = result.boxes.cls.numpy()
                xyxy_arr   = result.boxes.xyxy.numpy()

                for box_xyxy, label_idx, conf_val in zip(
                        xyxy_arr, class_arr, conf_arr):
                    label_idx = int(label_idx)
                    if self.show_helmet:
                        xh1, yh1, xh2, yh2 = box_xyxy.astype(int)
                        cv2.rectangle(img,
                                      (xh1, yh1), (xh2, yh2),
                                      (255, 255, 0), 2)
                        label_name = ["helmet", "no helmet"][label_idx]
                        cv2.putText(img,
                                    f"{label_name}-{conf_val:.2f}",
                                    (xh1, yh1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)

                    # No-helmet violation
                    if label_idx == 1 and conf_val > 0.7:
                        xh1, yh1, xh2, yh2 = box_xyxy.astype(int)
                        # Find the closest active tracklet
                        best_t = None
                        best_d = float("inf")
                        for t in active_tracklets:
                            if not t.bboxes:
                                continue
                            tx1, ty1, tx2, ty2 = t.bboxes[-1]
                            txc = (tx1 + tx2) // 2
                            tyc = (ty1 + ty2) // 2
                            d = abs(txc - (xh1 + xh2) // 2) + abs(
                                tyc - (yh1 + yh2) // 2)
                            if d < best_d:
                                best_d = d
                                best_t = t

                        if best_t and best_t.local_id not in track_list:
                            track_list.append(best_t.local_id)
                            tx1, ty1, tx2, ty2 = best_t.bboxes[-1]
                            new_img = self.processor.save_image(
                                img, best_t.local_id,
                                tx1, ty1, tx2, ty2
                            )
                            if new_img is not None:
                                plate_img = self.processor.plate_detection(
                                    new_img)
                                if plate_img is not None:
                                    text, conf = (
                                        self.processor.number_plate_extract(
                                            plate_img))
                                    best_t.plate_text       = text
                                    best_t.plate_confidence = conf

            # ── Collect finished tracklets ────────────────────────────────────
            done = tracker.pop_finished()
            if done:
                with self.queue_lock:
                    self.finished_queue.extend(done)

            cv2.imshow(f"Camera: {cam_id}", img)
            if cv2.waitKey(5) == 27:   # ESC to quit
                self.running = False
                break

        # Flush remaining tracklets at end of stream
        done = tracker.flush()
        if done:
            with self.queue_lock:
                self.finished_queue.extend(done)

        cap.release()
        cv2.destroyWindow(f"Camera: {cam_id}")
        print(f"[{cam_id}] Worker stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load config ───────────────────────────────────────────────────────────
    conf = ConfigParser()
    conf.read("project.ini")

    model_vehicle_path = conf.get("load_model", "model_vehicle_detect")
    model_helmet_path  = conf.get("load_model", "model_helmet_detect")
    model_plate_path   = conf.get("load_model", "model_plate_detect")
    model_ocr_path     = conf.get("load_model", "model_plate_number_detect")
    model_reid_path    = conf.get("load_model", "model_reid", fallback="")
    save_image_path    = conf.get("destination", "save_image")

    show_vehicle = conf.getboolean("graphic_setting", "show_vehicle_detect")
    show_tracking= conf.getboolean("graphic_setting", "show_tracking")
    show_helmet  = conf.getboolean("graphic_setting", "show_helmet")

    gt_cfg = conf["global_tracker"] if conf.has_section("global_tracker") else {}
    match_threshold          = float(gt_cfg.get("match_threshold",          0.50))
    lpr_conf_threshold       = float(gt_cfg.get("lpr_confidence_threshold", 0.80))
    weights_visible          = _parse_weights(gt_cfg.get("weights_plate_visible",
                                                          "0.30, 0.50, 0.20"))
    weights_hidden           = _parse_weights(gt_cfg.get("weights_plate_hidden",
                                                          "0.70, 0.00, 0.30"))
    avg_speed                = float(gt_cfg.get("avg_speed_mps",  14.0))
    st_sigma                 = float(gt_cfg.get("st_sigma",        30.0))
    gallery_ttl              = float(gt_cfg.get("gallery_ttl",    3600.0))

    camera_cfgs = _load_camera_configs(conf)
    if not camera_cfgs:
        print("[ERROR] No [camera_N] sections found in project.ini. Exiting.")
        sys.exit(1)

    print(f"[Main] Loaded {len(camera_cfgs)} camera(s): "
          f"{[c['id'] for c in camera_cfgs]}")

    # ── Load shared models ────────────────────────────────────────────────────
    print("[Main] Loading models …")
    model_vehicle = yolov5.load(model_vehicle_path)
    model_helmet  = YOLO(model_helmet_path)

    reid_weights = model_reid_path if Path(model_reid_path).exists() else None
    reid_model   = VehicleReID(
        weights_path = reid_weights,
        device       = "cpu",
    )

    # ── Shared processor (LPR) ────────────────────────────────────────────────
    os.makedirs(save_image_path, exist_ok=True)
    processor = Process(save_image_path)
    processor.load_number_plate_picture(model_plate_path)
    processor.load_number_plate(model_ocr_path)

    # ── Build GlobalTracker ───────────────────────────────────────────────────
    camera_positions = {c["id"]: c["position"] for c in camera_cfgs}
    global_tracker = GlobalTracker(
        reid_model               = reid_model,
        camera_positions         = camera_positions,
        avg_speed_mps            = avg_speed,
        st_sigma                 = st_sigma,
        lpr_confidence_threshold = lpr_conf_threshold,
        match_threshold          = match_threshold,
        gallery_ttl              = gallery_ttl,
        weights_plate_visible    = weights_visible,
        weights_plate_hidden     = weights_hidden,
    )

    # ── Database ──────────────────────────────────────────────────────────────
    try:
        db = DatabaseConnector()
        db.setup_schema()
    except Exception as e:
        print(f"[Main] DB unavailable ({e}). Trajectory persistence disabled.")
        db = None

    # ── Shared finished-tracklet queue ────────────────────────────────────────
    finished_queue: List[Tracklet] = []
    queue_lock = threading.Lock()

    # ── Start one worker thread per camera ───────────────────────────────────
    workers = []
    for cam_cfg in camera_cfgs:
        w = CameraWorker(
            cam_cfg        = cam_cfg,
            model_vehicle  = model_vehicle,
            model_helmet   = model_helmet,
            processor      = processor,
            finished_queue = finished_queue,
            queue_lock     = queue_lock,
            show_vehicle   = show_vehicle,
            show_tracking  = show_tracking,
            show_helmet    = show_helmet,
        )
        w.start()
        workers.append(w)

    print("[Main] All camera workers started. Press ESC in any window to quit.")

    # ── Main loop: drain finished tracklets and run GlobalTracker ────────────
    try:
        while any(w.is_alive() for w in workers):
            with queue_lock:
                batch = list(finished_queue)
                finished_queue.clear()

            if batch:
                global_ids = global_tracker.associate_batch(batch)
                for tracklet, gid in zip(batch, global_ids):
                    print(
                        f"[GlobalTracker] {tracklet.camera_id} "
                        f"local_id={tracklet.local_id} → "
                        f"global_id={gid}  "
                        f"plate='{tracklet.plate_text}'"
                        f"({tracklet.plate_confidence:.2f})"
                    )
                    # Persist to DB
                    if db:
                        try:
                            db.insert_trajectory_entry(
                                global_id        = gid,
                                camera_id        = tracklet.camera_id,
                                local_track_id   = tracklet.local_id,
                                entry_time       = tracklet.entry_time,
                                exit_time        = tracklet.exit_time,
                                plate_text       = tracklet.plate_text,
                                plate_confidence = tracklet.plate_confidence,
                                is_violation     = False,  # extend as needed
                            )
                        except Exception as e:
                            print(f"[DB] Insert error: {e}")

                    # Send violation email if plate is known
                    plate_file = os.path.join(
                        save_image_path,
                        f"plate_number{tracklet.local_id}.txt"
                    )
                    if os.path.exists(plate_file) and db:
                        trajectory = global_tracker.get_trajectory(gid)
                        traj_dicts = [e.to_dict() for e in trajectory]
                        try:
                            db.queryPlate(plate_file, trajectory=traj_dicts)
                        except Exception as e:
                            print(f"[DB] Email error: {e}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        for w in workers:
            w.running = False
        for w in workers:
            w.join(timeout=5)
        cv2.destroyAllWindows()
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()
