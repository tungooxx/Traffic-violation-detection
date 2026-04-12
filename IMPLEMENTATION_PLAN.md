# Implementation Plan: Multi-Camera Vehicle Tracking System

This document outlines a step-by-step, actionable plan to build the Multi-Target Multi-Camera (MTMC) tracking system on top of the existing `Traffic-violation-detection` repository. The plan is divided into five logical phases, moving from environment setup to final integration.

## Phase 1: Simulation Environment Setup (CARLA)

Before modifying the core repository, we need a controlled environment to generate multi-camera data with ground truth labels. This data is essential for training the Re-ID model and tuning the fusion weights.

**Goal**: Generate a synthetic dataset of vehicles passing through multiple cameras.

1.  **Install CARLA Simulator**:
    -   Download and install CARLA (version 0.9.15 is recommended for stability) on a machine with a dedicated GPU (minimum 6GB VRAM) [1].
    -   Set up the Python API environment (`pip install carla`).
2.  **Design the Camera Network**:
    -   Select a suitable map (e.g., `Town03` or `Town04` for urban intersections).
    -   Write a Python script (`carla_data_gen.py`) to spawn 3-4 RGB cameras at fixed coordinates, simulating a real-world traffic enforcement setup.
    -   Ensure the cameras have overlapping or sequential fields of view.
3.  **Generate Traffic and Record Data**:
    -   Use the CARLA Traffic Manager to spawn 100+ vehicles with random routes.
    -   Configure the script to record synchronized video streams from all cameras.
    -   Crucially, export the ground truth data: for every frame, record the bounding box, the global vehicle ID (provided by CARLA), the camera ID, and the timestamp. Save this to a CSV file.

## Phase 2: Upgrading Single-Camera Tracking

The current system uses SORT, which resets IDs per session and does not extract appearance features. We need to upgrade to a tracker that provides stable tracklets and appearance embeddings.

**Goal**: Replace SORT with BoT-SORT and output tracklets instead of per-frame detections.

1.  **Integrate BoT-SORT**:
    -   Install the `boxmot` library, which provides a clean implementation of BoT-SORT [2].
    -   In `yolo-detect.py`, replace the `Tracker = Sort(...)` initialization with the BoT-SORT equivalent.
2.  **Implement Tracklet Building**:
    -   Modify the tracking loop in `yolo-detect.py`. Instead of processing each frame independently, accumulate bounding box crops for each local ID.
    -   Create a `Tracklet` class to store these crops, the local ID, the entry/exit timestamps, and the camera ID.
    -   When a vehicle leaves the frame (i.e., the tracker loses it for `max_age` frames), finalize the `Tracklet` and pass it to the new Global Tracking Layer.

## Phase 3: Feature Extraction Integration

With stable tracklets available, we need to extract the two key signals: visual appearance (Re-ID) and license plate text (LPR).

**Goal**: Extract Re-ID embeddings and OCR strings with confidence scores from each tracklet.

1.  **Integrate Visual Re-ID (OSNet)**:
    -   Install the `torchreid` library [3].
    -   Download a pre-trained OSNet model (e.g., `osnet_x1_0_vehicleid.pth`).
    -   Create a new file `reid.py` with a `VehicleReID` class. This class should take a `Tracklet` (a list of image crops), run them through OSNet, and average the resulting feature vectors to produce a single, robust embedding for the vehicle.
2.  **Enhance the LPR Pipeline**:
    -   The existing `process.py` handles plate detection and OCR.
    -   Modify the `number_plate_extract` function to return not just the OCR string, but also a confidence score. This score can be derived from the YOLOv5 prediction confidences for each character.
    -   If the plate is not detected in any crop within the tracklet, return an empty string and a confidence of 0.0.

## Phase 4: The Global Tracking Layer

This is the core of the MTMC system, where tracklets from different cameras are associated using the weighted fusion logic.

**Goal**: Implement the weighted fusion scoring and assign persistent Global IDs.

1.  **Create the Global Tracker**:
    -   Create a new file `global_tracker.py` containing the `GlobalTracker` class.
    -   This class maintains a "gallery" of all previously seen tracklets and their assigned Global IDs.
2.  **Implement the Scoring Functions**:
    -   **`S_reid`**: Implement cosine similarity between the OSNet embeddings.
    -   **`S_lpr`**: Implement the normalized Levenshtein distance for the OCR strings.
    -   **`S_st`**: Implement a spatio-temporal probability function based on the camera coordinates and timestamps.
3.  **Implement Dynamic Weighting**:
    -   Write the logic to adjust weights based on the LPR confidence score (e.g., if confidence > 0.8, favor `S_lpr`; otherwise, favor `S_reid`).
    -   Calculate the final `S_total` score.
4.  **Bipartite Matching**:
    -   When a new tracklet arrives, compute `S_total` against all tracklets in the gallery.
    -   Use the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to find the optimal match.
    -   If the best match score exceeds a predefined threshold, assign the existing Global ID. Otherwise, create a new Global ID.

## Phase 5: Database and Alert Integration

Finally, the system must store the full trajectories and update the alert mechanism to report cross-camera violations.

**Goal**: Store trajectories in MySQL and send comprehensive violation emails.

1.  **Update Database Schema**:
    -   In MySQL, create a new table `trajectories` linked to the existing `Motobike` table.
    -   Columns should include: `global_id`, `camera_id`, `local_track_id`, `entry_time`, `exit_time`, `plate_number`, and a reference to the Re-ID embedding.
2.  **Modify `workWithDatabase.py`**:
    -   Update the `DatabaseConnector` class to handle inserts into the new `trajectories` table whenever the `GlobalTracker` finalizes an association.
3.  **Enhance Violation Alerts**:
    -   Modify `emailForm.py` and the email logic in `workWithDatabase.py`.
    -   When a violation is detected (e.g., no helmet in Camera 1), the alert should now include the full trajectory of that `global_id` (e.g., "Vehicle spotted at Camera 1 at 10:00, Camera 2 at 10:05").

## References

[1] CARLA Simulator Documentation. Available: https://carla.org/

[2] M. Yang et al., "Cross-Camera Multi-Target Vehicle Tracking," *The 37th IPPR Conference on Computer Vision, Graphics, and Image Processing*, 2024. Available: https://www.csie.ntu.edu.tw/~fuh/personal/Cross-CameraMulti-TargetVehicleTracking.pdf

[3] Z. Li, "Person Re-Identification in a Video Sequence," *Stanford CS231n*, 2024. Available: https://cs231n.stanford.edu/2024/papers/person-re-identification-in-a-video-sequence.pdf
