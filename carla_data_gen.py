"""
carla_data_gen.py
-----------------
CARLA Simulator multi-camera dataset generation script.

This script connects to a running CARLA server, places a configurable
number of RGB cameras at fixed infrastructure positions, spawns traffic,
and records synchronized video streams alongside ground-truth CSV annotations.

Requirements:
    - CARLA 0.9.15 server running (./CarlaUE4.sh or CarlaUE4.exe)
    - Python package: carla  (pip install carla==0.9.15)

Usage:
    python carla_data_gen.py --town Town03 --num-vehicles 80 \
                              --duration 120 --output-dir carla_dataset

Output structure:
    carla_dataset/
        camera_0/  <-- synchronized JPEG frames
        camera_1/
        camera_2/
        camera_3/
        annotations.csv  <-- frame, camera_id, global_vehicle_id, x1, y1, x2, y2
"""

import argparse
import csv
import os
import queue
import random
import sys
import time
import weakref
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── CARLA import guard ────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    print(
        "[ERROR] The 'carla' package is not installed.\n"
        "  Install it with:  pip install carla==0.9.15\n"
        "  Make sure the CARLA server is running before executing this script."
    )
    sys.exit(1)

# ── Camera placement definitions ─────────────────────────────────────────────
# Each entry is (x, y, z, pitch, yaw, roll) in CARLA world coordinates.
# These positions are tuned for Town03 intersections; adjust for other maps.
CAMERA_TRANSFORMS = [
    carla.Transform(carla.Location(x=-50, y=0,   z=8), carla.Rotation(pitch=-20, yaw=0)),
    carla.Transform(carla.Location(x=50,  y=0,   z=8), carla.Rotation(pitch=-20, yaw=180)),
    carla.Transform(carla.Location(x=0,   y=-50, z=8), carla.Rotation(pitch=-20, yaw=90)),
    carla.Transform(carla.Location(x=0,   y=50,  z=8), carla.Rotation(pitch=-20, yaw=270)),
]

IMAGE_WIDTH  = 854
IMAGE_HEIGHT = 480
FOV          = 90
WARMUP_TICKS = 30


def _build_camera_transforms(spawn_points):
    """Place cameras at real road spawn points for the loaded town."""
    if not spawn_points:
        return CAMERA_TRANSFORMS

    camera_transforms = []
    for spawn_point in spawn_points[:4]:
        location = spawn_point.location + carla.Location(z=8.0)
        rotation = carla.Rotation(
            pitch=-15.0,
            yaw=spawn_point.rotation.yaw,
            roll=0.0,
        )
        camera_transforms.append(carla.Transform(location, rotation))
    return camera_transforms


# ── Helper: RGB camera sensor callback ───────────────────────────────────────
def _make_camera_callback(frame_queue: queue.Queue, camera_id: int):
    """Return a callback that pushes (camera_id, frame_number, np_image) into queue."""
    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]          # drop alpha channel
        frame_queue.put((camera_id, image.frame, bgr.copy()))
    return callback


def _write_camera_videos(output_path: Path, camera_count: int, fps: float):
    """Create one MP4 video per camera from saved JPEG frames."""
    for camera_id in range(camera_count):
        camera_dir = output_path / f"camera_{camera_id}"
        frame_paths = sorted(camera_dir.glob("*.jpg"))
        if not frame_paths:
            print(f"[WARN] Camera {camera_id}: no frames found, skipping video.")
            continue

        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            print(f"[WARN] Camera {camera_id}: cannot read first frame, skipping video.")
            continue

        height, width = first_frame.shape[:2]
        video_path = output_path / f"camera_{camera_id}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
        writer.release()
        print(f"       Camera {camera_id} video: {video_path}")


def _warn_if_frame_looks_corrupt(frame_path: Path):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"[WARN] Could not read saved frame: {frame_path}")
        return

    row_change = np.mean(np.abs(np.diff(frame.astype(np.float32), axis=0)))
    col_change = np.mean(np.abs(np.diff(frame.astype(np.float32), axis=1)))
    if row_change > col_change * 3.0:
        print(
            f"[WARN] {frame_path} may be corrupted or striped. "
            "Fix CARLA rendering before trusting videos."
        )


# ── Helper: bounding-box projection ──────────────────────────────────────────
def _world_to_image(world_point: carla.Location,
                    camera_actor: carla.Actor,
                    image_w: int,
                    image_h: int,
                    fov: float):
    """
    Project a 3-D world point onto the 2-D camera image plane.
    Returns (u, v) pixel coordinates, or None if the point is behind the camera.
    """
    # Build the camera intrinsic matrix K
    focal = image_w / (2.0 * np.tan(np.radians(fov / 2.0)))
    K = np.array([
        [focal,  0,      image_w / 2.0],
        [0,      focal,  image_h / 2.0],
        [0,      0,      1.0          ],
    ])

    # Transform world → camera coordinate frame
    cam_transform = camera_actor.get_transform()
    world_to_cam  = np.array(cam_transform.get_inverse_matrix())

    wp = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    cam_coords = world_to_cam @ wp          # shape (4,)

    # CARLA uses left-handed UE4 axes; convert to standard camera axes
    # UE4: X=forward, Y=right, Z=up  →  camera: X=right, Y=down, Z=forward
    x_cam =  cam_coords[1]
    y_cam = -cam_coords[2]
    z_cam =  cam_coords[0]

    if z_cam <= 0:
        return None     # point is behind the camera

    u = int(K[0, 0] * x_cam / z_cam + K[0, 2])
    v = int(K[1, 1] * y_cam / z_cam + K[1, 2])
    return (u, v)


def _get_bounding_box_2d(vehicle: carla.Actor,
                          camera_actor: carla.Actor,
                          image_w: int,
                          image_h: int,
                          fov: float):
    """
    Compute the 2-D axis-aligned bounding box of a vehicle in the camera image.
    Returns (x1, y1, x2, y2) or None if the vehicle is not visible.
    """
    bb = vehicle.bounding_box
    verts_local = [
        carla.Location(x=bb.extent.x * sx,
                       y=bb.extent.y * sy,
                       z=bb.extent.z * sz)
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]

    vehicle_transform = vehicle.get_transform()
    verts_world = [vehicle_transform.transform(v) for v in verts_local]

    pixels = [_world_to_image(v, camera_actor, image_w, image_h, fov)
              for v in verts_world]
    pixels = [p for p in pixels if p is not None]

    if not pixels:
        return None

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(image_w - 1, max(xs)), min(image_h - 1, max(ys))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# ── Main generation function ──────────────────────────────────────────────────
def generate(town: str,
             num_vehicles: int,
             duration_seconds: int,
             output_dir: str,
             host: str = "127.0.0.1",
             port: int  = 2000):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    annotation_file = output_path / "annotations.csv"
    csv_fields = ["frame", "camera_id", "global_vehicle_id", "x1", "y1", "x2", "y2"]

    client  = carla.Client(host, port)
    client.set_timeout(120.0)
    world = client.get_world()
    current_map = world.get_map().name.rsplit("/", 1)[-1]
    if current_map != town:
        world = client.load_world(town)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    settings = world.get_settings()

    settings.synchronous_mode  = True
    settings.fixed_delta_seconds = 0.1    # 10 FPS
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()
    spawn_points       = world.get_map().get_spawn_points()
    camera_transforms  = _build_camera_transforms(spawn_points)

    # Create per-camera frame directories
    for i in range(len(camera_transforms)):
        (output_path / f"camera_{i}").mkdir(exist_ok=True)

    actor_list = []
    camera_actors = []
    frame_queue   = queue.Queue()
    videos_written = False

    try:
        # ── Spawn cameras ─────────────────────────────────────────────────────
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        cam_bp.set_attribute("fov",          str(FOV))

        for i, transform in enumerate(camera_transforms):
            cam = world.spawn_actor(cam_bp, transform)
            cam.listen(_make_camera_callback(frame_queue, i))
            camera_actors.append(cam)
            actor_list.append(cam)
            print(f"[INFO] Camera {i} spawned at {transform.location}")

        # ── Spawn vehicles ────────────────────────────────────────────────────
        vehicle_blueprints = blueprint_library.filter("vehicle.*")
        vehicle_blueprints = [bp for bp in vehicle_blueprints
                               if int(bp.get_attribute("number_of_wheels")) == 4]

        spawned_vehicles = []
        random.shuffle(spawn_points)
        for i, sp in enumerate(spawn_points[:num_vehicles]):
            bp = random.choice(vehicle_blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            v = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True, traffic_manager.get_port())
                spawned_vehicles.append(v)
                actor_list.append(v)

        print(f"[INFO] Spawned {len(spawned_vehicles)} vehicles.")

        # ── Record loop ───────────────────────────────────────────────────────
        print(f"[INFO] Warming up CARLA for {WARMUP_TICKS} ticks before recording.")
        for _ in range(WARMUP_TICKS):
            world.tick()
        while not frame_queue.empty():
            frame_queue.get_nowait()

        total_frames   = int(duration_seconds / settings.fixed_delta_seconds)
        saved_frames   = {i: 0 for i in range(len(camera_transforms))}
        annotations    = []

        print(f"[INFO] Recording {total_frames} ticks (~{duration_seconds}s) …")
        start = time.time()

        for tick in range(total_frames):
            world.tick()

            # Drain the frame queue
            while not frame_queue.empty():
                cam_id, frame_no, bgr = frame_queue.get_nowait()
                frame_filename = (
                    output_path / f"camera_{cam_id}" /
                    f"{saved_frames[cam_id]:06d}.jpg"
                )
                cv2.imwrite(str(frame_filename), bgr)
                saved_frames[cam_id] += 1

                # Annotate visible vehicles for this camera frame
                for vehicle in spawned_vehicles:
                    bb2d = _get_bounding_box_2d(
                        vehicle, camera_actors[cam_id],
                        IMAGE_WIDTH, IMAGE_HEIGHT, FOV
                    )
                    if bb2d:
                        x1, y1, x2, y2 = bb2d
                        annotations.append({
                            "frame":             saved_frames[cam_id] - 1,
                            "camera_id":         cam_id,
                            "global_vehicle_id": vehicle.id,
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2,
                        })

            if tick % 20 == 0:
                elapsed = time.time() - start
                print(f"  tick {tick}/{total_frames}  ({elapsed:.1f}s elapsed)")

        # Write CSV
        with open(annotation_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(annotations)

        print(f"\n[DONE] Dataset saved to '{output_dir}'")
        print(f"       Annotations: {len(annotations)} rows → {annotation_file}")
        for i in range(len(camera_transforms)):
            print(f"       Camera {i}: {saved_frames[i]} frames")
            first_frame = output_path / f"camera_{i}" / "000000.jpg"
            if first_frame.exists():
                _warn_if_frame_looks_corrupt(first_frame)
        _write_camera_videos(
            output_path,
            len(camera_transforms),
            fps=1.0 / settings.fixed_delta_seconds,
        )
        videos_written = True

    finally:
        if not videos_written:
            _write_camera_videos(
                output_path,
                len(camera_transforms),
                fps=1.0 / settings.fixed_delta_seconds,
            )
        # Restore async mode and destroy actors
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[INFO] Destroying actors …")
        for actor in actor_list:
            actor.destroy()
        print("[INFO] Cleanup complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate multi-camera traffic dataset using CARLA simulator"
    )
    p.add_argument("--town",          default="Town03",
                   help="CARLA map name (default: Town03)")
    p.add_argument("--num-vehicles",  type=int, default=80,
                   help="Number of vehicles to spawn (default: 80)")
    p.add_argument("--duration",      type=int, default=120,
                   help="Recording duration in seconds (default: 120)")
    p.add_argument("--output-dir",    default="carla_dataset",
                   help="Output directory (default: carla_dataset)")
    p.add_argument("--host",          default="127.0.0.1",
                   help="CARLA server host (default: 127.0.0.1)")
    p.add_argument("--port",          type=int, default=2000,
                   help="CARLA server port (default: 2000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        town=args.town,
        num_vehicles=args.num_vehicles,
        duration_seconds=args.duration,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
    )
