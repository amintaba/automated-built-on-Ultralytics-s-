Code 55% faster with GitHub Copilotimport carla
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from collections import defaultdict
import time
import math
import random

#  Custom DeepSORT class for tracking 
class DeepSORT:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """Initialize the DeepSORT tracker for multi-object tracking."""
        self.max_age = max_age  # How long to keep a track without a match
        self.min_hits = min_hits  # Minimum matches to confirm a track
        self.iou_threshold = iou_threshold  # IoU threshold for matching detections
        self.tracks = []  # List of active object tracks
        self.track_id = 0  # Unique ID for each tracked object
        self.history = defaultdict(list)  # Store past centers of tracked objects

    def update(self, detections, frame):
        """Update tracks with new detections from the current frame."""
        # If no tracks exist and we have new detections, create new tracks
        if len(self.tracks) == 0 and len(detections) > 0:
            for det in detections:
                self.tracks.append({
                    'id': self.track_id,
                    'bbox': det[:4],
                    'age': 0,
                    'hits': 1,
                    'last_center': self._get_center(det[:4])
                })
                self.history[self.track_id].append(self._get_center(det[:4]))
                self.track_id += 1
            return self.tracks

        # Calculate centers of tracks and detections, then compute distances
        track_centers = np.array([t['last_center'] for t in self.tracks])
        det_centers = np.array([self._get_center(det[:4]) for det in detections])
        distances = cdist(track_centers, det_centers, metric='euclidean')

        # Match tracks to detections using a simple distance-based algorithm
        matched = []
        for track_idx, det_idx in self._hungarian_matching(distances):
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track['bbox'] = det[:4]
            track['age'] = 0
            track['hits'] += 1
            track['last_center'] = self._get_center(det[:4])
            self.history[track['id']].append(track['last_center'])
            matched.append(det_idx)

        # Remove tracks that are too old
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        for t in self.tracks:
            t['age'] += 1

        # Add unmatched detections as new tracks
        unmatched_dets = [i for i in range(len(detections)) if i not in matched]
        for idx in unmatched_dets:
            self.tracks.append({
                'id': self.track_id,
                'bbox': detections[idx][:4],
                'age': 0,
                'hits': 1,
                'last_center': self._get_center(detections[idx][:4])
            })
            self.history[self.track_id].append(self._get_center(detections[idx][:4]))
            self.track_id += 1

        return self.tracks

    def _get_center(self, bbox):
        """Get the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def _hungarian_matching(self, distances):
        """Match tracks to detections based on minimum distance."""
        matched = []
        for track_idx in range(len(self.tracks)):
            det_idx = np.argmin(distances[track_idx])
            if distances[track_idx, det_idx] < 100:  # Distance threshold
                matched.append((track_idx, det_idx))
        return matched

# --- Speed estimation class ---
class SpeedEstimator:
    def __init__(self, fps, scale_m_per_pixel=0.05):
        """Initialize speed estimator based on pixel movement."""
        self.fps = fps  # Frame rate of the simulation
        self.scale_m_per_pixel = scale_m_per_pixel  # Convert pixels to meters
        self.prev_centers = {}  # Store previous centers for speed calculation

    def estimate(self, track_id, current_center, frame_idx):
        """Calculate speed in km/h based on pixel displacement."""
        if track_id not in self.prev_centers:
            self.prev_centers[track_id] = (current_center, frame_idx)
            return 0.0

        prev_center, prev_frame = self.prev_centers[track_id]
        displacement = np.sqrt((current_center[0] - prev_center[0])**2 +
                              (current_center[1] - prev_center[1])**2)
        time_diff = (frame_idx - prev_frame) / self.fps
        speed_mps = (displacement * self.scale_m_per_pixel) / time_diff if time_diff > 0 else 0
        speed_kmh = speed_mps * 3.6  # Convert meters/second to km/h
        self.prev_centers[track_id] = (current_center, frame_idx)
        return speed_kmh

# --- Render frame with information ---
def render_frame(frame, tracks, history, speed_estimator, frame_idx):
    """Draw bounding boxes, trajectories, and speed info on the frame."""
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        track_id = track['id']
        center = track['last_center']

        # Draw bounding box and object ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate and display speed
        speed = speed_estimator.estimate(track_id, center, frame_idx)
        cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw trajectory line for the last 10 points
        if len(history[track_id]) > 1:
            for i in range(max(0, len(history[track_id])-10), len(history[track_id])-1):
                pt1 = tuple(map(int, history[track_id][i]))
                pt2 = tuple(map(int, history[track_id][i+1]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    # Show the number of tracked objects on the frame
    cv2.putText(frame, f"Objects: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# --- Main function ---
def main():
    """Run the CARLA traffic intelligence simulation."""
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set up the simulation world
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()

    # Spawn the ego vehicle (main car)
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    ego_vehicle.set_autopilot(True)

    # Attach an RGB camera to the ego vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    # Spawn additional vehicles and pedestrians
    for _ in range(20):
        vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
        vehicle.set_autopilot(True)

    walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]
    for _ in range(10):
        walker = world.spawn_actor(walker_bp, random.choice(spawn_points))
        walker.set_simulate_physics(True)

    # Load the YOLOv11L model for object detection
    model = YOLO('yolov11l.pt')

    # Set up the tracker and speed estimator
    tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
    speed_estimator = SpeedEstimator(fps=30, scale_m_per_pixel=0.05)  # Assume 30 FPS for CARLA
    frame_idx = 0

    # Process camera images
    def process_image(image):
        nonlocal frame_idx
        frame = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]

        # Run YOLO for object detection
        results = model(frame, conf=0.5, classes=[0, 2, 5, 7])  # Detect people, cars, buses, trucks
        detections = results[0].boxes.data.cpu().numpy()

        # Update object tracks
        tracks = tracker.update(detections, frame)

        # Draw info on the frame
        frame = render_frame(frame, tracks, tracker.history, speed_estimator, frame_idx)

        # Display the frame in real-time
        cv2.imshow('CARLA Traffic Intelligence', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

        frame_idx += 1

    # Start the camera feed
    camera.listen(process_image)

    # Keep the simulation running
    try:
        while True:
            time.sleep(0.01)
    finally:
        # Clean up when done
        camera.stop()
        for actor in world.get_actors():
            actor.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
