# Football XY Tracking System — Complete Build Instructions

## Overview

Build an end-to-end football (soccer) XY tracking system that processes single-camera wide-angle match footage and outputs frame-by-frame positional data for all 22 players and the ball.

**Target Output**: JSON/CSV files containing timestamped XY coordinates (in meters) for every player and the ball, plus derived physical metrics (speed, distance, acceleration).

**Footage Specifications**:
- Camera: Centered, elevated, fixed position
- Resolution: 4K (3840×2160)
- Frame rate: 25-30 fps (standard broadcast)
- Angle: Consistent throughout match (no pan/tilt/zoom)

---

## Project Structure

```
football-tracker/
├── config/
│   ├── config.yaml
│   └── pitch_template.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── player_detector.py
│   │   ├── ball_detector.py
│   │   └── team_classifier.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   ├── reid.py
│   │   └── interpolation.py
│   ├── homography/
│   │   ├── __init__.py
│   │   ├── pitch_detector.py
│   │   ├── calibration.py
│   │   └── transformer.py
│   ├── identity/
│   │   ├── __init__.py
│   │   ├── jersey_ocr.py
│   │   ├── goalkeeper_detector.py
│   │   └── player_assignment.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── physical.py
│   │   └── zones.py
│   ├── output/
│   │   ├── __init__.py
│   │   ├── data_export.py
│   │   └── visualizer.py
│   └── utils/
│       ├── __init__.py
│       ├── video.py
│       ├── geometry.py
│       └── smoothing.py
├── models/
├── data/
│   ├── input/
│   ├── output/
│   └── cache/
├── scripts/
│   ├── download_models.sh
│   └── process_match.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

### requirements.txt

```
# Deep Learning & Computer Vision
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.2.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0

# Tracking
supervision>=0.22.0
lapx>=0.5.0

# Homography & Geometry
kornia>=0.7.0
scikit-image>=0.21.0

# Re-identification
timm>=1.0.0

# Jersey OCR
paddlepaddle>=2.5.0
paddleocr>=2.7.0

# Visualization
matplotlib>=3.7.0
mplsoccer>=1.2.0
plotly>=5.15.0

# Data Processing
pandas>=2.0.0
polars>=0.20.0
pyyaml>=6.0

# Utilities
tqdm>=4.65.0
loguru>=0.7.0
typer>=0.9.0
rich>=13.5.0
decord>=0.6.0
```

---

## Configuration Files

### config/config.yaml

```yaml
project:
  name: "football-tracker"
  version: "0.1.0"

video:
  input_path: "data/input/"
  output_path: "data/output/"
  cache_path: "data/cache/"

processing:
  frame_rate: 25
  batch_size: 8
  device: "cuda"
  half_precision: true

detection:
  player:
    model: "yolov8x.pt"
    confidence: 0.3
    iou_threshold: 0.5
    min_height: 30
    max_height: 400
  ball:
    model: "yolov8x.pt"
    confidence: 0.2
    temporal_window: 5

tracking:
  tracker: "bytetrack"
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30
  match_thresh: 0.8

homography:
  method: "keypoint"
  min_keypoints: 4
  ransac_threshold: 3.0
  temporal_smoothing: true
  smoothing_window: 15

team_classification:
  method: "kmeans"
  n_clusters: 3
  color_space: "LAB"

identity:
  jersey_ocr:
    enabled: true
    confidence_threshold: 0.7
    aggregation_window: 50
  manual_init: true

metrics:
  speed_smoothing: 5
  sprint_threshold: 25.0
  high_intensity_threshold: 19.8

output:
  format: ["json", "csv"]
  include_raw_detections: false
  visualization: true
```

### config/pitch_template.yaml

```yaml
pitch:
  length: 105.0
  width: 68.0
  center_circle:
    x: 52.5
    y: 34.0
    radius: 9.15

keypoints:
  - name: "corner_top_left"
    x: 0.0
    y: 0.0
  - name: "corner_top_right"
    x: 105.0
    y: 0.0
  - name: "corner_bottom_left"
    x: 0.0
    y: 68.0
  - name: "corner_bottom_right"
    x: 105.0
    y: 68.0
  - name: "center_top"
    x: 52.5
    y: 0.0
  - name: "center_bottom"
    x: 52.5
    y: 68.0
  - name: "center_spot"
    x: 52.5
    y: 34.0
  - name: "penalty_spot_left"
    x: 11.0
    y: 34.0
  - name: "penalty_spot_right"
    x: 94.0
    y: 34.0
  - name: "penalty_left_top_edge"
    x: 16.5
    y: 13.85
  - name: "penalty_left_bottom_edge"
    x: 16.5
    y: 54.15
  - name: "penalty_right_top_edge"
    x: 88.5
    y: 13.85
  - name: "penalty_right_bottom_edge"
    x: 88.5
    y: 54.15
```

---

## Core Implementation Files

### src/detection/player_detector.py

```python
"""Player detection using YOLOv8."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    track_id: Optional[int] = None

    @property
    def foot_position(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class PlayerDetector:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        min_height: int = 30,
        max_height: int = 400,
        device: str = "cuda",
        half: bool = True
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_height = min_height
        self.max_height = max_height

        logger.info(f"Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        if half and device == "cuda":
            self.model.model.half()

    def detect(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            height = bbox[3] - bbox[1]

            if height < self.min_height or height > self.max_height:
                continue

            if pitch_mask is not None:
                cx, cy = (bbox[0] + bbox[2]) / 2, bbox[3]
                h, w = pitch_mask.shape[:2]
                px, py = int(cx), int(cy)
                if 0 <= px < w and 0 <= py < h:
                    if pitch_mask[py, px] == 0:
                        continue

            detections.append(Detection(bbox=bbox, confidence=conf, class_id=self.PERSON_CLASS_ID))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        results = self.model(frames, conf=self.confidence, iou=self.iou_threshold, 
                           classes=[self.PERSON_CLASS_ID], verbose=False)
        all_detections = []
        for result in results:
            frame_detections = []
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                height = bbox[3] - bbox[1]
                if self.min_height <= height <= self.max_height:
                    frame_detections.append(Detection(bbox=bbox, confidence=conf, class_id=0))
            all_detections.append(frame_detections)
        return all_detections
```

### src/detection/ball_detector.py

```python
"""Ball detection with temporal consistency."""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class BallDetection:
    position: Tuple[float, float]
    confidence: float
    bbox: Optional[np.ndarray] = None
    is_interpolated: bool = False


class BallDetector:
    SPORTS_BALL_CLASS = 32

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence: float = 0.2,
        temporal_window: int = 5,
        max_velocity: float = 150.0,
        device: str = "cuda"
    ):
        self.confidence = confidence
        self.temporal_window = temporal_window
        self.max_velocity = max_velocity

        logger.info(f"Loading ball detection model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        self.position_buffer = deque(maxlen=temporal_window)
        self.velocity_buffer = deque(maxlen=temporal_window - 1)
        self.last_position: Optional[Tuple[float, float]] = None
        self.frames_since_detection = 0

    def detect(self, frame: np.ndarray, player_bboxes: Optional[List[np.ndarray]] = None) -> Optional[BallDetection]:
        # Try YOLO detection
        results = self.model(frame, conf=self.confidence, classes=[self.SPORTS_BALL_CLASS], verbose=False)[0]

        if len(results.boxes) > 0:
            best_idx = results.boxes.conf.argmax()
            bbox = results.boxes.xyxy[best_idx].cpu().numpy()
            conf = float(results.boxes.conf[best_idx])
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if self._validate_detection(center, player_bboxes):
                self._update_temporal_state(center)
                self.frames_since_detection = 0
                return BallDetection(position=center, confidence=conf, bbox=bbox)

        # Interpolation fallback
        self.frames_since_detection += 1
        if self.frames_since_detection <= self.temporal_window and self.last_position:
            interpolated = self._interpolate_position()
            if interpolated:
                return BallDetection(
                    position=interpolated,
                    confidence=0.5 / self.frames_since_detection,
                    is_interpolated=True
                )
        return None

    def _validate_detection(self, position: Tuple[float, float], player_bboxes: Optional[List[np.ndarray]]) -> bool:
        if self.last_position:
            dist = np.sqrt((position[0] - self.last_position[0])**2 + (position[1] - self.last_position[1])**2)
            if dist > self.max_velocity * 2:
                return False
        return True

    def _update_temporal_state(self, position: Tuple[float, float]):
        if self.last_position:
            velocity = (position[0] - self.last_position[0], position[1] - self.last_position[1])
            self.velocity_buffer.append(velocity)
        self.position_buffer.append(position)
        self.last_position = position

    def _interpolate_position(self) -> Optional[Tuple[float, float]]:
        if not self.last_position or len(self.velocity_buffer) == 0:
            return None
        avg_vx = np.mean([v[0] for v in self.velocity_buffer])
        avg_vy = np.mean([v[1] for v in self.velocity_buffer])
        return (
            self.last_position[0] + avg_vx * self.frames_since_detection,
            self.last_position[1] + avg_vy * self.frames_since_detection
        )

    def reset(self):
        self.position_buffer.clear()
        self.velocity_buffer.clear()
        self.last_position = None
        self.frames_since_detection = 0
```

### src/detection/team_classifier.py

```python
"""Team classification based on jersey colors."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


class Team(Enum):
    HOME = 0
    AWAY = 1
    REFEREE = 2
    GOALKEEPER_HOME = 3
    GOALKEEPER_AWAY = 4
    UNKNOWN = -1


@dataclass
class TeamClassification:
    team: Team
    confidence: float
    dominant_color: Tuple[int, int, int]


class TeamClassifier:
    def __init__(self, n_clusters: int = 3, color_space: str = "LAB",
                 jersey_region: Tuple[float, float, float, float] = (0.2, 0.15, 0.8, 0.55)):
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.jersey_region = jersey_region
        self.team_colors: Optional[Dict[Team, np.ndarray]] = None
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False
        self.cluster_to_team: Dict[int, Team] = {}

    def extract_jersey_color(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        jx1 = x1 + int(w * self.jersey_region[0])
        jy1 = y1 + int(h * self.jersey_region[1])
        jx2 = x1 + int(w * self.jersey_region[2])
        jy2 = y1 + int(h * self.jersey_region[3])
        jersey_crop = frame[jy1:jy2, jx1:jx2]
        if jersey_crop.size == 0:
            return np.array([0, 0, 0])

        if self.color_space == "LAB":
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2LAB)
        elif self.color_space == "HSV":
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        else:
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2RGB)

        pixels = jersey_crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 10:
            return np.array([0, 0, 0])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten())
        return centers[counts.argmax()]

    def fit(self, frame: np.ndarray, bboxes: List[np.ndarray]):
        all_colors = []
        for bbox in bboxes:
            color = self.extract_jersey_color(frame, bbox)
            if np.any(color > 0):
                all_colors.append(color)
        if len(all_colors) < self.n_clusters:
            raise ValueError(f"Not enough colors: {len(all_colors)}")

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(np.array(all_colors))

        cluster_counts = np.bincount(self.kmeans.labels_, minlength=self.n_clusters)
        sorted_clusters = np.argsort(cluster_counts)[::-1]

        self.team_colors = {
            Team.HOME: self.kmeans.cluster_centers_[sorted_clusters[0]],
            Team.AWAY: self.kmeans.cluster_centers_[sorted_clusters[1]],
        }
        if self.n_clusters > 2:
            self.team_colors[Team.REFEREE] = self.kmeans.cluster_centers_[sorted_clusters[2]]

        self.cluster_to_team = {sorted_clusters[0]: Team.HOME, sorted_clusters[1]: Team.AWAY}
        if self.n_clusters > 2:
            self.cluster_to_team[sorted_clusters[2]] = Team.REFEREE
        self.is_fitted = True

    def classify(self, frame: np.ndarray, bbox: np.ndarray) -> TeamClassification:
        if not self.is_fitted:
            return TeamClassification(team=Team.UNKNOWN, confidence=0.0, dominant_color=(0, 0, 0))

        color = self.extract_jersey_color(frame, bbox)
        if np.all(color == 0):
            return TeamClassification(team=Team.UNKNOWN, confidence=0.0, dominant_color=(0, 0, 0))

        distances = np.linalg.norm(self.kmeans.cluster_centers_ - color, axis=1)
        nearest_cluster = np.argmin(distances)
        confidence = max(0, 1 - (distances[nearest_cluster] / 100))
        team = self.cluster_to_team.get(nearest_cluster, Team.UNKNOWN)

        if self.color_space == "LAB":
            rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0, 0]
        elif self.color_space == "HSV":
            rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0, 0]
        else:
            rgb_color = color.astype(np.uint8)

        return TeamClassification(team=team, confidence=confidence, dominant_color=tuple(rgb_color))
```

### src/tracking/tracker.py

```python
"""Multi-object tracking using ByteTrack."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv
from ..detection.player_detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    team: Optional[int] = None
    jersey_number: Optional[int] = None
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def foot_position(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    def add_position(self, max_history: int = 100):
        self.history.append(self.foot_position)
        if len(self.history) > max_history:
            self.history.pop(0)


class PlayerTracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 25):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        self.frame_count += 1
        if len(detections) == 0:
            return list(self.tracks.values())

        bboxes = np.array([d.bbox for d in detections])
        confidences = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(xyxy=bboxes, confidence=confidences, class_id=class_ids)
        tracked = self.tracker.update_with_detections(sv_detections)

        active_tracks = []
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)
                if track_id not in self.tracks:
                    self.tracks[track_id] = Track(
                        track_id=track_id,
                        bbox=tracked.xyxy[i],
                        confidence=tracked.confidence[i] if tracked.confidence is not None else 1.0,
                        class_id=int(tracked.class_id[i]) if tracked.class_id is not None else 0
                    )
                else:
                    self.tracks[track_id].bbox = tracked.xyxy[i]
                    self.tracks[track_id].confidence = tracked.confidence[i] if tracked.confidence is not None else 1.0
                self.tracks[track_id].add_position()
                active_tracks.append(self.tracks[track_id])
        return active_tracks

    def reset(self):
        self.tracker.reset()
        self.tracks.clear()
        self.frame_count = 0
```

### src/homography/calibration.py

```python
"""Camera calibration and homography estimation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


@dataclass
class CalibrationResult:
    homography: np.ndarray
    reprojection_error: float
    keypoints_used: List[str]
    is_valid: bool


class HomographyEstimator:
    def __init__(self, min_keypoints: int = 4, ransac_threshold: float = 3.0):
        self.min_keypoints = min_keypoints
        self.ransac_threshold = ransac_threshold
        self.current_homography: Optional[np.ndarray] = None

    def estimate_from_manual_points(
        self,
        pixel_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> CalibrationResult:
        if len(pixel_points) < 4 or len(pixel_points) != len(world_points):
            return CalibrationResult(np.eye(3), float('inf'), [], False)

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)

        if H is None:
            return CalibrationResult(np.eye(3), float('inf'), [], False)

        projected = self._transform_points(src, H)
        error = np.mean(np.linalg.norm(projected - dst, axis=1))
        self.current_homography = H

        return CalibrationResult(H, error, ["manual"] * len(pixel_points), True)

    def _transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        points_h = np.column_stack([points, np.ones(len(points))])
        transformed_h = (H @ points_h.T).T
        return transformed_h[:, :2] / transformed_h[:, 2:3]


class InteractiveCalibrator:
    def __init__(self, pitch_template: Dict):
        self.pitch_template = pitch_template
        self.pixel_points: List[Tuple[float, float]] = []
        self.world_points: List[Tuple[float, float]] = []
        self.current_frame: Optional[np.ndarray] = None
        self.common_keypoints = [
            ("corner_top_left", (0, 0)),
            ("corner_top_right", (105, 0)),
            ("corner_bottom_left", (0, 68)),
            ("corner_bottom_right", (105, 68)),
            ("center_top", (52.5, 0)),
            ("center_bottom", (52.5, 68)),
            ("center_spot", (52.5, 34)),
            ("penalty_spot_left", (11, 34)),
            ("penalty_spot_right", (94, 34)),
        ]

    def calibrate_interactive(self, frame: np.ndarray) -> CalibrationResult:
        self.current_frame = frame.copy()
        self.pixel_points = []
        self.world_points = []

        window_name = "Calibration - Click keypoints"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("Click keypoints in order:")
        for i, (name, coords) in enumerate(self.common_keypoints):
            print(f"  {i+1}. {name} -> {coords}")
        print("Press 'q' when done, 'r' to reset, 'u' to undo")

        while True:
            display = self.current_frame.copy()
            for i, pt in enumerate(self.pixel_points):
                cv2.circle(display, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (int(pt[0])+10, int(pt[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(self.pixel_points) < len(self.common_keypoints):
                current_kp = self.common_keypoints[len(self.pixel_points)][0]
                cv2.putText(display, f"Click: {current_kp}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.pixel_points = []
                self.world_points = []
            elif key == ord('u') and len(self.pixel_points) > 0:
                self.pixel_points.pop()
                self.world_points.pop()

        cv2.destroyAllWindows()
        estimator = HomographyEstimator()
        return estimator.estimate_from_manual_points(self.pixel_points, self.world_points)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pixel_points) < len(self.common_keypoints):
                self.pixel_points.append((float(x), float(y)))
                self.world_points.append(self.common_keypoints[len(self.pixel_points)-1][1])
                print(f"Added point {len(self.pixel_points)}: ({x}, {y})")


class CoordinateTransformer:
    def __init__(self, homography: Optional[np.ndarray] = None):
        self.H = homography
        self.H_inv = None
        if homography is not None:
            self.set_homography(homography)

    def set_homography(self, H: np.ndarray):
        self.H = H
        self.H_inv = np.linalg.inv(H)

    def pixel_to_world(self, pixel_coords: Tuple[float, float]) -> Tuple[float, float]:
        if self.H is None:
            raise ValueError("Homography not set")
        px, py = pixel_coords
        point = np.array([px, py, 1.0])
        world_h = self.H @ point
        return (float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2]))

    def world_to_pixel(self, world_coords: Tuple[float, float]) -> Tuple[float, float]:
        if self.H_inv is None:
            raise ValueError("Homography not set")
        wx, wy = world_coords
        point = np.array([wx, wy, 1.0])
        pixel_h = self.H_inv @ point
        return (float(pixel_h[0] / pixel_h[2]), float(pixel_h[1] / pixel_h[2]))
```

### src/metrics/physical.py

```python
"""Physical performance metrics calculation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


@dataclass
class PhysicalMetrics:
    track_id: int
    total_distance: float
    max_speed: float
    avg_speed: float
    sprint_count: int
    sprint_distance: float
    high_intensity_distance: float
    max_acceleration: float
    max_deceleration: float


@dataclass
class FrameMetrics:
    position: Tuple[float, float]
    speed: float
    acceleration: float
    is_sprinting: bool
    is_high_intensity: bool


class PhysicalMetricsCalculator:
    SPRINT_THRESHOLD = 25.0
    HIGH_INTENSITY_THRESHOLD = 19.8
    MIN_SPRINT_DURATION = 1.0

    def __init__(self, fps: int = 25, smoothing_window: int = 5):
        self.fps = fps
        self.dt = 1.0 / fps
        self.smoothing_window = smoothing_window

    def calculate_frame_metrics(self, positions: List[Tuple[float, float]]) -> List[FrameMetrics]:
        if len(positions) < 2:
            return []

        positions = np.array(positions)
        velocities = np.diff(positions, axis=0) / self.dt
        speeds_mps = np.linalg.norm(velocities, axis=1)

        window = min(self.smoothing_window, len(speeds_mps))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            speeds_mps = savgol_filter(speeds_mps, window, 2)

        speeds_kmh = speeds_mps * 3.6
        accelerations = np.concatenate([[0], np.diff(speeds_mps) / self.dt])
        if len(accelerations) >= self.smoothing_window:
            accelerations = uniform_filter1d(accelerations, self.smoothing_window)

        metrics = []
        for i in range(len(positions)):
            speed = speeds_kmh[min(i, len(speeds_kmh)-1)] if i > 0 else (speeds_kmh[0] if len(speeds_kmh) > 0 else 0)
            accel = accelerations[min(i, len(accelerations)-1)]
            metrics.append(FrameMetrics(
                position=tuple(positions[i]),
                speed=speed,
                acceleration=accel,
                is_sprinting=speed >= self.SPRINT_THRESHOLD,
                is_high_intensity=speed >= self.HIGH_INTENSITY_THRESHOLD
            ))
        return metrics

    def calculate_match_metrics(self, positions: List[Tuple[float, float]], track_id: int) -> PhysicalMetrics:
        if len(positions) < 2:
            return PhysicalMetrics(track_id, 0, 0, 0, 0, 0, 0, 0, 0)

        positions_arr = np.array(positions)
        distances = np.linalg.norm(np.diff(positions_arr, axis=0), axis=1)
        total_distance = np.sum(distances)

        frame_metrics = self.calculate_frame_metrics(positions)
        speeds = [m.speed for m in frame_metrics]
        max_speed = max(speeds) if speeds else 0
        avg_speed = np.mean(speeds) if speeds else 0

        sprint_count = self._count_sprints(frame_metrics)
        sprint_distance = sum(distances[i] for i, m in enumerate(frame_metrics[:-1]) if m.is_sprinting)
        high_intensity_distance = sum(distances[i] for i, m in enumerate(frame_metrics[:-1]) if m.is_high_intensity)

        accels = [m.acceleration for m in frame_metrics]
        max_acceleration = max(accels) if accels else 0
        max_deceleration = abs(min(accels)) if accels else 0

        return PhysicalMetrics(
            track_id=track_id,
            total_distance=total_distance,
            max_speed=max_speed,
            avg_speed=avg_speed,
            sprint_count=sprint_count,
            sprint_distance=sprint_distance,
            high_intensity_distance=high_intensity_distance,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration
        )

    def _count_sprints(self, frame_metrics: List[FrameMetrics]) -> int:
        min_frames = int(self.MIN_SPRINT_DURATION * self.fps)
        count = 0
        in_sprint = False
        sprint_start = 0

        for i, m in enumerate(frame_metrics):
            if m.is_sprinting and not in_sprint:
                in_sprint = True
                sprint_start = i
            elif not m.is_sprinting and in_sprint:
                if i - sprint_start >= min_frames:
                    count += 1
                in_sprint = False
        return count


class KalmanSmoother:
    def __init__(self, dt: float = 0.04, process_noise: float = 0.1, measurement_noise: float = 1.0):
        self.dt = dt
        self.state = np.zeros(4)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 100
        self.initialized = False

    def update(self, measurement) -> Tuple[float, float]:
        if not self.initialized:
            if measurement is not None:
                self.state[0], self.state[1] = measurement
                self.initialized = True
                return measurement
            return (0, 0)

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        if measurement is not None:
            z = np.array(measurement)
            y = z - self.H @ self.state
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.state = self.state + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

        return (self.state[0], self.state[1])

    def reset(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100
        self.initialized = False
```

### src/output/data_export.py

```python
"""Export tracking data to various formats."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger


class TrackingDataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_frame_data(self, frame_data: List[Dict], filename: str, formats: List[str] = ["json", "csv"]):
        if "json" in formats:
            output_path = self.output_dir / f"{filename}.json"
            with open(output_path, 'w') as f:
                json.dump(frame_data, f, indent=2, default=str)
            logger.info(f"Exported JSON: {output_path}")

        if "csv" in formats:
            rows = []
            for frame in frame_data:
                frame_idx = frame.get('frame', 0)
                timestamp = frame.get('timestamp', 0)
                for player in frame.get('players', []):
                    rows.append({
                        'frame': frame_idx, 'timestamp': timestamp,
                        'track_id': player.get('track_id'), 'team': player.get('team'),
                        'jersey_number': player.get('jersey_number'),
                        'x': player.get('x'), 'y': player.get('y'),
                        'speed': player.get('speed'), 'acceleration': player.get('acceleration')
                    })
                ball = frame.get('ball')
                if ball:
                    rows.append({
                        'frame': frame_idx, 'timestamp': timestamp, 'track_id': 'ball',
                        'x': ball.get('x'), 'y': ball.get('y')
                    })
            df = pd.DataFrame(rows)
            output_path = self.output_dir / f"{filename}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Exported CSV: {output_path}")

    def export_metrics(self, metrics: List[Dict], filename: str):
        output_path = self.output_dir / f"{filename}_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        df = pd.DataFrame(metrics)
        df.to_csv(self.output_dir / f"{filename}_metrics.csv", index=False)
        logger.info(f"Exported metrics: {output_path}")


def create_frame_record(frame_idx: int, timestamp: float, players: List[Dict], ball: Optional[Dict]) -> Dict:
    return {"frame": frame_idx, "timestamp": round(timestamp, 3), "players": players, "ball": ball}
```

### src/output/visualizer.py

```python
"""Visualization of tracking data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

try:
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False


class PitchVisualizer:
    COLORS = {'home': '#e63946', 'away': '#457b9d', 'ball': '#f4a261', 'referee': '#2a9d8f'}

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, figsize: Tuple[int, int] = (12, 8)):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.figsize = figsize

    def plot_frame(self, players: List[Dict], ball: Optional[Dict] = None, title: str = "",
                   save_path: Optional[str] = None, show: bool = False):
        if HAS_MPLSOCCER:
            pitch = Pitch(pitch_type='custom', pitch_length=self.pitch_length, pitch_width=self.pitch_width,
                         line_color='white', pitch_color='grass')
            fig, ax = pitch.draw(figsize=self.figsize)

            for player in players:
                x, y = player.get('x', 0), player.get('y', 0)
                team = player.get('team', 'unknown')
                jersey = player.get('jersey_number')
                color = self.COLORS.get(team, '#888888')
                ax.scatter(x, y, c=color, s=200, edgecolors='white', linewidths=2, zorder=10)
                if jersey is not None:
                    ax.annotate(str(jersey), (x, y), ha='center', va='center',
                               fontsize=8, color='white', fontweight='bold', zorder=11)

            if ball:
                ax.scatter(ball['x'], ball['y'], c=self.COLORS['ball'], s=100,
                          edgecolors='black', linewidths=1, zorder=12)

            if title:
                ax.set_title(title, fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)
        else:
            logger.warning("mplsoccer not available")

    def create_animation(self, frame_data: List[Dict], output_path: str, fps: int = 25):
        logger.info(f"Creating animation with {len(frame_data)} frames")
        temp_dir = Path(output_path).parent / "temp_frames"
        temp_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frame_data):
            self.plot_frame(players=frame.get('players', []), ball=frame.get('ball'),
                           title=f"Frame {frame.get('frame', i)}",
                           save_path=str(temp_dir / f"frame_{i:06d}.png"))

        frame_files = sorted(temp_dir.glob("frame_*.png"))
        if len(frame_files) == 0:
            return

        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame_file in frame_files:
            out.write(cv2.imread(str(frame_file)))
        out.release()

        for f in frame_files:
            f.unlink()
        temp_dir.rmdir()
        logger.info(f"Animation saved: {output_path}")
```

### src/main.py

```python
"""Main tracking pipeline."""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from .detection.player_detector import PlayerDetector
from .detection.ball_detector import BallDetector
from .detection.team_classifier import TeamClassifier, Team
from .tracking.tracker import PlayerTracker
from .homography.calibration import HomographyEstimator, InteractiveCalibrator, CoordinateTransformer
from .metrics.physical import PhysicalMetricsCalculator, KalmanSmoother
from .output.data_export import TrackingDataExporter, create_frame_record
from .output.visualizer import PitchVisualizer


class GreenPitchMaskDetector:
    def __init__(self, lower_green=(35, 50, 50), upper_green=(85, 255, 255)):
        self.lower_green = np.array(lower_green)
        self.upper_green = np.array(upper_green)

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask


class FootballTracker:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        pitch_template_path = Path(config_path).parent / "pitch_template.yaml"
        with open(pitch_template_path, 'r') as f:
            self.pitch_template = yaml.safe_load(f)

        self.device = self.config['processing']['device']
        self.fps = self.config['processing']['frame_rate']

        self._init_components()
        logger.info("Football tracker initialized")

    def _init_components(self):
        det_config = self.config['detection']
        self.player_detector = PlayerDetector(
            model_path=det_config['player']['model'],
            confidence=det_config['player']['confidence'],
            min_height=det_config['player']['min_height'],
            max_height=det_config['player']['max_height'],
            device=self.device
        )
        self.ball_detector = BallDetector(
            model_path=det_config['ball']['model'],
            confidence=det_config['ball']['confidence'],
            temporal_window=det_config['ball']['temporal_window'],
            device=self.device
        )

        team_config = self.config['team_classification']
        self.team_classifier = TeamClassifier(n_clusters=team_config['n_clusters'], color_space=team_config['color_space'])
        self.pitch_mask_detector = GreenPitchMaskDetector()

        track_config = self.config['tracking']
        self.player_tracker = PlayerTracker(
            track_thresh=track_config['track_high_thresh'],
            track_buffer=track_config['track_buffer'],
            match_thresh=track_config['match_thresh'],
            frame_rate=self.fps
        )

        self.interactive_calibrator = InteractiveCalibrator(self.pitch_template)
        self.transformer = CoordinateTransformer()

        self.metrics_calculator = PhysicalMetricsCalculator(fps=self.fps, smoothing_window=self.config['metrics']['speed_smoothing'])
        self.kalman_filters: Dict[int, KalmanSmoother] = {}
        self.position_history: Dict[int, List[tuple]] = {}

        self.exporter = TrackingDataExporter(self.config['video']['output_path'])
        self.visualizer = PitchVisualizer()

    def calibrate(self, video_path: str, interactive: bool = True):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        if interactive:
            logger.info("Starting interactive calibration...")
            result = self.interactive_calibrator.calibrate_interactive(frame)
        else:
            result = HomographyEstimator().estimate_from_manual_points([], [])

        if result.is_valid:
            self.transformer.set_homography(result.homography)
            logger.info(f"Calibration successful. Error: {result.reprojection_error:.2f}m")
        else:
            logger.error("Calibration failed!")

    def process_video(self, video_path: str, output_name: Optional[str] = None,
                     start_frame: int = 0, end_frame: Optional[int] = None, visualize: bool = False) -> List[Dict]:
        video_path = Path(video_path)
        output_name = output_name or video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames

        logger.info(f"Processing {video_path.name}: frames {start_frame}-{end_frame}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        if self.transformer.H is None:
            self.calibrate(str(video_path), interactive=True)

        pitch_mask = self.pitch_mask_detector.get_mask(first_frame)
        initial_detections = self.player_detector.detect(first_frame, pitch_mask)
        if len(initial_detections) >= 6:
            bboxes = [d.bbox for d in initial_detections]
            self.team_classifier.fit(first_frame, bboxes)
            logger.info("Team classifier fitted")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_records = []
        pbar = tqdm(total=end_frame - start_frame, desc="Processing")

        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            record = self._process_frame(frame, frame_idx)
            frame_records.append(record)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        metrics = self._calculate_final_metrics()
        self.exporter.export_frame_data(frame_records, output_name, formats=self.config['output']['format'])
        self.exporter.export_metrics(metrics, output_name)

        if visualize:
            self._generate_visualizations(frame_records, output_name)

        logger.info(f"Processing complete. {len(frame_records)} frames processed.")
        return frame_records

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        timestamp = frame_idx / self.fps
        pitch_mask = self.pitch_mask_detector.get_mask(frame)
        player_detections = self.player_detector.detect(frame, pitch_mask)
        player_bboxes = [d.bbox for d in player_detections]
        ball_detection = self.ball_detector.detect(frame, player_bboxes)
        tracks = self.player_tracker.update(player_detections, frame)

        player_records = []
        for track in tracks:
            pixel_pos = track.foot_position
            world_pos = self.transformer.pixel_to_world(pixel_pos)

            if track.track_id not in self.kalman_filters:
                self.kalman_filters[track.track_id] = KalmanSmoother(dt=1/self.fps)
            smoothed_pos = self.kalman_filters[track.track_id].update(world_pos)

            if track.track_id not in self.position_history:
                self.position_history[track.track_id] = []
            self.position_history[track.track_id].append(smoothed_pos)

            team_result = self.team_classifier.classify(frame, track.bbox)
            team_name = team_result.team.name.lower()

            positions = self.position_history[track.track_id]
            speed, accel = 0, 0
            if len(positions) >= 2:
                frame_metrics = self.metrics_calculator.calculate_frame_metrics(positions[-min(10, len(positions)):])
                if frame_metrics:
                    speed, accel = frame_metrics[-1].speed, frame_metrics[-1].acceleration

            player_records.append({
                'track_id': track.track_id, 'team': team_name,
                'jersey_number': None,
                'x': round(smoothed_pos[0], 2), 'y': round(smoothed_pos[1], 2),
                'speed': round(speed, 1), 'acceleration': round(accel, 2)
            })

        ball_record = None
        if ball_detection:
            ball_world = self.transformer.pixel_to_world(ball_detection.position)
            ball_record = {'x': round(ball_world[0], 2), 'y': round(ball_world[1], 2),
                          'is_interpolated': ball_detection.is_interpolated}

        return create_frame_record(frame_idx, timestamp, player_records, ball_record)

    def _calculate_final_metrics(self) -> List[Dict]:
        metrics = []
        for track_id, positions in self.position_history.items():
            if len(positions) < 10:
                continue
            track_metrics = self.metrics_calculator.calculate_match_metrics(positions, track_id)
            metrics.append({
                'track_id': track_id,
                'total_distance_m': round(track_metrics.total_distance, 1),
                'max_speed_kmh': round(track_metrics.max_speed, 1),
                'avg_speed_kmh': round(track_metrics.avg_speed, 1),
                'sprint_count': track_metrics.sprint_count,
                'sprint_distance_m': round(track_metrics.sprint_distance, 1),
                'high_intensity_distance_m': round(track_metrics.high_intensity_distance, 1),
                'max_acceleration_ms2': round(track_metrics.max_acceleration, 2),
                'max_deceleration_ms2': round(track_metrics.max_deceleration, 2)
            })
        return metrics

    def _generate_visualizations(self, frame_records: List[Dict], output_name: str):
        output_dir = Path(self.config['video']['output_path'])
        sample_indices = [0, len(frame_records)//4, len(frame_records)//2, 3*len(frame_records)//4, len(frame_records)-1]
        for idx in sample_indices:
            if idx < len(frame_records):
                record = frame_records[idx]
                self.visualizer.plot_frame(
                    players=record['players'], ball=record.get('ball'),
                    title=f"Frame {record['frame']}",
                    save_path=str(output_dir / f"{output_name}_frame_{idx}.png")
                )

        animation_records = frame_records[::5] if len(frame_records) > 100 else frame_records
        self.visualizer.create_animation(animation_records, str(output_dir / f"{output_name}_animation.mp4"), fps=5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Football XY Tracking")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--config", default="config/config.yaml", help="Config file")
    parser.add_argument("--output", help="Output name")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, help="End frame")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()

    tracker = FootballTracker(args.config)
    tracker.process_video(args.video, output_name=args.output, start_frame=args.start,
                         end_frame=args.end, visualize=args.visualize)


if __name__ == "__main__":
    main()
```

---

## Quick Start

```bash
# 1. Setup
mkdir football-tracker && cd football-tracker
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download YOLOv8 model
mkdir models
wget -O models/yolov8x.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# 3. Create project structure and copy code files

# 4. Place video in data/input/

# 5. Run
python -m src.main data/input/match.mp4 --config config/config.yaml --visualize
```

---

## Calibration Instructions

When interactive calibration opens:
1. Click pitch keypoints in order (corners, center spots, etc.)
2. Press 'q' when done (minimum 4 points)
3. Press 'u' to undo, 'r' to reset

Recommended points: 4 corners, center circle intersections, penalty spots.

---

## Output Format

### Frame Data (JSON)
```json
{
  "frame": 1234,
  "timestamp": 49.36,
  "players": [
    {"track_id": 7, "team": "home", "jersey_number": 10, "x": 45.2, "y": 23.1, "speed": 18.5, "acceleration": 2.1}
  ],
  "ball": {"x": 48.5, "y": 27.3, "is_interpolated": false}
}
```

### Metrics Summary
```json
{
  "track_id": 7,
  "total_distance_m": 10234.5,
  "max_speed_kmh": 32.4,
  "avg_speed_kmh": 7.2,
  "sprint_count": 45,
  "sprint_distance_m": 1856.2,
  "high_intensity_distance_m": 2341.8
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or use half_precision: true |
| Poor tracking | Increase track_buffer, lower confidence |
| Bad homography | Use more keypoints accurately |
| Missed ball | Lower ball confidence, increase temporal_window |
| Wrong teams | Adjust color_space (try HSV or RGB) |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1660 (6GB) | RTX 3080+ (10GB+) |
| RAM | 16GB | 32GB+ |
| Storage | 500GB SSD | 2TB+ |
