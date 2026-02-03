"""
SoccerNet Feature Extractor for Template Prediction

Uses pretrained DeepLabv3 weights trained on soccer pitch line segmentation.
The ResNet50 backbone features are much better for matching camera views
than generic ImageNet features.

Usage:
    extractor = SoccerNetFeatureExtractor()
    features = extractor.extract(frame)  # Returns 2048-dim feature vector

    db = FeatureDatabase('path/to/db.npz')
    db.add(features, homography_matrix)
    best_match_homography = db.find_similar(query_features)
"""

import os
import json
import numpy as np
from pathlib import Path

# Lazy imports for torch to avoid startup cost
_torch = None
_models = None
_transforms = None
_extractor_model = None

SOCCERNET_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "soccernet" / "soccernet_deeplabv3.pth"
FEATURE_DB_PATH = Path(__file__).parent / "feature_database.npz"

# Field template coordinates for homography computation
FIELD_TEMPLATE = {
    'corner_tl': (0, 0),
    'corner_tr': (105, 0),
    'corner_bl': (0, 68),
    'corner_br': (105, 68),
    'center_top': (52.5, 0),
    'center_mid': (52.5, 34),
    'center_bottom': (52.5, 68),
    'penalty_left_goal_top': (0, 13.84),
    'penalty_left_top': (16.5, 13.84),
    'penalty_left_bottom': (16.5, 54.16),
    'penalty_left_goal_bottom': (0, 54.16),
    'penalty_right_goal_top': (105, 13.84),
    'penalty_right_top': (88.5, 13.84),
    'penalty_right_bottom': (88.5, 54.16),
    'penalty_right_goal_bottom': (105, 54.16),
    'goal_area_left_goal_top': (0, 24.84),
    'goal_area_left_top': (5.5, 24.84),
    'goal_area_left_bottom': (5.5, 43.16),
    'goal_area_left_goal_bottom': (0, 43.16),
    'goal_area_right_goal_top': (105, 24.84),
    'goal_area_right_top': (99.5, 24.84),
    'goal_area_right_bottom': (99.5, 43.16),
    'goal_area_right_goal_bottom': (105, 43.16),
}


def _lazy_import():
    """Lazy import torch modules to avoid slow startup."""
    global _torch, _models, _transforms
    if _torch is None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        _torch = torch
        _models = models
        _transforms = transforms
    return _torch, _models, _transforms


def _load_extractor():
    """Load and cache the feature extractor model."""
    global _extractor_model
    if _extractor_model is not None:
        return _extractor_model

    torch, models, transforms = _lazy_import()

    # Check if SoccerNet weights exist
    if not SOCCERNET_WEIGHTS_PATH.exists():
        print(f"Warning: SoccerNet weights not found at {SOCCERNET_WEIGHTS_PATH}")
        print("Using ImageNet pretrained ResNet50 instead (less accurate)")
        resnet = models.resnet50(weights='IMAGENET1K_V1')
    else:
        # Load SoccerNet checkpoint
        checkpoint = torch.load(SOCCERNET_WEIGHTS_PATH, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']

        # Create ResNet50 and load backbone weights
        resnet = models.resnet50(weights=None)
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.backbone.'):
                new_key = k.replace('module.backbone.', '')
                if not new_key.startswith('fc'):
                    backbone_state[new_key] = v
        resnet.load_state_dict(backbone_state, strict=False)
        print("Loaded SoccerNet pretrained features")

    # Create feature extractor (remove avgpool and fc)
    class FeatureExtractor(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            self.pool = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            features = self.backbone(x)
            pooled = self.pool(features)
            return pooled.flatten(1)

    _extractor_model = FeatureExtractor(resnet)
    _extractor_model.eval()
    return _extractor_model


class SoccerNetFeatureExtractor:
    """Extract features from frames using SoccerNet pretrained backbone."""

    def __init__(self):
        self._model = None
        self._transform = None

    def _ensure_loaded(self):
        if self._model is None:
            torch, models, transforms = _lazy_import()
            self._model = _load_extractor()
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract 2048-dimensional feature vector from frame.

        Args:
            frame: BGR or RGB numpy array (H, W, 3)

        Returns:
            Feature vector of shape (2048,)
        """
        self._ensure_loaded()
        torch, _, _ = _lazy_import()

        # Convert BGR to RGB if needed (OpenCV default is BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume it's in the right format
            pass

        img_tensor = self._transform(frame).unsqueeze(0)

        with torch.no_grad():
            features = self._model(img_tensor)

        return features.numpy().flatten()


class FeatureDatabase:
    """Database for storing and retrieving similar frames based on features."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else FEATURE_DB_PATH
        self.features = []  # List of 2048-dim vectors
        self.homographies = []  # List of 3x3 matrices
        self.metadata = []  # List of dicts with video_id, frame_num, etc.
        self._load()

    def _load(self):
        """Load existing database if it exists."""
        if self.db_path.exists():
            try:
                data = np.load(self.db_path, allow_pickle=True)
                self.features = list(data['features'])
                self.homographies = list(data['homographies'])
                self.metadata = list(data['metadata'])
                print(f"Loaded feature database with {len(self.features)} entries")
            except Exception as e:
                print(f"Error loading feature database: {e}")
                self.features = []
                self.homographies = []
                self.metadata = []

    def save(self):
        """Save database to disk."""
        if not self.features:
            return

        np.savez(
            self.db_path,
            features=np.array(self.features),
            homographies=np.array(self.homographies),
            metadata=np.array(self.metadata, dtype=object)
        )
        print(f"Saved feature database with {len(self.features)} entries")

    def add(self, features: np.ndarray, homography: np.ndarray,
            video_id: str = None, frame_num: int = None):
        """Add a new entry to the database.

        Args:
            features: 2048-dim feature vector
            homography: 3x3 homography matrix (image -> field)
            video_id: Optional video identifier
            frame_num: Optional frame number
        """
        self.features.append(features)
        self.homographies.append(homography)
        self.metadata.append({
            'video_id': video_id,
            'frame_num': frame_num
        })
        self.save()

    def find_similar(self, query_features: np.ndarray, k: int = 1,
                     exclude_video: str = None) -> list:
        """Find k most similar entries in the database.

        Args:
            query_features: 2048-dim feature vector
            k: Number of results to return
            exclude_video: Optional video ID to exclude (for leave-one-out)

        Returns:
            List of (homography, similarity, metadata) tuples
        """
        if not self.features:
            return []

        from numpy.linalg import norm

        results = []
        for i, (feat, H, meta) in enumerate(zip(self.features, self.homographies, self.metadata)):
            # Skip if same video
            if exclude_video and meta.get('video_id') == exclude_video:
                continue

            # Compute cosine similarity
            sim = np.dot(query_features, feat) / (norm(query_features) * norm(feat) + 1e-8)
            results.append((H, sim, meta))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def size(self) -> int:
        """Return number of entries in database."""
        return len(self.features)


def compute_homography_from_annotations(annotations: list) -> np.ndarray:
    """Compute homography from annotation point correspondences.

    Args:
        annotations: List of annotation dicts with 'points' and 'templatePoints'

    Returns:
        3x3 homography matrix (image -> field), or None if insufficient points
    """
    import cv2

    img_pts = []
    field_pts = []

    for ann in annotations:
        if ann.get('isTemplate') and 'templatePoints' in ann:
            points = ann['points']
            template_points = ann['templatePoints']

            for i, tp_name in enumerate(template_points):
                if i < len(points) and tp_name in FIELD_TEMPLATE:
                    img_pts.append(points[i])
                    field_pts.append(FIELD_TEMPLATE[tp_name])

    if len(img_pts) < 4:
        return None

    img_pts = np.array(img_pts, dtype=np.float32)
    field_pts = np.array(field_pts, dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, field_pts, cv2.RANSAC, 5.0)
    return H


def build_database_from_annotations(annotations_dir: str, video_cache_dir: str):
    """Build feature database from existing annotations.

    Includes ALL GT frames, not just frame 0. This maximizes the training data
    from your annotations.

    Args:
        annotations_dir: Path to directory with annotation JSON files
        video_cache_dir: Path to directory with video files
    """
    import cv2
    from glob import glob

    extractor = SoccerNetFeatureExtractor()
    db = FeatureDatabase()

    annotations_dir = Path(annotations_dir)
    video_cache_dir = Path(video_cache_dir)

    added_count = 0

    for ann_file in glob(str(annotations_dir / "*.json")):
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)

        video_name = ann_data.get('video', Path(ann_file).stem)
        frames = ann_data.get('frames', {})

        if not frames:
            continue

        # Find video file
        video_path = video_cache_dir / video_name
        if not video_path.exists():
            video_path = video_cache_dir / (video_name + '.mp4')
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        # Process ALL frames with GT annotations
        for frame_key, frame_anns in frames.items():
            # Check if any annotation has isGT=true and is a template annotation
            has_gt_template = any(
                ann.get('isGT') and ann.get('isTemplate')
                for ann in frame_anns
            )

            if not has_gt_template:
                continue

            # Compute homography from annotations
            H = compute_homography_from_annotations(frame_anns)
            if H is None:
                continue

            # Check if already in database
            frame_num = int(frame_key)
            already_exists = any(
                m.get('video_id') == video_name and m.get('frame_num') == frame_num
                for m in db.metadata
            )
            if already_exists:
                continue

            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            # Extract features and add to database
            features = extractor.extract(frame)
            db.add(features, H, video_id=video_name, frame_num=frame_num)
            added_count += 1
            print(f"Added: {video_name} frame {frame_key}")

        cap.release()

    print(f"\nAdded {added_count} new entries. Database now has {db.size()} entries")


# Singleton instances for use in app
_feature_extractor = None
_feature_database = None


def get_feature_extractor():
    """Get singleton feature extractor."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = SoccerNetFeatureExtractor()
    return _feature_extractor


def get_feature_database():
    """Get singleton feature database."""
    global _feature_database
    if _feature_database is None:
        _feature_database = FeatureDatabase()
    return _feature_database


def _is_valid_homography(H, img_width=1280, img_height=720):
    """Check if homography is not completely degenerate.

    Note: Bottom corners being 2-3x outside the frame is NORMAL for angled
    soccer camera views. Only filter out truly broken homographies.
    """
    try:
        # Check determinant - should be non-zero and not extreme
        det = np.linalg.det(H)
        if abs(det) < 1e-10 or abs(det) > 1e10:
            return False

        H_inv = np.linalg.inv(H)

        # Just check that corners don't map to infinity or NaN
        corners_field = [(0, 0), (105, 0), (0, 68), (105, 68)]
        for fx, fy in corners_field:
            pt = np.array([fx, fy, 1.0])
            result = H_inv @ pt
            if abs(result[2]) < 1e-8:
                return False
            ix, iy = result[:2] / result[2]

            # Only reject truly extreme values (>50x frame size = clearly broken)
            if abs(ix) > img_width * 50 or abs(iy) > img_height * 50:
                return False

            if not np.isfinite(ix) or not np.isfinite(iy):
                return False

        return True
    except:
        return False


def predict_template(frame: np.ndarray, exclude_video: str = None) -> dict:
    """Predict initial homography for a frame using feature similarity.

    Uses best single match (not averaging, which causes shrinkage toward center).
    Filters out degenerate homographies.

    Args:
        frame: BGR numpy array
        exclude_video: Optional video ID to exclude from matching

    Returns:
        Dict with 'homography' (3x3 list), 'confidence' (float),
        'similar_video' (str), or None if no matches
    """
    extractor = get_feature_extractor()
    db = get_feature_database()

    if db.size() == 0:
        return None

    # Extract features
    features = extractor.extract(frame)

    from numpy.linalg import norm
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

    # Find best match with valid homography
    # Prefer frame 0 entries (more stable camera positions)
    best_frame0 = None
    best_frame0_sim = -1
    best_other = None
    best_other_sim = -1

    for i, (db_feat, H, meta) in enumerate(zip(db.features, db.homographies, db.metadata)):
        if exclude_video and meta.get('video_id') == exclude_video:
            continue

        # Skip invalid homographies
        if not _is_valid_homography(H):
            continue

        sim = cosine_sim(features, db_feat)

        if meta.get('frame_num', 0) == 0:
            if sim > best_frame0_sim:
                best_frame0_sim = sim
                best_frame0 = (H, sim, meta)
        else:
            if sim > best_other_sim:
                best_other_sim = sim
                best_other = (H, sim, meta)

    # Prefer frame 0 match if similarity is reasonable, else use best other
    if best_frame0 and best_frame0_sim > 0.5:
        best_match = best_frame0
    elif best_other:
        best_match = best_other
    elif best_frame0:
        best_match = best_frame0
    else:
        return None

    H, similarity, meta = best_match

    return {
        'homography': H.tolist(),
        'confidence': float(similarity),
        'similar_video': meta.get('video_id'),
        'similar_frame': meta.get('frame_num')
    }


if __name__ == '__main__':
    # Build database from existing annotations
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        build_database_from_annotations(
            '/Users/Will/Downloads/football/labeling/annotations',
            '/Users/Will/Downloads/football/labeling/video_cache'
        )
    else:
        print("Usage: python soccernet_features.py build")
        print("  Build feature database from existing annotations")
