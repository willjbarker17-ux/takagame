#!/usr/bin/env python3
"""
Camera Motion Tracker V2 - SOTA method for static field tracking

Key improvements:
1. SIFT features (scale-invariant, better for zoom)
2. Decompose homography into rotation, scale, translation
3. Separate smoothing for each component
4. Geometric validation (reject unrealistic transformations)
5. Proper zoom handling - template scales with camera zoom
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from collections import deque


class CameraMotionTrackerV2:
    """SOTA camera motion estimation for static field tracking.

    Designed for broadcast cameras that:
    - Pan left/right (rotation around vertical axis)
    - Tilt up/down (rotation around horizontal axis)
    - Zoom in/out (scale change)

    The field is STATIC - only camera moves.
    """

    def __init__(
        self,
        use_field_mask: bool = True,
        smoothing_window: int = 5,
        min_matches: int = 20,
        max_rotation_per_frame: float = 5.0,  # degrees
        max_zoom_per_frame: float = 0.1,  # 10% zoom change
        max_translation_per_frame: float = 100.0,  # pixels
    ):
        self.use_field_mask = use_field_mask
        self.smoothing_window = smoothing_window
        self.min_matches = min_matches
        self.max_rotation = np.radians(max_rotation_per_frame)
        self.max_zoom = max_zoom_per_frame
        self.max_translation = max_translation_per_frame

        # Use SIFT for better scale invariance (important for zoom)
        self.sift = cv2.SIFT_create(
            nfeatures=2000,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )

        # FLANN matcher for SIFT
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Motion history for smoothing
        self.rotation_history = deque(maxlen=smoothing_window)
        self.scale_history = deque(maxlen=smoothing_window)
        self.tx_history = deque(maxlen=smoothing_window)
        self.ty_history = deque(maxlen=smoothing_window)

        # Previous frame data
        self.prev_frame = None
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None

        # Cumulative transform (for long-range tracking)
        self.cumulative_rotation = 0.0
        self.cumulative_scale = 1.0
        self.cumulative_tx = 0.0
        self.cumulative_ty = 0.0

        self.frame_count = 0

    def reset(self):
        """Reset tracker state."""
        self.rotation_history.clear()
        self.scale_history.clear()
        self.tx_history.clear()
        self.ty_history.clear()

        self.prev_frame = None
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None

        self.cumulative_rotation = 0.0
        self.cumulative_scale = 1.0
        self.cumulative_tx = 0.0
        self.cumulative_ty = 0.0

        self.frame_count = 0

    def create_field_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create mask for green field regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green grass
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # White lines
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Include white near green (field lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        green_dilated = cv2.dilate(green_mask, kernel, iterations=2)
        white_field = cv2.bitwise_and(white_mask, green_dilated)

        # Combine
        field_mask = cv2.bitwise_or(green_mask, white_field)

        # Clean up
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_small)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel_small)

        return field_mask

    def detect_features(self, frame: np.ndarray, mask: Optional[np.ndarray] = None):
        """Detect SIFT features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kp, desc = self.sift.detectAndCompute(gray, mask)
        return kp, desc, gray

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features using Lowe's ratio test."""
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        return good

    def decompose_homography(self, H: np.ndarray, img_center: Tuple[float, float]) -> Dict:
        """Decompose homography into rotation, scale, translation.

        For a camera rotating and zooming on a distant planar surface (field),
        the transformation can be approximated as:
        - Rotation around image center
        - Uniform scale (zoom)
        - Translation (pan)
        """
        cx, cy = img_center

        # Move to image center for decomposition
        T_to_center = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ], dtype=np.float64)

        T_from_center = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # H in centered coordinates
        H_centered = T_to_center @ H @ T_from_center
        H_centered = H_centered / H_centered[2, 2]

        # Extract 2x2 linear part
        A = H_centered[:2, :2]

        # SVD decomposition: A = U * S * V^T
        # For rotation + scale: A = R * s
        U, S, Vt = np.linalg.svd(A)

        # Rotation matrix (ensure proper rotation, not reflection)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Rotation angle
        rotation = np.arctan2(R[1, 0], R[0, 0])

        # Scale (average of singular values)
        scale = np.mean(S)

        # Translation
        tx = H_centered[0, 2]
        ty = H_centered[1, 2]

        return {
            'rotation': rotation,
            'scale': scale,
            'tx': tx,
            'ty': ty,
            'rotation_deg': np.degrees(rotation),
            'scale_percent': (scale - 1) * 100
        }

    def compose_homography(
        self,
        rotation: float,
        scale: float,
        tx: float,
        ty: float,
        img_center: Tuple[float, float]
    ) -> np.ndarray:
        """Compose homography from rotation, scale, translation."""
        cx, cy = img_center

        cos_r, sin_r = np.cos(rotation), np.sin(rotation)

        # Rotation + scale matrix
        RS = np.array([
            [scale * cos_r, -scale * sin_r, 0],
            [scale * sin_r, scale * cos_r, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        # Translate to origin, apply RS, translate back, then add translation
        T_to = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        T_from = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)

        H = T_from @ RS @ T_to
        return H.astype(np.float32)

    def validate_transform(self, decomp: Dict) -> bool:
        """Validate that transform is physically plausible."""
        # Check rotation
        if abs(decomp['rotation']) > self.max_rotation:
            return False

        # Check zoom (scale should be close to 1)
        if abs(decomp['scale'] - 1.0) > self.max_zoom:
            return False

        # Check translation
        if abs(decomp['tx']) > self.max_translation or abs(decomp['ty']) > self.max_translation:
            return False

        return True

    def smooth_components(self, rotation: float, scale: float, tx: float, ty: float) -> Tuple:
        """Apply exponential moving average smoothing to motion components."""
        # Add to history
        self.rotation_history.append(rotation)
        self.scale_history.append(scale)
        self.tx_history.append(tx)
        self.ty_history.append(ty)

        # Weighted average (more recent = more weight)
        def weighted_avg(history):
            if len(history) == 0:
                return 0
            weights = np.exp(np.linspace(-1, 0, len(history)))
            weights /= weights.sum()
            return np.average(list(history), weights=weights)

        smooth_rotation = weighted_avg(self.rotation_history)
        smooth_scale = weighted_avg(self.scale_history)
        smooth_tx = weighted_avg(self.tx_history)
        smooth_ty = weighted_avg(self.ty_history)

        return smooth_rotation, smooth_scale, smooth_tx, smooth_ty

    def estimate_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Estimate camera motion between frames.

        Returns:
            H: Homography matrix (frame1 -> frame2)
            info: Debug information including rotation/zoom/pan
        """
        h, w = frame1.shape[:2]
        img_center = (w / 2, h / 2)

        info = {
            'success': False,
            'method': 'sift_decomposed',
            'num_matches': 0,
            'num_inliers': 0,
            'rotation_deg': 0,
            'zoom_percent': 0,
            'tx': 0,
            'ty': 0
        }

        # Create masks
        mask1 = self.create_field_mask(frame1) if self.use_field_mask else None
        mask2 = self.create_field_mask(frame2) if self.use_field_mask else None

        # Detect features
        kp1, desc1, gray1 = self.detect_features(frame1, mask1)
        kp2, desc2, gray2 = self.detect_features(frame2, mask2)

        if len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return None, info

        # Match features
        matches = self.match_features(desc1, desc2)
        info['num_matches'] = len(matches)

        if len(matches) < self.min_matches:
            return None, info

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if H is None:
            return None, info

        num_inliers = int(np.sum(mask)) if mask is not None else 0
        info['num_inliers'] = num_inliers

        if num_inliers < self.min_matches // 2:
            return None, info

        # Decompose homography
        decomp = self.decompose_homography(H, img_center)

        # Validate (reject implausible transformations)
        if not self.validate_transform(decomp):
            # Return identity if implausible
            return np.eye(3, dtype=np.float32), info

        # Smooth components
        smooth_rot, smooth_scale, smooth_tx, smooth_ty = self.smooth_components(
            decomp['rotation'], decomp['scale'], decomp['tx'], decomp['ty']
        )

        # Recompose smoothed homography
        H_smooth = self.compose_homography(smooth_rot, smooth_scale, smooth_tx, smooth_ty, img_center)

        # Update info
        info['success'] = True
        info['rotation_deg'] = np.degrees(smooth_rot)
        info['zoom_percent'] = (smooth_scale - 1) * 100
        info['tx'] = smooth_tx
        info['ty'] = smooth_ty
        info['raw_rotation_deg'] = decomp['rotation_deg']
        info['raw_zoom_percent'] = decomp['scale_percent']

        # Update cumulative transform
        self.cumulative_rotation += smooth_rot
        self.cumulative_scale *= smooth_scale
        self.cumulative_tx += smooth_tx
        self.cumulative_ty += smooth_ty

        self.frame_count += 1

        return H_smooth, info

    def track_points(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        points: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], Dict]:
        """Track points from frame1 to frame2."""
        if not points:
            return [], {'success': False}

        H, info = self.estimate_motion(frame1, frame2)

        if H is None:
            return points, info

        pts_array = np.float32(points).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts_array, H)
        tracked = [(float(p[0][0]), float(p[0][1])) for p in transformed]

        return tracked, info

    def propagate_annotations(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        annotations: List[dict]
    ) -> Tuple[List[dict], Dict]:
        """Propagate annotations with proper zoom handling."""
        if not annotations:
            return [], {'success': False}

        H, info = self.estimate_motion(frame1, frame2)

        if H is None:
            return [ann.copy() for ann in annotations], info

        propagated = []

        for ann in annotations:
            new_ann = ann.copy()
            new_ann['isGT'] = False

            if ann['type'] == 'line':
                points = ann['points']
                pts_array = np.float32([[p[0], p[1]] for p in points]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts_array, H)
                new_ann['points'] = [[float(p[0][0]), float(p[0][1])] for p in transformed]

            elif ann['type'] == 'ellipse':
                # Transform ellipse properly with scale
                cx, cy = ann['center']
                rx, ry = ann['axes']
                angle = ann.get('angle', 0)

                # Sample points on ellipse
                num_samples = 16
                sample_pts = []
                for i in range(num_samples):
                    t = 2 * np.pi * i / num_samples
                    rad = np.radians(angle)
                    x = cx + rx * np.cos(t) * np.cos(rad) - ry * np.sin(t) * np.sin(rad)
                    y = cy + rx * np.cos(t) * np.sin(rad) + ry * np.sin(t) * np.cos(rad)
                    sample_pts.append([x, y])

                # Transform samples
                pts_array = np.float32([[cx, cy]] + sample_pts).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts_array, H)

                new_center = transformed[0][0]
                new_ann['center'] = [float(new_center[0]), float(new_center[1])]

                # Fit ellipse to transformed points
                trans_pts = transformed[1:].reshape(-1, 2)
                if len(trans_pts) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(trans_pts.astype(np.float32))
                        new_ann['center'] = [float(ellipse[0][0]), float(ellipse[0][1])]
                        new_ann['axes'] = [float(ellipse[1][0]/2), float(ellipse[1][1]/2)]
                        new_ann['angle'] = float(ellipse[2])
                    except:
                        # Fallback: simple scale
                        scale = info.get('zoom_percent', 0) / 100 + 1
                        new_ann['axes'] = [rx * scale, ry * scale]

            elif ann['type'] == 'point':
                point = ann['point']
                pts_array = np.float32([[point[0], point[1]]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts_array, H)
                new_ann['point'] = [float(transformed[0][0][0]), float(transformed[0][0][1])]

            propagated.append(new_ann)

        return propagated, info

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            'frame_count': self.frame_count,
            'cumulative_rotation_deg': np.degrees(self.cumulative_rotation),
            'cumulative_zoom_percent': (self.cumulative_scale - 1) * 100,
            'cumulative_tx': self.cumulative_tx,
            'cumulative_ty': self.cumulative_ty,
            'method': 'sift_decomposed_v2'
        }


# Global instance
_tracker_v2: Optional[CameraMotionTrackerV2] = None


def get_camera_tracker_v2(force_new: bool = False) -> CameraMotionTrackerV2:
    """Get or create global camera motion tracker v2."""
    global _tracker_v2
    if _tracker_v2 is None or force_new:
        _tracker_v2 = CameraMotionTrackerV2()
    return _tracker_v2


def reset_camera_tracker_v2():
    """Reset the global tracker."""
    global _tracker_v2
    if _tracker_v2 is not None:
        _tracker_v2.reset()
