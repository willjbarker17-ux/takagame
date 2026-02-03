#!/usr/bin/env python3
"""
Camera Motion Tracker for Static Field Tracking

This module provides camera motion estimation specifically designed for
football field tracking where:
- The field is STATIC (doesn't move)
- Only the CAMERA moves (pan, tilt, zoom)
- Players and ball are moving objects to ignore

Key approach:
1. Mask to green field regions (ignore players, crowd, etc.)
2. Use ORB feature matching (robust to viewpoint changes)
3. Prioritize white line features (field markings are stable)
4. Estimate global homography from matched features
5. Apply Kalman filtering for smooth camera motion
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


class CameraMotionTracker:
    """Estimates camera motion for static field tracking.

    Unlike optical flow which tracks pixel movement, this estimates
    the camera's motion as a homography transformation, then applies
    that transformation to move all field annotations together.
    """

    def __init__(
        self,
        use_field_mask: bool = True,
        max_features: int = 1000,
        match_threshold: float = 0.75,  # Lowe's ratio test
        ransac_threshold: float = 5.0,
        min_matches: int = 10,
        kalman_enabled: bool = True
    ):
        """Initialize camera motion tracker.

        Args:
            use_field_mask: Mask to green field regions
            max_features: Maximum ORB features to detect
            match_threshold: Lowe's ratio for good matches
            ransac_threshold: RANSAC reprojection threshold
            min_matches: Minimum matches required
            kalman_enabled: Use Kalman filter for smoothing
        """
        self.use_field_mask = use_field_mask
        self.match_threshold = match_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches

        # ORB feature detector (fast and rotation-invariant)
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31
        )

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Kalman filter for homography smoothing
        self.kalman_enabled = kalman_enabled
        if kalman_enabled:
            self.kalman = HomographyKalmanFilter()
        else:
            self.kalman = None

        # State
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.cumulative_H = np.eye(3, dtype=np.float32)
        self.frame_count = 0

    def reset(self):
        """Reset tracker state (call when switching videos)."""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.cumulative_H = np.eye(3, dtype=np.float32)
        self.frame_count = 0
        if self.kalman:
            self.kalman.reset()
        logger.debug("Camera motion tracker reset")

    def create_field_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create mask for green field regions.

        This masks out:
        - Players (non-green)
        - Crowd/stands (non-green)
        - Advertisements (varied colors)

        Keeps:
        - Green grass
        - White field lines (within green regions)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green grass detection (wide range for different lighting)
        # Hue: 35-85 (green range)
        # Saturation: 30-255 (not too gray)
        # Value: 30-255 (not too dark)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Also include white lines within/near green regions
        # White: low saturation, high value
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Dilate green mask to include nearby white lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        green_dilated = cv2.dilate(green_mask, kernel, iterations=2)

        # White lines that are near green (field lines, not ads)
        white_field = cv2.bitwise_and(white_mask, green_dilated)

        # Combine green grass + white field lines
        field_mask = cv2.bitwise_or(green_mask, white_field)

        # Clean up with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_small)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel_small)

        return field_mask

    def detect_features(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray]:
        """Detect ORB features in frame.

        Args:
            frame: BGR image
            mask: Optional mask (detect only in white regions)

        Returns:
            keypoints: List of cv2.KeyPoint
            descriptors: Feature descriptors
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

        return keypoints, descriptors, gray

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match features using Lowe's ratio test.

        Args:
            desc1: Descriptors from frame 1
            desc2: Descriptors from frame 2

        Returns:
            good_matches: List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        # KNN matching with k=2 for ratio test
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_camera_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Estimate camera motion between two frames.

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)

        Returns:
            H: Homography matrix (frame1 -> frame2), or None if failed
            info: Dictionary with debug info
        """
        info = {
            'num_features1': 0,
            'num_features2': 0,
            'num_matches': 0,
            'num_inliers': 0,
            'method': 'camera_motion',
            'success': False
        }

        # Create field masks if enabled
        mask1 = self.create_field_mask(frame1) if self.use_field_mask else None
        mask2 = self.create_field_mask(frame2) if self.use_field_mask else None

        # Detect features
        kp1, desc1, gray1 = self.detect_features(frame1, mask1)
        kp2, desc2, gray2 = self.detect_features(frame2, mask2)

        info['num_features1'] = len(kp1)
        info['num_features2'] = len(kp2)

        if len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            logger.debug(f"Not enough features: {len(kp1)}, {len(kp2)}")
            return None, info

        # Match features
        matches = self.match_features(desc1, desc2)
        info['num_matches'] = len(matches)

        if len(matches) < self.min_matches:
            logger.debug(f"Not enough matches: {len(matches)}")
            return None, info

        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.ransac_threshold
        )

        if H is None:
            logger.debug("Homography estimation failed")
            return None, info

        num_inliers = int(np.sum(mask)) if mask is not None else 0
        info['num_inliers'] = num_inliers

        if num_inliers < self.min_matches:
            logger.debug(f"Not enough inliers: {num_inliers}")
            return None, info

        # Apply Kalman filtering for smooth motion
        if self.kalman:
            H = self.kalman.update(H)

        info['success'] = True
        return H, info

    def track_points(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        points: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], Dict]:
        """Track points from frame1 to frame2 using camera motion.

        All points are transformed together using the estimated homography,
        maintaining their geometric relationships (unlike optical flow).

        Args:
            frame1: Source frame
            frame2: Target frame
            points: List of (x, y) points to track

        Returns:
            tracked_points: List of tracked (x, y) points
            info: Debug information
        """
        if not points:
            return [], {'success': False, 'reason': 'no_points'}

        # Estimate camera motion
        H, info = self.estimate_camera_motion(frame1, frame2)

        if H is None:
            # Fallback: return original points (no motion detected)
            logger.warning("Camera motion estimation failed, keeping points static")
            return points, info

        # Transform all points using homography
        pts_array = np.float32(points).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts_array, H)
        tracked = [(float(p[0][0]), float(p[0][1])) for p in transformed]

        self.frame_count += 1
        return tracked, info

    def propagate_annotations(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        annotations: List[dict]
    ) -> Tuple[List[dict], Dict]:
        """Propagate all annotations from frame1 to frame2.

        This is designed for template/field annotations where all points
        should move together according to camera motion.

        Args:
            frame1: Source frame
            frame2: Target frame
            annotations: List of annotation dictionaries

        Returns:
            propagated: List of propagated annotations
            info: Debug information
        """
        if not annotations:
            return [], {'success': False, 'reason': 'no_annotations'}

        # Estimate camera motion once for all annotations
        H, info = self.estimate_camera_motion(frame1, frame2)

        if H is None:
            # Fallback: return original annotations
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
                cx, cy = ann['center']
                rx, ry = ann['axes']
                angle = ann.get('angle', 0)

                # Transform center and edge points
                edge_points = []
                for t in [0, 90, 180, 270]:
                    rad = np.radians(t + angle)
                    x = cx + rx * np.cos(rad)
                    y = cy + ry * np.sin(rad)
                    edge_points.append([x, y])

                all_pts = [[cx, cy]] + edge_points
                pts_array = np.float32(all_pts).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts_array, H)

                new_center = transformed[0][0]
                new_ann['center'] = [float(new_center[0]), float(new_center[1])]

                # Recompute axes from transformed edge points
                edges = transformed[1:]
                if len(edges) >= 4:
                    new_rx = (abs(edges[0][0][0] - new_center[0]) +
                             abs(edges[2][0][0] - new_center[0])) / 2
                    new_ry = (abs(edges[1][0][1] - new_center[1]) +
                             abs(edges[3][0][1] - new_center[1])) / 2
                    new_ann['axes'] = [max(float(new_rx), 5), max(float(new_ry), 5)]

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
            'kalman_enabled': self.kalman_enabled,
            'use_field_mask': self.use_field_mask,
            'method': 'camera_motion_orb'
        }


class HomographyKalmanFilter:
    """Kalman filter for smooth homography estimation.

    Filters the 8 homography parameters to reduce jitter.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2
    ):
        self.dim = 8

        # State: 8 homography parameters
        self.state = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

        # Covariance
        self.P = np.eye(self.dim, dtype=np.float32)

        # Process noise
        self.Q = np.eye(self.dim, dtype=np.float32) * process_noise

        # Measurement noise
        self.R = np.eye(self.dim, dtype=np.float32) * measurement_noise

        self.initialized = False

    def reset(self):
        """Reset filter state."""
        self.state = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(self.dim, dtype=np.float32)
        self.initialized = False

    def update(self, H: np.ndarray) -> np.ndarray:
        """Update filter with new homography measurement.

        Args:
            H: 3x3 homography matrix

        Returns:
            Filtered 3x3 homography matrix
        """
        # Normalize and extract parameters
        H_norm = H / (H[2, 2] + 1e-10)
        z = H_norm.flatten()[:8]

        if not self.initialized:
            self.state = z
            self.initialized = True
            return H

        # Predict (assume constant motion)
        # State prediction: x_pred = x (identity transition)
        # Covariance prediction
        self.P = self.P + self.Q

        # Update
        # Kalman gain
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # State update
        innovation = z - self.state
        self.state = self.state + K @ innovation

        # Covariance update
        self.P = (np.eye(self.dim) - K) @ self.P

        # Reconstruct homography
        H_filtered = np.zeros((3, 3), dtype=np.float32)
        H_filtered.flat[:8] = self.state
        H_filtered[2, 2] = 1.0

        return H_filtered


# Global instance
_camera_tracker: Optional[CameraMotionTracker] = None


def get_camera_tracker(force_new: bool = False) -> CameraMotionTracker:
    """Get or create global camera motion tracker."""
    global _camera_tracker

    if _camera_tracker is None or force_new:
        _camera_tracker = CameraMotionTracker(
            use_field_mask=True,
            kalman_enabled=True
        )

    return _camera_tracker


def reset_camera_tracker():
    """Reset the global camera tracker."""
    global _camera_tracker
    if _camera_tracker is not None:
        _camera_tracker.reset()


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test camera motion tracker")
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=100, help='End frame')
    parser.add_argument('--show-mask', action='store_true', help='Show field mask')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        exit(1)

    tracker = CameraMotionTracker()

    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read first frame")
        exit(1)

    h, w = frame1.shape[:2]

    # Create test points (field corners)
    test_points = [
        (w * 0.2, h * 0.3),
        (w * 0.8, h * 0.3),
        (w * 0.9, h * 0.9),
        (w * 0.1, h * 0.9),
    ]

    current_points = test_points

    for frame_num in range(args.start + 1, args.end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame2 = cap.read()
        if not ret:
            break

        # Track points
        tracked, info = tracker.track_points(frame1, frame2, current_points)

        # Visualize
        vis = frame2.copy()

        if args.show_mask:
            mask = tracker.create_field_mask(frame2)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        # Draw tracked points as quadrilateral
        pts = np.array(tracked, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        for i, pt in enumerate(tracked):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
            cv2.putText(vis, str(i), (int(pt[0])+10, int(pt[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show info
        status = f"Frame {frame_num} | Matches: {info.get('num_matches', 0)} | Inliers: {info.get('num_inliers', 0)}"
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Camera Motion Tracking', vis)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            current_points = test_points

        frame1 = frame2
        current_points = tracked

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nFinal stats: {tracker.get_stats()}")
