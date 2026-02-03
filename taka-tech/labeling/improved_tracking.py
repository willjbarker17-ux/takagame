#!/usr/bin/env python3
"""
Improved Tracking Module for Labeling App

Replaces basic Lucas-Kanade optical flow with:
1. Homography-constrained tracking
2. Kalman filtering for smoothness
3. Learned drift correction (when model available)

This module provides drop-in replacement functions for the labeling app.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

# Add parent src directory to path - import tvcalib_model directly to avoid torch dependencies
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'homography'))

try:
    # Import directly from the module file to avoid __init__.py which requires torch
    from tvcalib_model import (
        HomographyTracker,
        HomographyConstrainedFlow,
        HomographyKalmanFilter,
        transform_points,
        compute_homography_from_points
    )
    IMPROVED_TRACKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Improved tracking not available: {e}")
    IMPROVED_TRACKING_AVAILABLE = False

# Camera motion tracker for static field tracking (V2 with proper zoom/rotation)
try:
    from camera_motion_tracker_v2 import CameraMotionTrackerV2, get_camera_tracker_v2, reset_camera_tracker_v2
    get_camera_tracker = get_camera_tracker_v2
    reset_camera_tracker = reset_camera_tracker_v2
    CAMERA_MOTION_AVAILABLE = True
    logger.info("Using Camera Motion Tracker V2 (SIFT + decomposed rotation/zoom)")
except ImportError as e:
    # Fallback to V1
    try:
        from camera_motion_tracker import CameraMotionTracker, get_camera_tracker, reset_camera_tracker
        CAMERA_MOTION_AVAILABLE = True
        logger.info("Using Camera Motion Tracker V1 (ORB)")
    except ImportError as e2:
        logger.warning(f"Camera motion tracker not available: {e2}")
        CAMERA_MOTION_AVAILABLE = False


# Global tracker instance
_tracker: Optional['ImprovedAnnotationTracker'] = None


class ImprovedAnnotationTracker:
    """Improved annotation tracking with homography constraints.

    Key improvements over basic Lucas-Kanade:
    1. Estimates global homography instead of tracking points independently
    2. Uses Kalman filter for temporal smoothing
    3. Can use learned drift correction model
    4. Tracks points using homography transformation (geometric consistency)
    """

    def __init__(
        self,
        drift_model_path: Optional[str] = None,
        use_kalman: bool = True,
        use_dense_flow: bool = True
    ):
        """Initialize improved tracker.

        Args:
            drift_model_path: Path to trained drift correction model
            use_kalman: Enable Kalman filtering for smooth tracking
            use_dense_flow: Use dense optical flow (more robust but slower)
        """
        self.use_kalman = use_kalman
        self.use_dense_flow = use_dense_flow

        # Core flow tracker with homography constraint
        self.flow_tracker = HomographyConstrainedFlow(
            use_dense_flow=use_dense_flow,
            ransac_threshold=3.0,
            min_inliers=10,
            kalman_enabled=use_kalman
        )

        # Full tracker with drift correction (if model available)
        self.full_tracker = None
        if drift_model_path and Path(drift_model_path).exists():
            try:
                self.full_tracker = HomographyTracker(
                    drift_model_path=drift_model_path,
                    use_kalman=use_kalman
                )
                logger.info(f"Loaded drift correction model from {drift_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load drift model: {e}")

        # State
        self.prev_frame = None
        self.prev_homography = None
        self.frame_count = 0

    def reset(self):
        """Reset tracker state (call when switching videos)."""
        self.flow_tracker.reset()
        if self.full_tracker:
            self.full_tracker.reset()
        self.prev_frame = None
        self.prev_homography = None
        self.frame_count = 0
        logger.debug("Tracker reset")

    def track_points(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Track points from frame1 to frame2 using homography-constrained flow.

        This is a drop-in replacement for track_points_optical_flow().

        Args:
            frame1: Source frame [H, W, 3] BGR
            frame2: Target frame [H, W, 3] BGR
            points: List of (x, y) points to track

        Returns:
            List of tracked (x, y) points
        """
        if not points:
            return []

        points_arr = np.array(points, dtype=np.float32).reshape(-1, 2)

        # Use full tracker if available (has drift correction)
        if self.full_tracker is not None:
            if self.prev_frame is None:
                self.full_tracker.reset()
                self.prev_frame = frame1.copy()

            tracked, confidence, info = self.full_tracker.track(
                frame2, points_arr, apply_drift_correction=True
            )

            self.prev_frame = frame2.copy()
            self.frame_count += 1

            return [(float(p[0]), float(p[1])) for p in tracked]

        # Otherwise use homography-constrained flow
        tracked, confidence, H = self.flow_tracker.track_points(
            frame1, frame2, points_arr, use_homography=True
        )

        if H is not None:
            self.prev_homography = H

        self.prev_frame = frame2.copy()
        self.frame_count += 1

        return [(float(p[0]), float(p[1])) for p in tracked]

    def propagate_annotations(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        annotations: List[dict]
    ) -> List[dict]:
        """Propagate annotations from frame1 to frame2.

        This is a drop-in replacement for propagate_annotations_optical_flow().

        Args:
            frame1: Source frame
            frame2: Target frame
            annotations: List of annotation dictionaries

        Returns:
            List of propagated annotation dictionaries
        """
        if not annotations:
            return []

        propagated = []

        for ann in annotations:
            new_ann = ann.copy()
            new_ann['isGT'] = False

            if ann['type'] == 'line':
                points = ann['points']
                tracked_points = self.track_points(
                    frame1, frame2,
                    [(p[0], p[1]) for p in points]
                )
                new_ann['points'] = [[p[0], p[1]] for p in tracked_points]

            elif ann['type'] == 'ellipse':
                cx, cy = ann['center']
                rx, ry = ann['axes']
                angle = ann.get('angle', 0)

                # Create points around ellipse to track
                edge_points = []
                for t in [0, 90, 180, 270]:
                    rad = np.radians(t + angle)
                    x = cx + rx * np.cos(rad)
                    y = cy + ry * np.sin(rad)
                    edge_points.append((x, y))

                # Track center + edges
                all_points = [(cx, cy)] + edge_points
                tracked = self.track_points(frame1, frame2, all_points)

                # Update ellipse
                new_center = tracked[0]
                new_ann['center'] = [new_center[0], new_center[1]]

                tracked_edges = tracked[1:]
                if len(tracked_edges) >= 4:
                    new_rx = (abs(tracked_edges[0][0] - new_center[0]) +
                             abs(tracked_edges[2][0] - new_center[0])) / 2
                    new_ry = (abs(tracked_edges[1][1] - new_center[1]) +
                             abs(tracked_edges[3][1] - new_center[1])) / 2
                    new_ann['axes'] = [max(new_rx, 5), max(new_ry, 5)]

            elif ann['type'] == 'point':
                point = ann['point']
                tracked = self.track_points(frame1, frame2, [(point[0], point[1])])
                new_ann['point'] = [tracked[0][0], tracked[0][1]]

            propagated.append(new_ann)

        return propagated

    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics for debugging."""
        return {
            'frame_count': self.frame_count,
            'has_drift_model': self.full_tracker is not None,
            'kalman_enabled': self.use_kalman,
            'dense_flow': self.use_dense_flow,
            'prev_homography': self.prev_homography.tolist() if self.prev_homography is not None else None
        }


def get_tracker(
    drift_model_path: Optional[str] = None,
    force_new: bool = False
) -> ImprovedAnnotationTracker:
    """Get or create the global tracker instance.

    Args:
        drift_model_path: Path to drift correction model
        force_new: Force creation of new tracker

    Returns:
        ImprovedAnnotationTracker instance
    """
    global _tracker

    if _tracker is None or force_new:
        # Look for drift model in default locations
        if drift_model_path is None:
            default_paths = [
                # Training output directory
                Path(__file__).parent.parent / 'training' / 'models' / 'drift_correction' / 'drift_correction_best.pth',
                Path(__file__).parent.parent / 'training' / 'models' / 'drift_correction' / 'best.pth',
                # Alternative locations
                Path(__file__).parent.parent / 'models' / 'drift_correction' / 'best.pth',
                Path(__file__).parent / 'models' / 'drift_correction.pth',
            ]
            for p in default_paths:
                if p.exists():
                    drift_model_path = str(p)
                    logger.info(f"Found trained drift model: {p}")
                    break

        _tracker = ImprovedAnnotationTracker(
            drift_model_path=drift_model_path,
            use_kalman=True,
            use_dense_flow=True
        )

    return _tracker


def reset_tracker():
    """Reset the global tracker state."""
    global _tracker
    if _tracker is not None:
        _tracker.reset()


# =============================================================================
# DROP-IN REPLACEMENT FUNCTIONS
# =============================================================================

def track_points_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Drop-in replacement for basic optical flow tracking.

    Uses homography-constrained tracking instead of independent point tracking.
    """
    if not IMPROVED_TRACKING_AVAILABLE:
        # Fallback to basic Lucas-Kanade
        return _basic_lk_tracking(frame1, frame2, points)

    tracker = get_tracker()
    return tracker.track_points(frame1, frame2, points)


def propagate_annotations_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    annotations: List[dict]
) -> List[dict]:
    """Drop-in replacement for basic annotation propagation.

    Uses homography-constrained tracking for better stability.
    """
    if not IMPROVED_TRACKING_AVAILABLE:
        # Fallback to basic propagation
        return _basic_propagation(frame1, frame2, annotations)

    tracker = get_tracker()
    return tracker.propagate_annotations(frame1, frame2, annotations)


def _basic_lk_tracking(
    frame1: np.ndarray,
    frame2: np.ndarray,
    points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Basic Lucas-Kanade tracking (fallback)."""
    if not points:
        return []

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    p1, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    if p1 is None:
        return points

    tracked = []
    for i, (new_pt, st) in enumerate(zip(p1, status)):
        if st[0] == 1:
            tracked.append((float(new_pt[0][0]), float(new_pt[0][1])))
        else:
            tracked.append(points[i])

    return tracked


def _basic_propagation(
    frame1: np.ndarray,
    frame2: np.ndarray,
    annotations: List[dict]
) -> List[dict]:
    """Basic annotation propagation (fallback)."""
    if not annotations:
        return []

    propagated = []

    for ann in annotations:
        new_ann = ann.copy()
        new_ann['isGT'] = False

        if ann['type'] == 'line':
            points = ann['points']
            tracked_points = _basic_lk_tracking(
                frame1, frame2,
                [(p[0], p[1]) for p in points]
            )
            new_ann['points'] = [[p[0], p[1]] for p in tracked_points]

        elif ann['type'] == 'ellipse':
            cx, cy = ann['center']
            rx, ry = ann['axes']
            angle = ann.get('angle', 0)

            edge_points = []
            for t in [0, 90, 180, 270]:
                rad = np.radians(t + angle)
                x = cx + rx * np.cos(rad)
                y = cy + ry * np.sin(rad)
                edge_points.append((x, y))

            all_points = [(cx, cy)] + edge_points
            tracked = _basic_lk_tracking(frame1, frame2, all_points)

            new_center = tracked[0]
            new_ann['center'] = [new_center[0], new_center[1]]

            tracked_edges = tracked[1:]
            if len(tracked_edges) >= 4:
                new_rx = (abs(tracked_edges[0][0] - new_center[0]) +
                         abs(tracked_edges[2][0] - new_center[0])) / 2
                new_ry = (abs(tracked_edges[1][1] - new_center[1]) +
                         abs(tracked_edges[3][1] - new_center[1])) / 2
                new_ann['axes'] = [max(new_rx, 5), max(new_ry, 5)]

        elif ann['type'] == 'point':
            point = ann['point']
            tracked = _basic_lk_tracking(frame1, frame2, [(point[0], point[1])])
            new_ann['point'] = [tracked[0][0], tracked[0][1]]

        propagated.append(new_ann)

    return propagated


# =============================================================================
# CAMERA MOTION TRACKING FOR TEMPLATES
# =============================================================================

def propagate_template_camera_motion(
    frame1: np.ndarray,
    frame2: np.ndarray,
    annotations: List[dict]
) -> Tuple[List[dict], Dict]:
    """Propagate template annotations using camera motion estimation.

    This is specifically designed for STATIC field annotations where:
    - The field doesn't move
    - Only the camera pans/tilts/zooms
    - All template points should move together

    Unlike optical flow which tracks pixels independently, this:
    1. Estimates global camera motion as a homography
    2. Applies the homography to all points at once
    3. Maintains geometric consistency

    Args:
        frame1: Source frame
        frame2: Target frame
        annotations: List of annotation dictionaries

    Returns:
        propagated: List of propagated annotations
        info: Debug information
    """
    if not CAMERA_MOTION_AVAILABLE:
        # Fallback to basic propagation
        logger.debug("Camera motion not available, using basic tracking")
        return _basic_propagation(frame1, frame2, annotations), {'method': 'basic_fallback'}

    tracker = get_camera_tracker()
    return tracker.propagate_annotations(frame1, frame2, annotations)


def track_template_points_camera_motion(
    frame1: np.ndarray,
    frame2: np.ndarray,
    points: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float]], Dict]:
    """Track template points using camera motion estimation.

    All points are transformed together using estimated camera homography.

    Args:
        frame1: Source frame
        frame2: Target frame
        points: List of (x, y) points to track

    Returns:
        tracked_points: List of tracked (x, y) points
        info: Debug information
    """
    if not CAMERA_MOTION_AVAILABLE:
        # Fallback to basic tracking
        return _basic_lk_tracking(frame1, frame2, points), {'method': 'basic_fallback'}

    tracker = get_camera_tracker()
    return tracker.track_points(frame1, frame2, points)


def is_template_annotation(annotations: List[dict]) -> bool:
    """Check if annotations are template/field annotations.

    Template annotations should use camera motion tracking instead of
    optical flow, since the field is static.
    """
    for ann in annotations:
        if ann.get('isTemplate', False):
            return True
        # Also check for common template point names
        if ann.get('templatePoints'):
            return True
    return False


def propagate_annotations_smart(
    frame1: np.ndarray,
    frame2: np.ndarray,
    annotations: List[dict]
) -> List[dict]:
    """Smart annotation propagation that chooses the best method.

    - Template/field annotations: Use camera motion tracking
    - Player/ball annotations: Use optical flow tracking

    Args:
        frame1: Source frame
        frame2: Target frame
        annotations: List of annotation dictionaries

    Returns:
        propagated: List of propagated annotations
    """
    if not annotations:
        return []

    # Separate template vs non-template annotations
    template_anns = [a for a in annotations if a.get('isTemplate', False)]
    other_anns = [a for a in annotations if not a.get('isTemplate', False)]

    propagated = []

    # Template annotations: use camera motion
    if template_anns:
        if CAMERA_MOTION_AVAILABLE:
            prop_template, info = propagate_template_camera_motion(
                frame1, frame2, template_anns
            )
            logger.debug(f"Template tracking: {info.get('num_matches', 0)} matches, {info.get('num_inliers', 0)} inliers")
            propagated.extend(prop_template)
        else:
            propagated.extend(_basic_propagation(frame1, frame2, template_anns))

    # Other annotations: use optical flow
    if other_anns:
        propagated.extend(propagate_annotations_optical_flow(frame1, frame2, other_anns))

    return propagated


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test improved tracking")
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=100, help='End frame')
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)

        # Read first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        ret, frame1 = cap.read()

        if not ret:
            print("Failed to read video")
            sys.exit(1)

        # Define some test points (corners of frame)
        h, w = frame1.shape[:2]
        test_points = [
            (w * 0.25, h * 0.25),
            (w * 0.75, h * 0.25),
            (w * 0.75, h * 0.75),
            (w * 0.25, h * 0.75),
        ]

        tracker = get_tracker()
        current_points = test_points

        for frame_num in range(args.start + 1, args.end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame2 = cap.read()

            if not ret:
                break

            tracked = tracker.track_points(frame1, frame2, current_points)

            # Visualize
            vis_frame = frame2.copy()
            for pt in tracked:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            cv2.imshow('Tracking', vis_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            frame1 = frame2
            current_points = tracked

        cap.release()
        cv2.destroyAllWindows()

        print("Tracking stats:", tracker.get_tracking_stats())
    else:
        print("No video specified. Use --video to test.")
