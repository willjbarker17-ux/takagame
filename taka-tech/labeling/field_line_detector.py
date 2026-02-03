#!/usr/bin/env python3
"""
Automatic Field Line Detection for Football Pitch

Detects white field lines on green grass using color segmentation
and Hough line transform. Matches detected lines to known field
template to compute camera homography.

This enables GT labeling even when corners are off-screen by
using visible field markings (penalty box, center line, etc.)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from loguru import logger


# Field template dimensions (in arbitrary units, typically meters * 10)
FIELD_WIDTH = 1050  # 105m
FIELD_HEIGHT = 680  # 68m

# Known field line positions (start, end points in field coords)
FIELD_LINES = {
    # Boundary lines
    'sideline_top': [(0, 0), (FIELD_WIDTH, 0)],
    'sideline_bottom': [(0, FIELD_HEIGHT), (FIELD_WIDTH, FIELD_HEIGHT)],
    'goalline_left': [(0, 0), (0, FIELD_HEIGHT)],
    'goalline_right': [(FIELD_WIDTH, 0), (FIELD_WIDTH, FIELD_HEIGHT)],

    # Center line
    'center_line': [(FIELD_WIDTH/2, 0), (FIELD_WIDTH/2, FIELD_HEIGHT)],

    # Left penalty box (16.5m from goal line, 40.3m wide)
    'penalty_left_top': [(0, 138), (165, 138)],
    'penalty_left_bottom': [(0, 542), (165, 542)],
    'penalty_left_side': [(165, 138), (165, 542)],

    # Right penalty box
    'penalty_right_top': [(FIELD_WIDTH, 138), (FIELD_WIDTH-165, 138)],
    'penalty_right_bottom': [(FIELD_WIDTH, 542), (FIELD_WIDTH-165, 542)],
    'penalty_right_side': [(FIELD_WIDTH-165, 138), (FIELD_WIDTH-165, 542)],

    # Left goal area (5.5m from goal line, 18.3m wide)
    'goal_area_left_top': [(0, 248), (55, 248)],
    'goal_area_left_bottom': [(0, 432), (55, 432)],
    'goal_area_left_side': [(55, 248), (55, 432)],

    # Right goal area
    'goal_area_right_top': [(FIELD_WIDTH, 248), (FIELD_WIDTH-55, 248)],
    'goal_area_right_bottom': [(FIELD_WIDTH, 432), (FIELD_WIDTH-55, 432)],
    'goal_area_right_side': [(FIELD_WIDTH-55, 248), (FIELD_WIDTH-55, 432)],
}

# Key points on the field (intersections of lines)
FIELD_POINTS = {
    'corner_tl': (0, 0),
    'corner_tr': (FIELD_WIDTH, 0),
    'corner_bl': (0, FIELD_HEIGHT),
    'corner_br': (FIELD_WIDTH, FIELD_HEIGHT),
    'center_top': (FIELD_WIDTH/2, 0),
    'center_bottom': (FIELD_WIDTH/2, FIELD_HEIGHT),
    'center_mid': (FIELD_WIDTH/2, FIELD_HEIGHT/2),
    'penalty_left_top': (165, 138),
    'penalty_left_bottom': (165, 542),
    'penalty_right_top': (FIELD_WIDTH-165, 138),
    'penalty_right_bottom': (FIELD_WIDTH-165, 542),
}


@dataclass
class DetectedLine:
    """A detected line in image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float  # Angle in degrees (0=horizontal, 90=vertical)
    length: float

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def is_horizontal(self) -> bool:
        return abs(self.angle) < 30 or abs(self.angle) > 150

    @property
    def is_vertical(self) -> bool:
        return 60 < abs(self.angle) < 120


class FieldLineDetector:
    """Detects and matches field lines in football video frames."""

    def __init__(
        self,
        green_hue_range: Tuple[int, int] = (35, 85),
        green_sat_min: int = 30,
        green_val_min: int = 30,
        white_sat_max: int = 60,
        white_val_min: int = 160,
        min_line_length: int = 50,
        max_line_gap: int = 20,
        hough_threshold: int = 50
    ):
        """Initialize detector with color thresholds.

        Args:
            green_hue_range: HSV hue range for green grass
            green_sat_min: Minimum saturation for green
            green_val_min: Minimum value for green
            white_sat_max: Maximum saturation for white lines
            white_val_min: Minimum value for white lines
            min_line_length: Minimum line length in pixels
            max_line_gap: Maximum gap to connect line segments
            hough_threshold: Hough transform threshold
        """
        self.green_hue_range = green_hue_range
        self.green_sat_min = green_sat_min
        self.green_val_min = green_val_min
        self.white_sat_max = white_sat_max
        self.white_val_min = white_val_min
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.hough_threshold = hough_threshold

    def detect_grass_mask(self, frame: np.ndarray) -> np.ndarray:
        """Detect green grass regions.

        Args:
            frame: BGR image

        Returns:
            Binary mask of grass regions
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([self.green_hue_range[0], self.green_sat_min, self.green_val_min])
        upper_green = np.array([self.green_hue_range[1], 255, 255])

        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)

        return grass_mask

    def detect_line_mask(self, frame: np.ndarray, grass_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Detect white field lines.

        Args:
            frame: BGR image
            grass_mask: Optional grass mask to constrain line detection

        Returns:
            Binary mask of detected lines
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White detection: low saturation, high value
        lower_white = np.array([0, 0, self.white_val_min])
        upper_white = np.array([180, self.white_sat_max, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Constrain to grass regions (dilated to include lines on edges)
        if grass_mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            grass_dilated = cv2.dilate(grass_mask, kernel, iterations=2)
            white_mask = cv2.bitwise_and(white_mask, grass_dilated)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        return white_mask

    def detect_lines(self, frame: np.ndarray) -> List[DetectedLine]:
        """Detect field lines in a frame.

        Args:
            frame: BGR image

        Returns:
            List of detected lines
        """
        h, w = frame.shape[:2]

        # Get masks
        grass_mask = self.detect_grass_mask(frame)
        line_mask = self.detect_line_mask(frame, grass_mask)

        # Edge detection
        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return []

        detected = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle and length
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            length = np.sqrt(dx*dx + dy*dy)

            detected.append(DetectedLine(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                angle=angle,
                length=length
            ))

        # Merge nearby parallel lines
        detected = self._merge_lines(detected)

        return detected

    def _merge_lines(self, lines: List[DetectedLine], angle_thresh: float = 10, dist_thresh: float = 30) -> List[DetectedLine]:
        """Merge nearby parallel lines into single lines."""
        if len(lines) < 2:
            return lines

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            # Find lines to merge with
            to_merge = [line1]

            for j, line2 in enumerate(lines):
                if j <= i or j in used:
                    continue

                # Check if parallel (similar angle)
                angle_diff = abs(line1.angle - line2.angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff > angle_thresh:
                    continue

                # Check if close (midpoints within threshold)
                mid1 = line1.midpoint
                mid2 = line2.midpoint
                dist = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)

                if dist < dist_thresh:
                    to_merge.append(line2)
                    used.add(j)

            # Merge all lines in group
            if len(to_merge) == 1:
                merged.append(line1)
            else:
                # Find extreme points
                all_points = []
                for line in to_merge:
                    all_points.extend([(line.x1, line.y1), (line.x2, line.y2)])

                # Fit line to all points and take extremes
                points = np.array(all_points)

                # Use PCA to find line direction
                mean = points.mean(axis=0)
                centered = points - mean
                _, _, Vt = np.linalg.svd(centered)
                direction = Vt[0]

                # Project points onto line direction
                projections = centered @ direction
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)

                x1, y1 = all_points[min_idx]
                x2, y2 = all_points[max_idx]

                dx = x2 - x1
                dy = y2 - y1

                merged.append(DetectedLine(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    angle=np.degrees(np.arctan2(dy, dx)),
                    length=np.sqrt(dx*dx + dy*dy)
                ))

            used.add(i)

        return merged

    def find_line_intersections(self, lines: List[DetectedLine]) -> List[Tuple[float, float]]:
        """Find intersections between detected lines.

        Args:
            lines: List of detected lines

        Returns:
            List of intersection points (x, y)
        """
        intersections = []

        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if j <= i:
                    continue

                # Skip parallel lines
                angle_diff = abs(line1.angle - line2.angle)
                if angle_diff < 20 or angle_diff > 160:
                    continue

                # Compute intersection
                pt = self._line_intersection(
                    (line1.x1, line1.y1), (line1.x2, line1.y2),
                    (line2.x1, line2.y1), (line2.x2, line2.y2)
                )

                if pt is not None:
                    intersections.append(pt)

        return intersections

    def _line_intersection(
        self,
        p1: Tuple[float, float], p2: Tuple[float, float],
        p3: Tuple[float, float], p4: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Compute intersection of two lines."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom

        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)

        return (x, y)

    def match_lines_to_template(
        self,
        lines: List[DetectedLine],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, DetectedLine]:
        """Match detected lines to field template lines.

        Uses position and orientation to identify which field line
        each detected line corresponds to.

        Args:
            lines: Detected lines
            frame_shape: (height, width) of frame

        Returns:
            Dict mapping template line names to detected lines
        """
        h, w = frame_shape
        matches = {}

        # Separate horizontal and vertical lines
        horizontal = [l for l in lines if l.is_horizontal]
        vertical = [l for l in lines if l.is_vertical]

        # Sort by position
        horizontal.sort(key=lambda l: l.midpoint[1])  # Top to bottom
        vertical.sort(key=lambda l: l.midpoint[0])    # Left to right

        # Match horizontal lines
        for line in horizontal:
            mid_y = line.midpoint[1]
            rel_y = mid_y / h  # Relative position (0=top, 1=bottom)

            # Classify based on position
            if rel_y < 0.3:
                # Upper part - could be sideline_top or penalty box top
                if line.length > w * 0.5:
                    matches['sideline_top'] = line
                else:
                    # Shorter line - penalty box
                    mid_x = line.midpoint[0]
                    if mid_x < w * 0.5:
                        matches['penalty_left_top'] = line
                    else:
                        matches['penalty_right_top'] = line
            elif rel_y > 0.7:
                # Lower part
                if line.length > w * 0.5:
                    matches['sideline_bottom'] = line
                else:
                    mid_x = line.midpoint[0]
                    if mid_x < w * 0.5:
                        matches['penalty_left_bottom'] = line
                    else:
                        matches['penalty_right_bottom'] = line
            else:
                # Middle - could be goal area lines
                mid_x = line.midpoint[0]
                if mid_x < w * 0.3:
                    if rel_y < 0.5:
                        matches['goal_area_left_top'] = line
                    else:
                        matches['goal_area_left_bottom'] = line
                elif mid_x > w * 0.7:
                    if rel_y < 0.5:
                        matches['goal_area_right_top'] = line
                    else:
                        matches['goal_area_right_bottom'] = line

        # Match vertical lines
        for line in vertical:
            mid_x = line.midpoint[0]
            rel_x = mid_x / w  # Relative position (0=left, 1=right)

            if 0.45 < rel_x < 0.55:
                # Center line
                matches['center_line'] = line
            elif rel_x < 0.3:
                # Left side - goal line or penalty box side
                if line.length > h * 0.7:
                    matches['goalline_left'] = line
                else:
                    matches['penalty_left_side'] = line
            elif rel_x > 0.7:
                # Right side
                if line.length > h * 0.7:
                    matches['goalline_right'] = line
                else:
                    matches['penalty_right_side'] = line

        return matches

    def compute_homography_from_lines(
        self,
        matched_lines: Dict[str, DetectedLine],
        frame_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Compute homography from matched field lines.

        Uses line intersections as point correspondences.

        Args:
            matched_lines: Dict mapping template line names to detected lines
            frame_shape: (height, width) of frame

        Returns:
            3x3 homography matrix (field -> image) or None if not enough matches
        """
        # Find point correspondences from line intersections
        image_points = []
        field_points = []

        # Define which line pairs create which field points
        intersection_pairs = [
            (('sideline_top', 'goalline_left'), 'corner_tl'),
            (('sideline_top', 'goalline_right'), 'corner_tr'),
            (('sideline_bottom', 'goalline_left'), 'corner_bl'),
            (('sideline_bottom', 'goalline_right'), 'corner_br'),
            (('sideline_top', 'center_line'), 'center_top'),
            (('sideline_bottom', 'center_line'), 'center_bottom'),
            (('penalty_left_top', 'penalty_left_side'), 'penalty_left_top'),
            (('penalty_left_bottom', 'penalty_left_side'), 'penalty_left_bottom'),
            (('penalty_right_top', 'penalty_right_side'), 'penalty_right_top'),
            (('penalty_right_bottom', 'penalty_right_side'), 'penalty_right_bottom'),
        ]

        for (line1_name, line2_name), point_name in intersection_pairs:
            if line1_name in matched_lines and line2_name in matched_lines:
                line1 = matched_lines[line1_name]
                line2 = matched_lines[line2_name]

                pt = self._line_intersection(
                    (line1.x1, line1.y1), (line1.x2, line1.y2),
                    (line2.x1, line2.y1), (line2.x2, line2.y2)
                )

                if pt is not None and point_name in FIELD_POINTS:
                    image_points.append(pt)
                    field_points.append(FIELD_POINTS[point_name])

        # Also use line endpoints that are clearly on field boundaries
        for line_name, line in matched_lines.items():
            if line_name in FIELD_LINES:
                template_line = FIELD_LINES[line_name]
                # Add endpoints if they seem to be on the field edge
                # This is a heuristic - endpoints near image edge likely correspond to field edge
                h, w = frame_shape

                for img_pt, field_pt in [
                    ((line.x1, line.y1), template_line[0]),
                    ((line.x2, line.y2), template_line[1])
                ]:
                    # Check if point is near image boundary
                    near_edge = (img_pt[0] < 20 or img_pt[0] > w-20 or
                                img_pt[1] < 20 or img_pt[1] > h-20)

                    if near_edge:
                        image_points.append(img_pt)
                        field_points.append(field_pt)

        if len(image_points) < 4:
            logger.debug(f"Not enough point correspondences: {len(image_points)}")
            return None

        # Compute homography
        src = np.array(field_points, dtype=np.float32)
        dst = np.array(image_points, dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        return H

    def detect_and_compute_homography(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, DetectedLine], List[DetectedLine]]:
        """Full pipeline: detect lines, match to template, compute homography.

        Args:
            frame: BGR image

        Returns:
            H: Homography matrix (field -> image) or None
            matched: Dict of matched lines
            all_lines: All detected lines
        """
        h, w = frame.shape[:2]

        # Detect lines
        lines = self.detect_lines(frame)
        logger.debug(f"Detected {len(lines)} lines")

        if len(lines) < 3:
            return None, {}, lines

        # Match to template
        matched = self.match_lines_to_template(lines, (h, w))
        logger.debug(f"Matched {len(matched)} lines to template")

        # Compute homography
        H = self.compute_homography_from_lines(matched, (h, w))

        return H, matched, lines

    def get_field_corners_from_homography(
        self,
        H: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Project field corners using homography.

        Args:
            H: Homography matrix (field -> image)

        Returns:
            Dict mapping corner names to image coordinates
        """
        corners = {}

        for name, field_pt in [
            ('corner_tl', (0, 0)),
            ('corner_tr', (FIELD_WIDTH, 0)),
            ('corner_bl', (0, FIELD_HEIGHT)),
            ('corner_br', (FIELD_WIDTH, FIELD_HEIGHT)),
        ]:
            pt = np.array([[field_pt]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)[0][0]
            corners[name] = (float(img_pt[0]), float(img_pt[1]))

        return corners

    def visualize(
        self,
        frame: np.ndarray,
        lines: List[DetectedLine],
        matched: Dict[str, DetectedLine],
        H: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Visualize detected lines and matches.

        Args:
            frame: BGR image
            lines: All detected lines
            matched: Matched lines
            H: Optional homography for projecting field template

        Returns:
            Annotated image
        """
        vis = frame.copy()

        # Draw all detected lines (gray)
        for line in lines:
            cv2.line(vis, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)),
                    (128, 128, 128), 1)

        # Draw matched lines (colored by type)
        colors = {
            'sideline': (0, 255, 0),     # Green
            'goalline': (255, 0, 0),     # Blue
            'center': (0, 255, 255),     # Yellow
            'penalty': (255, 0, 255),    # Magenta
            'goal_area': (255, 128, 0),  # Orange
        }

        for name, line in matched.items():
            color = (0, 255, 0)  # Default green
            for key, c in colors.items():
                if key in name:
                    color = c
                    break

            cv2.line(vis, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)),
                    color, 2)

            # Label
            mid = line.midpoint
            cv2.putText(vis, name[:10], (int(mid[0]), int(mid[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Project field template if homography available
        if H is not None:
            corners = self.get_field_corners_from_homography(H)
            pts = np.array([
                corners['corner_tl'],
                corners['corner_tr'],
                corners['corner_br'],
                corners['corner_bl']
            ], dtype=np.int32)

            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)

            for name, pt in corners.items():
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        return vis


# Global detector instance
_detector: Optional[FieldLineDetector] = None


def get_field_line_detector() -> FieldLineDetector:
    """Get or create global field line detector."""
    global _detector
    if _detector is None:
        _detector = FieldLineDetector()
    return _detector


def detect_field_from_lines(frame: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
    """Convenience function to detect field corners from frame.

    Args:
        frame: BGR image

    Returns:
        Dict mapping corner names to image coordinates, or None if detection failed
    """
    detector = get_field_line_detector()
    H, matched, lines = detector.detect_and_compute_homography(frame)

    if H is None:
        return None

    return detector.get_field_corners_from_homography(H)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test field line detection")
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to analyze')
    parser.add_argument('--show-mask', action='store_true', help='Show detection masks')
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Failed to read frame {args.frame}")
            exit(1)

        detector = FieldLineDetector()

        if args.show_mask:
            grass = detector.detect_grass_mask(frame)
            lines_mask = detector.detect_line_mask(frame, grass)

            cv2.imshow('Grass Mask', grass)
            cv2.imshow('Line Mask', lines_mask)

        H, matched, lines = detector.detect_and_compute_homography(frame)

        print(f"Detected {len(lines)} lines")
        print(f"Matched {len(matched)} lines:")
        for name, line in matched.items():
            print(f"  {name}: length={line.length:.0f}, angle={line.angle:.1f}")

        if H is not None:
            corners = detector.get_field_corners_from_homography(H)
            print("\nProjected corners:")
            for name, pt in corners.items():
                print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
        else:
            print("\nCould not compute homography")

        vis = detector.visualize(frame, lines, matched, H)
        cv2.imshow('Field Line Detection', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Usage: python field_line_detector.py --video <path> [--frame N] [--show-mask]")
