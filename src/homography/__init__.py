"""Homography and calibration modules."""

from .calibration import (
    HomographyEstimator,
    InteractiveCalibrator,
    CoordinateTransformer,
    DynamicCoordinateTransformer,
    CalibrationResult,
)
from .pitch_detector import (
    PitchLineDetector,
    PitchKeypoint,
    PitchDetectionResult,
    FeatureBasedPitchMatcher,
)
from .rotation_handler import (
    RotationHandler,
    AdaptiveHomographyManager,
    CameraState,
    RotationState,
    DynamicHomography,
)

__all__ = [
    # Calibration
    "HomographyEstimator",
    "InteractiveCalibrator",
    "CoordinateTransformer",
    "DynamicCoordinateTransformer",
    "CalibrationResult",
    # Pitch detection
    "PitchLineDetector",
    "PitchKeypoint",
    "PitchDetectionResult",
    "FeatureBasedPitchMatcher",
    # Rotation handling
    "RotationHandler",
    "AdaptiveHomographyManager",
    "CameraState",
    "RotationState",
    "DynamicHomography",
]
