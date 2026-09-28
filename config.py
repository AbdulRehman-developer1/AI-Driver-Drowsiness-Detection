"""
Configuration and Dataclasses for AI Driver Sleep & Drowsiness Detection System.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# MediaPipe Face Landmarker Landmark Indices (468/478 Mesh Model)

# Left Eye Landmark Indices (6-point EAR calculation & contour)
# P1 (Outer corner), P2 (Top 1), P3 (Top 2), P4 (Inner corner), P5 (Bottom 2), P6 (Bottom 1)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]

# Right Eye Landmark Indices (6-point EAR calculation & contour)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]

# Eyeball / Iris center landmarks (478 model)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Mouth Landmark Indices (MAR calculation & contour)
# Outer horizontal corners: 61 (left), 291 (right)
# Vertical pairs: (13, 14), (81, 178), (311, 402)
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14
MOUTH_INNER_LEFT_TOP = 81
MOUTH_INNER_LEFT_BOTTOM = 178
MOUTH_INNER_RIGHT_TOP = 311
MOUTH_INNER_RIGHT_BOTTOM = 402
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

MOUTH_OUTER_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER_CONTOUR = [78, 191, 80, 13, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Head Pose 3D facial landmarks standard indices
# Nose tip: 1, Chin: 152, Left eye left corner: 33, Right eye right corner: 263, Left mouth: 61, Right mouth: 291
POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


@dataclass
class DrowsinessConfig:
    """Configurable system thresholds and parameters."""
    ear_threshold: float = 0.23          # EAR below this is considered eye closed
    mar_threshold: float = 0.55          # MAR above this is considered open mouth / yawn
    
    drowsy_time_thresh: float = 1.2      # Continuous eye closed duration (sec) for DROWSY state
    sleep_time_thresh: float = 3.0       # Continuous eye closed duration (sec) for SLEEPING state
    yawn_time_thresh: float = 1.8        # Continuous mouth open duration (sec) for YAWNING state
    
    min_blink_duration: float = 0.08     # Minimum duration (sec) for a valid blink
    max_blink_duration: float = 0.85     # Maximum duration (sec) for a blink before counting as prolonged closure
    
    head_pitch_down_thresh: float = 18.0 # Degrees pitch nod down for drowsiness indicator
    
    # Risk score weights (must sum to ~1.0)
    weight_eye_closure: float = 0.50
    weight_yawning: float = 0.20
    weight_blink_anomaly: float = 0.18
    weight_head_posture: float = 0.12
    
    # Audio Alert enable
    enable_audio_alerts: bool = True
    audio_volume: float = 0.8


@dataclass
class DetectionState:
    """System temporal state and metrics payload for a single frame."""
    timestamp: float
    face_detected: bool = False
    
    # Eye metrics
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    eye_closed: bool = False
    eye_closed_duration: float = 0.0
    
    # Blink metrics
    total_blinks: int = 0
    blink_rate_per_min: float = 0.0
    avg_blink_duration: float = 0.0
    
    # Yawn metrics
    mar: float = 0.0
    yawning: bool = False
    yawn_duration: float = 0.0
    total_yawns: int = 0
    
    # Head Pose
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    head_nodding: bool = False
    
    # Overall State & Risk
    state_label: str = "NO FACE"  # AWAKE, DROWSY, SLEEPING, YAWNING, NO FACE
    drowsiness_score: float = 0.0  # 0 to 100%
    risk_level: str = "ALERT"      # ALERT, DROWSY, HIGH DROWSINESS
    
    # Alert Flags
    trigger_drowsy_alert: bool = False
    trigger_sleep_alert: bool = False
