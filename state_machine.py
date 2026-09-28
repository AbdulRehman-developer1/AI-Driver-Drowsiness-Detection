"""
Temporal State Machine, Blink & Yawn Trackers, and Drowsiness Risk Score Engine.
"""

import time
from collections import deque
from typing import List, Tuple
from config import DrowsinessConfig, DetectionState


class DrowsinessStateMachine:
    """
    Manages continuous state transitions, blink tracking, yawning tracking,
    and combined Drowsiness Risk Score evaluation over time.
    """
    
    def __init__(self, config: DrowsinessConfig = None):
        self.config = config or DrowsinessConfig()
        self.reset()
        
    def reset(self):
        """Reset all temporal state variables."""
        self.eye_closed_start: float = 0.0
        self.mouth_open_start: float = 0.0
        self.eye_was_closed: bool = False
        self.mouth_was_open: bool = False
        
        self.total_blinks: int = 0
        self.total_yawns: int = 0
        
        self.blink_timestamps: deque = deque(maxlen=100)
        self.blink_durations: deque = deque(maxlen=50)
        self.yawn_timestamps: deque = deque(maxlen=50)
        
        self.smoothed_score: float = 0.0
        self.last_timestamp: float = time.time()
        
    def update_config(self, config: DrowsinessConfig):
        """Update system threshold parameters dynamically."""
        self.config = config
        
    def process_frame(
        self,
        current_time: float,
        face_detected: bool,
        left_ear: float = 0.0,
        right_ear: float = 0.0,
        mar: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0
    ) -> DetectionState:
        """
        Process a single frame metrics update and advance the state machine.
        """
        state = DetectionState(timestamp=current_time, face_detected=face_detected)
        
        if not face_detected:
            state.state_label = "NO FACE"
            state.drowsiness_score = 0.0
            state.risk_level = "ALERT"
            # Reset ongoing eye/mouth timers if face lost
            self.eye_closed_start = 0.0
            self.mouth_open_start = 0.0
            self.eye_was_closed = False
            self.mouth_was_open = False
            return state
            
        avg_ear = (left_ear + right_ear) / 2.0
        state.left_ear = left_ear
        state.right_ear = right_ear
        state.avg_ear = avg_ear
        state.mar = mar
        state.pitch = pitch
        state.yaw = yaw
        state.roll = roll
        
        # -------------------------------------------------------------
        # 1. EYE CLOSURE & TEMPORAL SLEEP/DROWSY LOGIC
        # -------------------------------------------------------------
        is_eye_closed = avg_ear <= self.config.ear_threshold
        state.eye_closed = is_eye_closed
        
        if is_eye_closed:
            if not self.eye_was_closed:
                self.eye_closed_start = current_time
                self.eye_was_closed = True
            eye_closed_duration = current_time - self.eye_closed_start
        else:
            if self.eye_was_closed:
                closed_duration = current_time - self.eye_closed_start
                # BLINK DETECTION LOGIC:
                # Valid blink must be between min_blink_duration and max_blink_duration
                if self.config.min_blink_duration <= closed_duration <= self.config.max_blink_duration:
                    self.total_blinks += 1
                    self.blink_timestamps.append(current_time)
                    self.blink_durations.append(closed_duration)
                # If closed_duration > max_blink_duration, it's prolonged closure/sleep, NOT a blink!
                self.eye_was_closed = False
                self.eye_closed_start = 0.0
            eye_closed_duration = 0.0
            
        state.eye_closed_duration = eye_closed_duration
        
        # -------------------------------------------------------------
        # 2. BLINK METRICS (Rate per minute, avg blink duration)
        # -------------------------------------------------------------
        state.total_blinks = self.total_blinks
        # Filter blinks within last 60 seconds
        recent_blinks = [t for t in self.blink_timestamps if current_time - t <= 60.0]
        state.blink_rate_per_min = float(len(recent_blinks))
        
        if len(self.blink_durations) > 0:
            state.avg_blink_duration = float(sum(self.blink_durations) / len(self.blink_durations))
        else:
            state.avg_blink_duration = 0.0
            
        # -------------------------------------------------------------
        # 3. YAWNING TEMPORAL LOGIC
        # -------------------------------------------------------------
        is_mouth_open = mar >= self.config.mar_threshold
        
        if is_mouth_open:
            if not self.mouth_was_open:
                self.mouth_open_start = current_time
                self.mouth_was_open = True
            yawn_duration = current_time - self.mouth_open_start
        else:
            if self.mouth_was_open:
                duration = current_time - self.mouth_open_start
                if duration >= self.config.yawn_time_thresh:
                    self.total_yawns += 1
                    self.yawn_timestamps.append(current_time)
                self.mouth_was_open = False
                self.mouth_open_start = 0.0
            yawn_duration = 0.0
            
        state.yawn_duration = yawn_duration
        state.yawning = (yawn_duration >= self.config.yawn_time_thresh) or (is_mouth_open and mar >= self.config.mar_threshold * 1.2)
        state.total_yawns = self.total_yawns
        
        # -------------------------------------------------------------
        # 4. HEAD POSTURE NODDING LOGIC
        # -------------------------------------------------------------
        # Positive pitch angles indicate looking down / nodding
        state.head_nodding = pitch > self.config.head_pitch_down_thresh
        
        # -------------------------------------------------------------
        # 5. STATE MACHINE CLASSIFICATION
        # -------------------------------------------------------------
        if eye_closed_duration >= self.config.sleep_time_thresh:
            state.state_label = "SLEEPING"
            state.trigger_sleep_alert = True
        elif eye_closed_duration >= self.config.drowsy_time_thresh:
            state.state_label = "DROWSY"
            state.trigger_drowsy_alert = True
        elif state.yawning:
            state.state_label = "YAWNING"
        else:
            state.state_label = "AWAKE"
            
        # -------------------------------------------------------------
        # 6. COMBINED DROWSINESS RISK SCORE (0 to 100%)
        # -------------------------------------------------------------
        # A) Eye Closure Score (50%)
        if is_eye_closed:
            eye_ratio = min(1.0, eye_closed_duration / self.config.sleep_time_thresh)
            eye_score = 60.0 + 40.0 * eye_ratio  # 60-100 when closed
        else:
            # Partial eye drooping ratio
            if avg_ear < self.config.ear_threshold * 1.3:
                droop = (self.config.ear_threshold * 1.3 - avg_ear) / (self.config.ear_threshold * 0.3)
                eye_score = min(50.0, float(droop) * 50.0)
            else:
                eye_score = 0.0
                
        # B) Yawning Score (20%)
        recent_yawns = [t for t in self.yawn_timestamps if current_time - t <= 120.0]
        yawn_score = 0.0
        if state.yawning:
            yawn_score = 90.0
        elif len(recent_yawns) > 0:
            yawn_score = min(80.0, len(recent_yawns) * 30.0)
            
        # C) Blink Anomaly Score (18%)
        blink_score = 0.0
        if state.avg_blink_duration > 0.35: # Long blinks indicate fatigue
            blink_score += min(50.0, (state.avg_blink_duration - 0.35) * 200.0)
        if state.blink_rate_per_min > 25.0: # Rapid fluttering blinks
            blink_score += min(50.0, (state.blink_rate_per_min - 25.0) * 3.0)
        blink_score = min(100.0, blink_score)
        
        # D) Head Posture Score (12%)
        head_score = 0.0
        if state.head_nodding:
            head_score = min(100.0, (pitch - self.config.head_pitch_down_thresh) * 5.0 + 50.0)
            
        raw_score = (
            eye_score * self.config.weight_eye_closure +
            yawn_score * self.config.weight_yawning +
            blink_score * self.config.weight_blink_anomaly +
            head_score * self.config.weight_head_posture
        )
        
        # Override score to 95-100% if in SLEEPING state, 65-80% if DROWSY
        # Do this BEFORE EMA so the smoothed score reflects state immediately
        if state.state_label == "SLEEPING":
            raw_score = max(raw_score, 95.0)
            # Also force the smoothed score up immediately for display
            self.smoothed_score = max(self.smoothed_score, 85.0)
        elif state.state_label == "DROWSY":
            raw_score = max(raw_score, 65.0)
            self.smoothed_score = max(self.smoothed_score, 55.0)
            
        # Exponential moving average smoothing (faster alpha = 0.5 for responsiveness)
        alpha = 0.5
        self.smoothed_score = alpha * raw_score + (1.0 - alpha) * self.smoothed_score
        state.drowsiness_score = float(max(0.0, min(100.0, self.smoothed_score)))
        
        # Risk level classification
        if state.drowsiness_score <= 30.0:
            state.risk_level = "ALERT"
        elif state.drowsiness_score <= 60.0:
            state.risk_level = "DROWSY"
        else:
            state.risk_level = "HIGH DROWSINESS"
            
        return state
