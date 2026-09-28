"""
Utility functions: Synthetic driver video generator, Plotly analytics chart builders,
video frame pipeline helpers, and CSV logger.
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict, Any, Optional

from config import DetectionState
from visualizer import COLOR_GREEN, COLOR_AMBER, COLOR_RED


def generate_synthetic_driver_video(output_path: str = "sample_driver.mp4", duration_sec: int = 15, fps: int = 30) -> str:
    """
    Generates a synthetic driver video file simulating:
      - Phase 1: Awake driving with normal blinks (0 - 4s)
      - Phase 2: Yawning (4 - 7s)
      - Phase 3: Drowsy eye drooping (7 - 10s)
      - Phase 4: Sleeping / Continuous eye closure with head nod (10 - 15s)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 640, 480
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    total_frames = duration_sec * fps
    center = (w // 2, h // 2 - 20)
    
    for i in range(total_frames):
        t = i / fps
        frame = np.full((h, w, 3), (35, 40, 50), dtype=np.uint8)
        
        # Determine face animation parameters based on phase
        pitch_shift = 0
        eye_open_factor = 1.0  # 1.0 = fully open, 0.0 = closed
        mouth_open_factor = 0.0 # 0.0 = closed, 1.0 = wide open
        
        if t < 4.0:
            # Phase 1: Normal open eyes, periodic blink at t=2.0s
            if 1.9 <= t <= 2.1:
                eye_open_factor = 0.05
            else:
                eye_open_factor = 0.95 + 0.05 * np.sin(2 * np.pi * t)
        elif t < 7.0:
            # Phase 2: Yawning
            eye_open_factor = 0.6
            yawn_t = (t - 4.0) / 3.0
            mouth_open_factor = float(np.sin(np.pi * yawn_t))
        elif t < 10.0:
            # Phase 3: Drowsy eye drooping
            droop_t = (t - 7.0) / 3.0
            eye_open_factor = float(max(0.1, 0.8 - 0.6 * droop_t + 0.1 * np.sin(4 * np.pi * t)))
            pitch_shift = int(15 * droop_t)
        else:
            # Phase 4: Sleeping / Eyes closed continuous & head nod
            eye_open_factor = 0.0
            pitch_shift = 30 + int(5 * np.sin(np.pi * (t - 10.0)))
            
        # Draw realistic stylized driver face
        face_center = (center[0], center[1] + pitch_shift)
        
        # Face contour oval
        cv2.ellipse(frame, face_center, (110, 140), 0, 0, 360, (190, 210, 230), -1)
        cv2.ellipse(frame, face_center, (110, 140), 0, 0, 360, (120, 140, 160), 3)
        
        # Eyes
        left_eye_c = (face_center[0] - 45, face_center[1] - 30)
        right_eye_c = (face_center[0] + 45, face_center[1] - 30)
        
        eye_h = max(2, int(22 * eye_open_factor))
        cv2.ellipse(frame, left_eye_c, (24, eye_h), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, right_eye_c, (24, eye_h), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, left_eye_c, (24, eye_h), 0, 0, 360, (50, 50, 50), 2)
        cv2.ellipse(frame, right_eye_c, (24, eye_h), 0, 0, 360, (50, 50, 50), 2)
        
        # Pupils (if eyes open)
        if eye_h > 4:
            cv2.circle(frame, left_eye_c, 8, (60, 40, 20), -1)
            cv2.circle(frame, right_eye_c, 8, (60, 40, 20), -1)
            cv2.circle(frame, (left_eye_c[0] - 2, left_eye_c[1] - 2), 2, (255, 255, 255), -1)
            cv2.circle(frame, (right_eye_c[0] - 2, right_eye_c[1] - 2), 2, (255, 255, 255), -1)
        else:
            # Closed eye slit line
            cv2.line(frame, (left_eye_c[0] - 22, left_eye_c[1]), (left_eye_c[0] + 22, left_eye_c[1]), (40, 40, 40), 2)
            cv2.line(frame, (right_eye_c[0] - 22, right_eye_c[1]), (right_eye_c[0] + 22, right_eye_c[1]), (40, 40, 40), 2)
            
        # Eyebrows
        cv2.line(frame, (left_eye_c[0] - 25, left_eye_c[1] - 28), (left_eye_c[0] + 20, left_eye_c[1] - 26), (40, 30, 20), 3)
        cv2.line(frame, (right_eye_c[0] - 20, right_eye_c[1] - 26), (right_eye_c[0] + 25, right_eye_c[1] - 28), (40, 30, 20), 3)
        
        # Nose
        nose_top = (face_center[0], face_center[1] - 10)
        nose_bottom = (face_center[0], face_center[1] + 20)
        cv2.line(frame, nose_top, nose_bottom, (140, 150, 170), 3)
        cv2.circle(frame, (face_center[0] - 8, face_center[1] + 22), 4, (140, 150, 170), -1)
        cv2.circle(frame, (face_center[0] + 8, face_center[1] + 22), 4, (140, 150, 170), -1)
        
        # Mouth
        mouth_c = (face_center[0], face_center[1] + 55)
        mouth_w = 40
        mouth_h = max(4, int(35 * mouth_open_factor))
        
        if mouth_h > 6:
            cv2.ellipse(frame, mouth_c, (mouth_w, mouth_h), 0, 0, 360, (50, 20, 30), -1)
            cv2.ellipse(frame, mouth_c, (mouth_w, mouth_h), 0, 0, 360, (120, 50, 60), 2)
        else:
            cv2.line(frame, (mouth_c[0] - mouth_w, mouth_c[1]), (mouth_c[0] + mouth_w, mouth_c[1]), (120, 50, 60), 3)
            
        # Steering wheel silhouette
        cv2.circle(frame, (w // 2, h + 120), 220, (30, 35, 45), 25)
        
        out.write(frame)
        
    out.release()
    return output_path


def create_plotly_analytics(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Generates interactive Plotly figures for EAR timeline, Drowsiness Risk timeline,
    MAR timeline, and state distribution breakdown.
    """
    figures = {}
    
    if df.empty or "timestamp" not in df.columns:
        return figures
        
    # Standard template styling
    template = "plotly_dark"
    
    # 1. EAR Timeline Chart
    fig_ear = go.Figure()
    fig_ear.add_trace(go.Scatter(
        x=df["timestamp"], y=df["avg_ear"],
        mode="lines", name="Average EAR",
        line=dict(color="#00E676", width=2)
    ))
    if "ear_threshold" in df.columns:
        thresh = df["ear_threshold"].iloc[0]
    else:
        thresh = 0.23
        
    fig_ear.add_hline(
        y=thresh, line_dash="dash", line_color="#FF5252",
        annotation_text=f"Threshold ({thresh:.2f})", annotation_position="top right"
    )
    fig_ear.update_layout(
        title="<b>Eye Aspect Ratio (EAR) Over Time</b>",
        xaxis_title="Time (s)", yaxis_title="EAR",
        template=template, height=320, margin=dict(l=40, r=40, t=50, b=40)
    )
    figures["ear_chart"] = fig_ear

    # 2. Drowsiness Risk Score Chart
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=df["timestamp"], y=df["drowsiness_score"],
        mode="lines", name="Drowsiness Risk %",
        fill="tozeroy",
        line=dict(color="#FF9100", width=2.5)
    ))
    # Risk Threshold Bands
    fig_risk.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0, annotation_text="ALERT Zone")
    fig_risk.add_hrect(y0=30, y1=60, fillcolor="orange", opacity=0.12, line_width=0, annotation_text="DROWSY Zone")
    fig_risk.add_hrect(y0=60, y1=100, fillcolor="red", opacity=0.15, line_width=0, annotation_text="HIGH DROWSINESS Zone")
    
    fig_risk.update_layout(
        title="<b>Drowsiness Risk Score (%) Over Time</b>",
        xaxis_title="Time (s)", yaxis_title="Risk Score (%)",
        yaxis=dict(range=[0, 105]),
        template=template, height=340, margin=dict(l=40, r=40, t=50, b=40)
    )
    figures["risk_chart"] = fig_risk

    # 3. MAR (Mouth Aspect Ratio) Chart
    fig_mar = go.Figure()
    fig_mar.add_trace(go.Scatter(
        x=df["timestamp"], y=df["mar"],
        mode="lines", name="MAR",
        line=dict(color="#00E5FF", width=2)
    ))
    mar_thresh = df["mar_threshold"].iloc[0] if "mar_threshold" in df.columns else 0.55
    fig_mar.add_hline(
        y=mar_thresh, line_dash="dash", line_color="#FF4081",
        annotation_text=f"Yawn Threshold ({mar_thresh:.2f})", annotation_position="top right"
    )
    fig_mar.update_layout(
        title="<b>Mouth Aspect Ratio (MAR) Over Time</b>",
        xaxis_title="Time (s)", yaxis_title="MAR",
        template=template, height=300, margin=dict(l=40, r=40, t=50, b=40)
    )
    figures["mar_chart"] = fig_mar

    # 4. Driver State Breakdown Donut Chart
    if "state_label" in df.columns:
        counts = df["state_label"].value_counts().reset_index()
        counts.columns = ["State", "Count"]
        
        color_map = {
            "AWAKE": "#00E676",
            "DROWSY": "#FF9100",
            "SLEEPING": "#FF1744",
            "YAWNING": "#D500F9",
            "NO FACE": "#78909C"
        }
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=counts["State"], values=counts["Count"],
            hole=0.45, marker=dict(colors=[color_map.get(s, "#90A4AE") for s in counts["State"]])
        )])
        fig_donut.update_layout(
            title="<b>Driver State Distribution</b>",
            template=template, height=320, margin=dict(l=20, r=20, t=50, b=20)
        )
        figures["donut_chart"] = fig_donut

    return figures
