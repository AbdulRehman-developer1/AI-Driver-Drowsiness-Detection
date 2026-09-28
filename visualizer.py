"""
OpenCV Computer Vision Visualizer and Glassmorphic HUD Renderer.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from config import (
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    MOUTH_OUTER_CONTOUR,
    MOUTH_INNER_CONTOUR,
    DetectionState
)

# Color Palette (BGR format)
COLOR_BG_DARK = (20, 24, 33)
COLOR_TEXT_WHITE = (245, 245, 250)
COLOR_GREEN = (80, 220, 100)
COLOR_AMBER = (30, 180, 255)      # Orange / Amber
COLOR_RED = (50, 50, 240)         # Bright Red
COLOR_CYAN = (235, 206, 135)
COLOR_PURPLE = (220, 80, 200)
COLOR_GRAY = (120, 125, 135)


class HUDVisualizer:
    """
    Renders high-end computer vision overlays, landmark meshes, dynamic HUD cards,
    risk gauges, and emergency alarm banners onto video frames.
    """
    
    def __init__(self):
        pass
        
    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        landmarks_px: Optional[np.ndarray],
        state: DetectionState,
        show_mesh: bool = True,
        show_hud: bool = True
    ) -> np.ndarray:
        """
        Draw all visual overlays on frame.
        """
        if frame_bgr is None:
            return frame_bgr
            
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        
        # 1. Draw Facial Mesh & Specific Landmarks
        if landmarks_px is not None and len(landmarks_px) > 0:
            if show_mesh:
                self._draw_translucent_mesh(out, landmarks_px)
                
            self._draw_eye_contours(out, landmarks_px, state)
            self._draw_mouth_contour(out, landmarks_px, state)
            self._draw_head_pose_axes(out, landmarks_px, state)
            
        # 2. Draw HUD Data Card & Risk Meter
        if show_hud:
            self._draw_hud_panel(out, state, w, h)
            
        # 3. Draw Emergency Banner Overlays
        if state.state_label == "SLEEPING" or state.trigger_sleep_alert:
            self._draw_alert_banner(out, "🚨 DRIVER SLEEP ALERT", COLOR_RED, w, h)
        elif state.state_label == "DROWSY" or state.trigger_drowsy_alert:
            self._draw_alert_banner(out, "⚠️ DROWSINESS DETECTED", COLOR_AMBER, w, h)
            
        return out

    def _draw_translucent_mesh(self, img: np.ndarray, landmarks_px: np.ndarray):
        """Draw subtle translucent facial mesh dots."""
        overlay = img.copy()
        step = 4
        pts = landmarks_px[::step, :2].astype(np.int32)
        for pt in pts:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, (180, 200, 220), -1)
        # Blend: overlay (with dots) 40% + original img 60%
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    def _draw_eye_contours(self, img: np.ndarray, landmarks_px: np.ndarray, state: DetectionState):
        """Draw eye contours with color coding based on closure state."""
        if state.eye_closed:
            color = COLOR_RED
            thickness = 2
        elif state.avg_ear < 0.25:
            color = COLOR_AMBER
            thickness = 2
        else:
            color = COLOR_GREEN
            thickness = 1
            
        left_pts = landmarks_px[LEFT_EYE_CONTOUR, :2].astype(np.int32)
        right_pts = landmarks_px[RIGHT_EYE_CONTOUR, :2].astype(np.int32)
        
        cv2.polylines(img, [left_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.polylines(img, [right_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
        # Highlight eye corner dots
        for pt in left_pts:
            cv2.circle(img, (pt[0], pt[1]), 2, color, -1, cv2.LINE_AA)
        for pt in right_pts:
            cv2.circle(img, (pt[0], pt[1]), 2, color, -1, cv2.LINE_AA)

    def _draw_mouth_contour(self, img: np.ndarray, landmarks_px: np.ndarray, state: DetectionState):
        """Draw mouth contour with color coding for yawning."""
        color = COLOR_AMBER if state.yawning else COLOR_CYAN
        thickness = 2 if state.yawning else 1
        
        outer_pts = landmarks_px[MOUTH_OUTER_CONTOUR, :2].astype(np.int32)
        inner_pts = landmarks_px[MOUTH_INNER_CONTOUR, :2].astype(np.int32)
        
        cv2.polylines(img, [outer_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.polylines(img, [inner_pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    def _draw_head_pose_axes(self, img: np.ndarray, landmarks_px: np.ndarray, state: DetectionState):
        """Draw 3D pose orientation vector pointing from nose tip."""
        try:
            nose_pt = landmarks_px[1, :2].astype(np.int32)
            # Compute end point from pitch and yaw
            length = 50
            dx = int(length * np.sin(np.radians(state.yaw)))
            dy = int(-length * np.sin(np.radians(state.pitch)))
            end_pt = (nose_pt[0] + dx, nose_pt[1] + dy)
            
            cv2.arrowedLine(img, (nose_pt[0], nose_pt[1]), end_pt, (255, 255, 0), 2, tipLength=0.25, lineType=cv2.LINE_AA)
        except Exception:
            pass

    def _draw_hud_panel(self, img: np.ndarray, state: DetectionState, w: int, h: int):
        """Draw glassmorphic HUD status card at top-left corner."""
        panel_w = min(360, int(w * 0.45))
        # Auto-size panel height: header(42) + 6 lines(22px each) + bar(26) + padding(30)
        panel_h = min(h - 20, 260)
        margin = 15
        
        # Semi-transparent background box
        overlay = img.copy()
        cv2.rectangle(overlay, (margin, margin), (margin + panel_w, margin + panel_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        cv2.rectangle(img, (margin, margin), (margin + panel_w, margin + panel_h), (60, 70, 90), 1)
        
        # State Badge Color & Label
        if state.state_label == "SLEEPING":
            badge_color = COLOR_RED
        elif state.state_label == "DROWSY":
            badge_color = COLOR_AMBER
        elif state.state_label == "YAWNING":
            badge_color = COLOR_PURPLE
        elif state.state_label == "AWAKE":
            badge_color = COLOR_GREEN
        else:
            badge_color = COLOR_GRAY
            
        # Header Badge
        cv2.rectangle(img, (margin + 12, margin + 12), (margin + 160, margin + 42), badge_color, -1)
        cv2.putText(img, f" {state.state_label} ", (margin + 20, margin + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Metrics List
        y_offset = margin + 65
        line_height = 22
        
        metrics_text = [
            f"L.EAR:{state.left_ear:.2f}  R.EAR:{state.right_ear:.2f}",
            f"Avg EAR:{state.avg_ear:.2f}  Eye:{'CLOSED' if state.eye_closed else 'OPEN'}",
            f"MAR:{state.mar:.2f}  Yawns:{state.total_yawns}",
            f"Blinks:{state.total_blinks}  Rate:{state.blink_rate_per_min:.0f}/min",
            f"Eye Closed:{state.eye_closed_duration:.1f}s",
            f"Pitch:{state.pitch:.1f}deg  Nod:{'YES' if state.head_nodding else 'NO'}"
        ]
        
        for text in metrics_text:
            cv2.putText(img, text, (margin + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            y_offset += line_height
            
        # Drowsiness Risk Meter Bar at bottom of panel
        bar_x1 = margin + 15
        bar_y1 = y_offset + 5
        bar_x2 = margin + panel_w - 15
        bar_y2 = bar_y1 + 16
        
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 45, 55), -1)
        
        score_pct = state.drowsiness_score / 100.0
        fill_w = int((bar_x2 - bar_x1) * score_pct)
        
        if state.drowsiness_score <= 30:
            meter_color = COLOR_GREEN
        elif state.drowsiness_score <= 60:
            meter_color = COLOR_AMBER
        else:
            meter_color = COLOR_RED
            
        if fill_w > 0:
            cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), meter_color, -1)
            
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 110, 130), 1)
        cv2.putText(img, f"Risk Score: {state.drowsiness_score:.0f}%", (bar_x1 + 8, bar_y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_alert_banner(self, img: np.ndarray, title: str, bg_color: Tuple[int, int, int], w: int, h: int):
        """Draw prominent full-width banner at the bottom of the frame."""
        banner_h = 60
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
        # Use ASCII-safe text (cv2.putText does not render emoji)
        safe_title = title.encode('ascii', errors='replace').decode('ascii').replace('?', '')
        text_size = cv2.getTextSize(safe_title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = max(10, (w - text_size[0]) // 2)
        cv2.putText(img, safe_title, (text_x, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
