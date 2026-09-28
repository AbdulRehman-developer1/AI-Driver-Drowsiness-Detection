"""
AI Driver Sleep & Drowsiness Detection System
Main Streamlit Application
"""

import os
import sys
import time
import tempfile
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import DrowsinessConfig, DetectionState, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from detector import FaceLandmarkerDetector
from metrics import calculate_ear, calculate_mar, estimate_head_pose
from state_machine import DrowsinessStateMachine
from visualizer import HUDVisualizer
from audio import AudioAlertManager
from utils import generate_synthetic_driver_video, create_plotly_analytics

# ─────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="DrowseGuard AI — Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# GLOBAL CSS STYLING
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1117 0%, #131b2a 50%, #0d1117 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1620 0%, #111d2c 100%);
    border-right: 1px solid #1f2d3d;
}

.hero-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid #1e3a4f;
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #00e5ff, #0097a7, #00bcd4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.hero-subtitle { color: #78909c; font-size: 1.0rem; margin-top: 6px; }
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1); border: 1px solid rgba(0,229,255,0.3);
    color: #00e5ff; border-radius: 20px; padding: 3px 14px;
    font-size: 0.75rem; font-weight: 600; margin-right: 8px; margin-top: 10px;
}

.state-banner {
    border-radius: 12px; padding: 16px 24px; text-align: center;
    font-size: 1.4rem; font-weight: 800; letter-spacing: 2px;
    margin-bottom: 16px; animation: statePulse 2s ease-in-out infinite;
}
.state-awake { background: rgba(0,200,83,0.12); border: 2px solid rgba(0,200,83,0.5); color: #00e676; }
.state-drowsy { background: rgba(255,145,0,0.15); border: 2px solid rgba(255,145,0,0.6); color: #ff9100; }
.state-sleeping { background: rgba(255,23,68,0.18); border: 2px solid rgba(255,23,68,0.7); color: #ff1744; animation: sleepAlarm 0.8s ease-in-out infinite; }
.state-yawning { background: rgba(213,0,249,0.12); border: 2px solid rgba(213,0,249,0.5); color: #d500f9; }
.state-noface { background: rgba(96,125,139,0.1); border: 2px solid rgba(96,125,139,0.3); color: #78909c; }

@keyframes statePulse { 0%,100%{opacity:1} 50%{opacity:0.85} }
@keyframes sleepAlarm { 0%,100%{opacity:1;box-shadow:0 0 20px rgba(255,23,68,0.4)} 50%{opacity:0.7;box-shadow:0 0 40px rgba(255,23,68,0.8)} }

.risk-gauge-container {
    background: #131b2a; border: 1px solid #1f2d3d;
    border-radius: 14px; padding: 18px 22px; margin-bottom: 12px;
}
.risk-label { color: #78909c; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px; }
.risk-bar-track { background: #1a2538; border-radius: 8px; height: 18px; overflow: hidden; border: 1px solid #1f2d3d; }
.risk-bar-fill { height: 100%; border-radius: 8px; transition: width 0.4s ease; }
.risk-pct { color: #e0e0e0; font-size: 0.85rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-top: 6px; text-align: right; }

.alert-warning {
    background: rgba(255,145,0,0.12); border: 2px solid rgba(255,145,0,0.5);
    border-radius: 12px; padding: 18px 24px; color: #ffcc02;
    font-weight: 700; font-size: 1.1rem; text-align: center;
    margin: 8px 0; animation: statePulse 1.5s ease-in-out infinite;
}
.alert-critical {
    background: rgba(255,23,68,0.18); border: 3px solid rgba(255,23,68,0.7);
    border-radius: 12px; padding: 22px 24px; color: #ff5252;
    font-weight: 800; font-size: 1.25rem; text-align: center;
    margin: 8px 0; animation: sleepAlarm 0.9s ease-in-out infinite;
}

.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
.stat-chip {
    background: linear-gradient(135deg, #131b2a, #1a2538);
    border: 1px solid #1f2d3d;
    border-radius: 12px; padding: 14px 10px; text-align: center;
    transition: all 0.25s ease;
    cursor: default;
}
.stat-chip:hover {
    border-color: #00bcd4;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,188,212,0.15);
    background: linear-gradient(135deg, #162030, #1e2e42);
}
.stat-chip-label {
    color: #546e7a; font-size: 0.63rem;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700;
    margin-bottom: 4px;
}
.stat-chip-val {
    color: #e0f7fa; font-size: 1.15rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 12px rgba(0,229,255,0.2);
}

.sidebar-section-title {
    color: #00bcd4; font-size: 0.73rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.8px;
    margin: 22px 0 10px 0; padding: 6px 0 6px 10px;
    border-bottom: 1px solid #1f2d3d;
    border-left: 3px solid #00bcd4;
    border-radius: 0 0 0 2px;
    background: linear-gradient(90deg, rgba(0,188,212,0.06), transparent);
}

[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif; font-weight: 600;
    color: #546e7a !important;
    transition: color 0.2s ease !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] button:hover { color: #80deea !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
    text-shadow: 0 0 12px rgba(0,229,255,0.4);
}

/* ── Cockpit Control Panel Bar ── */
.cockpit-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #131b2a, #1a2538);
    border: 1px solid #1f2d3d;
    border-radius: 12px;
    padding: 10px 16px;
    margin: 12px 0 10px 0;
}
.cockpit-title {
    color: #90a4ae;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.cockpit-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 20px;
}
.status-idle {
    background: rgba(144, 164, 174, 0.12);
    color: #90a4ae;
    border: 1px solid rgba(144, 164, 174, 0.3);
}
.status-live {
    background: rgba(0, 230, 118, 0.15);
    color: #00e676;
    border: 1px solid rgba(0, 230, 118, 0.5);
    box-shadow: 0 0 12px rgba(0, 230, 118, 0.25);
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.status-idle .pulse-dot {
    background: #90a4ae;
}
.status-live .pulse-dot {
    background: #00e676;
    box-shadow: 0 0 8px #00e676;
    animation: livePulse 1.2s infinite ease-in-out;
}
@keyframes livePulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.5; }
}

/* ── Base Streamlit Button Overrides ── */
.stButton > button {
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    padding: 12px 6px !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
/* Ensure button text never truncates or has ellipses */
.stButton > button p,
.stButton > button span,
.stButton > button div {
    font-size: clamp(0.78rem, 1vw, 0.90rem) !important;
    font-weight: 800 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Shimmer sweep animation */
.stButton > button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    transition: left 0.5s ease;
}
.stButton > button:hover::after { left: 160%; }

/* ── 1. START BUTTON: Vivid Emerald & Cyan Gradient ── */
div.st-key-btn_start button,
div[class*="st-key-btn_start"] button,
div[data-testid="stColumn"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, #00E676 0%, #00B0FF 100%) !important;
    color: #022415 !important;
    border: 1px solid rgba(0, 230, 118, 0.8) !important;
    box-shadow: 0 4px 18px rgba(0, 230, 118, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
    animation: pulseStart 2.4s ease-in-out infinite !important;
}
div.st-key-btn_start button p,
div.st-key-btn_start button span {
    color: #022415 !important;
    text-shadow: 0 1px 0 rgba(255,255,255,0.3) !important;
}
div.st-key-btn_start button:hover,
div[data-testid="stColumn"]:nth-child(1) .stButton > button:hover {
    background: linear-gradient(135deg, #33eb8e 0%, #40c4ff 100%) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(0, 230, 118, 0.65), 0 0 18px rgba(0, 176, 255, 0.5) !important;
    animation: none !important;
}
div.st-key-btn_start button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(0, 230, 118, 0.5) !important;
}

/* ── 2. STOP BUTTON: Bold Crimson & Pink Alert Gradient ── */
div.st-key-btn_stop button,
div[class*="st-key-btn_stop"] button,
div[data-testid="stColumn"]:nth-child(2) .stButton > button {
    background: linear-gradient(135deg, #FF1744 0%, #C2185B 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 23, 68, 0.8) !important;
    box-shadow: 0 4px 18px rgba(255, 23, 68, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
}
div.st-key-btn_stop button p,
div.st-key-btn_stop button span {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
}
div.st-key-btn_stop button:hover,
div[data-testid="stColumn"]:nth-child(2) .stButton > button:hover {
    background: linear-gradient(135deg, #ff4569 0%, #d81b60 100%) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(255, 23, 68, 0.7) !important;
}
div.st-key-btn_stop button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(255, 23, 68, 0.5) !important;
}

/* ── 3. RESET BUTTON: Electric Amber Gradient ── */
div.st-key-btn_reset button,
div[class*="st-key-btn_reset"] button,
div[data-testid="stColumn"]:nth-child(3) .stButton > button {
    background: linear-gradient(135deg, #FF9100 0%, #E65100 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 145, 0, 0.8) !important;
    box-shadow: 0 4px 18px rgba(255, 145, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
}
div.st-key-btn_reset button p,
div.st-key-btn_reset button span {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
}
div.st-key-btn_reset button:hover,
div[data-testid="stColumn"]:nth-child(3) .stButton > button:hover {
    background: linear-gradient(135deg, #ffa726 0%, #f57c00 100%) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(255, 145, 0, 0.65) !important;
}
div.st-key-btn_reset button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(255, 145, 0, 0.5) !important;
}

@keyframes pulseStart {
    0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.7), 0 4px 18px rgba(0,230,118,0.4); }
    50%  { box-shadow: 0 0 0 12px rgba(0,230,118,0.0), 0 4px 22px rgba(0,230,118,0.6); }
    100% { box-shadow: 0 0 0 0 rgba(0,230,118,0.0), 0 4px 18px rgba(0,230,118,0.4); }
}

div[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "detector": None,
        "state_machine": None,
        "recording": False,
        "session_log": [],
        "session_start_time": None,
        "total_frames": 0,
        "last_alert_time": 0.0,
        "alert_cooldown": 4.0,
        "config": DrowsinessConfig(),
        "show_mesh": True,
        "show_hud": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ─────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI model…")
def load_detector():
    return FaceLandmarkerDetector()

@st.cache_resource(show_spinner=False)
def load_audio():
    return AudioAlertManager()


# ─────────────────────────────────────────────────
# CONFIG FROM SIDEBAR
# ─────────────────────────────────────────────────
def build_config() -> DrowsinessConfig:
    return DrowsinessConfig(
        ear_threshold=st.session_state.get("sl_ear", 0.23),
        mar_threshold=st.session_state.get("sl_mar", 0.55),
        drowsy_time_thresh=st.session_state.get("sl_drowsy_time", 1.2),
        sleep_time_thresh=st.session_state.get("sl_sleep_time", 3.0),
        yawn_time_thresh=st.session_state.get("sl_yawn_time", 1.8),
        enable_audio_alerts=st.session_state.get("cb_audio", True),
    )


# ─────────────────────────────────────────────────
# DETECTION PIPELINE
# ─────────────────────────────────────────────────
def process_frame(frame_bgr, detector, state_machine, hud, config):
    current_time = time.time()
    face_detected, landmarks_px, _ = detector.process_frame(frame_bgr)

    if face_detected and landmarks_px is not None:
        left_ear = calculate_ear(landmarks_px[LEFT_EYE_INDICES, :3])
        right_ear = calculate_ear(landmarks_px[RIGHT_EYE_INDICES, :3])
        mar = calculate_mar(landmarks_px)
        pitch, yaw, roll = estimate_head_pose(landmarks_px, frame_bgr.shape)
    else:
        left_ear = right_ear = mar = pitch = yaw = roll = 0.0

    state = state_machine.process_frame(
        current_time=current_time,
        face_detected=face_detected,
        left_ear=left_ear,
        right_ear=right_ear,
        mar=mar,
        pitch=pitch, yaw=yaw, roll=roll
    )

    annotated = hud.draw_overlay(
        frame_bgr,
        landmarks_px if face_detected else None,
        state,
        show_mesh=st.session_state.get("show_mesh", True),
        show_hud=st.session_state.get("show_hud", True)
    )
    return annotated, state


# ─────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────
def render_metric_cards(state: DetectionState):
    state_css = {"AWAKE":"state-awake","DROWSY":"state-drowsy","SLEEPING":"state-sleeping","YAWNING":"state-yawning","NO FACE":"state-noface"}
    icon_map = {"AWAKE":"✅","DROWSY":"⚠️","SLEEPING":"🚨","YAWNING":"🥱","NO FACE":"👤"}
    css = state_css.get(state.state_label, "state-noface")
    icon = icon_map.get(state.state_label, "")

    st.markdown(f'<div class="state-banner {css}">{icon} {state.state_label}</div>', unsafe_allow_html=True)

    if state.trigger_sleep_alert or state.state_label == "SLEEPING":
        st.markdown('<div class="alert-critical">🚨 DRIVER SLEEP ALERT — WAKE UP IMMEDIATELY!</div>', unsafe_allow_html=True)
    elif state.trigger_drowsy_alert or state.state_label == "DROWSY":
        st.markdown('<div class="alert-warning">⚠️ DROWSINESS DETECTED — Please take a break</div>', unsafe_allow_html=True)

    score = state.drowsiness_score
    if score <= 30:
        bar_color = "linear-gradient(90deg,#00e676,#00c853)"
        risk_text = f"🟢 ALERT &nbsp; {score:.0f}%"
    elif score <= 60:
        bar_color = "linear-gradient(90deg,#ff9100,#ff6d00)"
        risk_text = f"🟡 DROWSY &nbsp; {score:.0f}%"
    else:
        bar_color = "linear-gradient(90deg,#ff1744,#b71c1c)"
        risk_text = f"🔴 HIGH DROWSINESS &nbsp; {score:.0f}%"

    st.markdown(f"""
    <div class="risk-gauge-container">
        <div class="risk-label">⚡ Drowsiness Risk Score</div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{score:.1f}%;background:{bar_color};"></div>
        </div>
        <div class="risk-pct">{risk_text}</div>
    </div>
    <div class="stats-row">
        <div class="stat-chip"><div class="stat-chip-label">Left EAR</div><div class="stat-chip-val">{state.left_ear:.2f}</div></div>
        <div class="stat-chip"><div class="stat-chip-label">Right EAR</div><div class="stat-chip-val">{state.right_ear:.2f}</div></div>
        <div class="stat-chip"><div class="stat-chip-label">Avg EAR</div><div class="stat-chip-val">{state.avg_ear:.2f}</div></div>
    </div>
    <div class="stats-row">
        <div class="stat-chip"><div class="stat-chip-label">MAR</div><div class="stat-chip-val">{state.mar:.2f}</div></div>
        <div class="stat-chip"><div class="stat-chip-label">Blinks</div><div class="stat-chip-val">{state.total_blinks}</div></div>
        <div class="stat-chip"><div class="stat-chip-label">Rate/min</div><div class="stat-chip-val">{state.blink_rate_per_min:.0f}</div></div>
    </div>
    <div class="stats-row">
        <div class="stat-chip"><div class="stat-chip-label">Eye Closed</div><div class="stat-chip-val">{state.eye_closed_duration:.1f}s</div></div>
        <div class="stat-chip"><div class="stat-chip-label">Yawns</div><div class="stat-chip-val">{state.total_yawns}</div></div>
        <div class="stat-chip"><div class="stat-chip-label">Head Pitch</div><div class="stat-chip-val">{state.pitch:.0f}°</div></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 0 8px 0;">
            <div style="font-size:1.5rem;font-weight:800;color:#00e5ff;">DrowseGuard AI</div>
            <div style="font-size:0.75rem;color:#546e7a;margin-top:2px;">Driver Safety System v1.0</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sidebar-section-title">📹 Input Source</div>', unsafe_allow_html=True)
        source = st.selectbox("Video Source",
            ["Webcam (Live)", "Upload Video File", "Sample Demo Video"],
            key="video_source_select", label_visibility="collapsed")

        uploaded_file = None
        if source == "Upload Video File":
            uploaded_file = st.file_uploader("Upload MP4/AVI",
                type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

        st.markdown('<div class="sidebar-section-title">🎯 Detection Thresholds</div>', unsafe_allow_html=True)
        st.slider("EAR Threshold (Eye Closure)", 0.10, 0.40, 0.23, 0.01, key="sl_ear")
        st.slider("MAR Threshold (Yawn)", 0.30, 0.80, 0.55, 0.01, key="sl_mar")

        st.markdown('<div class="sidebar-section-title">⏱️ Alert Timing</div>', unsafe_allow_html=True)
        st.slider("Drowsy Alert (sec)", 0.5, 3.0, 1.2, 0.1, key="sl_drowsy_time")
        st.slider("Sleep Alert (sec)", 1.5, 6.0, 3.0, 0.1, key="sl_sleep_time")
        st.slider("Yawn Confirm (sec)", 0.5, 4.0, 1.8, 0.1, key="sl_yawn_time")

        st.markdown('<div class="sidebar-section-title">🖼️ Display Options</div>', unsafe_allow_html=True)
        st.toggle("Show Facial Mesh", value=True, key="show_mesh")
        st.toggle("Show HUD Overlay", value=True, key="show_hud")
        st.toggle("🔊 Audio Alerts", value=True, key="cb_audio")

        st.divider()
        st.markdown("""
        <div style="color:#37474f;font-size:0.68rem;text-align:center;padding:8px 0;">
            🧠 Powered by MediaPipe Face Landmarker<br>
            👁️ EAR + MAR + Head Pose Analysis<br>
            <span style="color:#1565c0;">⚠️ Not a medical device.</span>
        </div>
        """, unsafe_allow_html=True)

    return source, uploaded_file


# ─────────────────────────────────────────────────
# ANALYTICS TAB
# ─────────────────────────────────────────────────
def render_analytics_tab():
    log = st.session_state.session_log
    if len(log) < 5:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#37474f;">
            <div style="font-size:3rem;">📊</div>
            <div style="font-size:1.1rem;font-weight:600;color:#546e7a;margin-top:12px;">No session data yet</div>
            <div style="font-size:0.85rem;margin-top:6px;">Start a detection session to see analytics</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(log)
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]
    cfg = st.session_state.config
    df["ear_threshold"] = cfg.ear_threshold
    df["mar_threshold"] = cfg.mar_threshold

    figures = create_plotly_analytics(df)

    awake_pct = (df["state_label"] == "AWAKE").sum() / len(df) * 100
    drowsy_pct = (df["state_label"] == "DROWSY").sum() / len(df) * 100
    peak_risk = df["drowsiness_score"].max()
    session_dur = df["timestamp"].iloc[-1] if len(df) > 1 else 0
    total_blinks = df["total_blinks"].max() if "total_blinks" in df.columns else 0
    total_yawns = df["total_yawns"].max() if "total_yawns" in df.columns else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Session Duration", f"{session_dur:.0f}s")
    c2.metric("Awake Time", f"{awake_pct:.0f}%")
    c3.metric("Drowsy Time", f"{drowsy_pct:.0f}%")
    c4.metric("Peak Risk", f"{peak_risk:.0f}%")
    c5.metric("Total Blinks", total_blinks)
    c6.metric("Total Yawns", total_yawns)
    st.divider()

    if "ear_chart" in figures:
        st.plotly_chart(figures["ear_chart"], width='stretch')
    ca, cb = st.columns(2)
    with ca:
        if "mar_chart" in figures:
            st.plotly_chart(figures["mar_chart"], width='stretch')
    with cb:
        if "donut_chart" in figures:
            st.plotly_chart(figures["donut_chart"], width='stretch')
    if "risk_chart" in figures:
        st.plotly_chart(figures["risk_chart"], width='stretch')

    st.divider()
    st.download_button("⬇️ Download Session Log (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="drowsiness_session_log.csv", mime="text/csv")


# ─────────────────────────────────────────────────
# ABOUT TAB
# ─────────────────────────────────────────────────
def render_about_tab():
    st.markdown("""
<div style="max-width:860px;margin:0 auto;">

## 🚗 DrowseGuard AI — System Overview

**DrowseGuard AI** monitors a driver's face in real-time using AI to detect fatigue and drowsiness.

---
### 👁️ Eye Aspect Ratio (EAR)
Measures the vertical-to-horizontal eye opening ratio using 6 landmark points per eye:
```
EAR = (||P2−P6|| + ||P3−P5||) / (2 × ||P1−P4||)
```
- **Open eye** → EAR ≈ 0.28–0.35
- **Closed eye** → EAR ≈ 0.05–0.15

---
### 🥱 Mouth Aspect Ratio (MAR)
Detects yawning using 8 mouth landmarks:
```
MAR = (||Vtop−Vbottom|| + ||VL_top−VL_bot|| + ||VR_top−VR_bot||) / (3 × ||Hleft−Hright||)
```

---
### 🧠 Temporal State Machine

| State | Trigger |
|-------|---------|
| ✅ AWAKE | EAR > threshold |
| ⚠️ DROWSY | EAR ≤ threshold for 1–2s |
| 🚨 SLEEPING | EAR ≤ threshold for 3+ sec |
| 🥱 YAWNING | MAR ≥ threshold for 1.8+ sec |

---
### ⚡ Drowsiness Risk Score (0–100%)

| Signal | Weight |
|--------|--------|
| Eye Closure | 50% |
| Yawning | 20% |
| Blink Anomaly | 18% |
| Head Posture | 12% |

> ⚠️ **Drowsiness Risk Score** — not a medically validated measurement.

---
### 🔧 Technologies
- **MediaPipe Face Landmarker** (478-point mesh)
- **OpenCV** (video capture + HUD rendering)
- **NumPy / SciPy** (math)
- **Streamlit** (web UI)
- **Plotly** (analytics)

</div>
""")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🚗 DrowseGuard AI</div>
        <div class="hero-subtitle">Real-time Driver Sleep &amp; Drowsiness Detection System</div>
        <div style="margin-top:12px;">
            <span class="hero-badge">MediaPipe</span>
            <span class="hero-badge">EAR + MAR</span>
            <span class="hero-badge">Head Pose</span>
            <span class="hero-badge">Live Analytics</span>
            <span class="hero-badge">Audio Alerts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    source, uploaded_file = render_sidebar()
    st.session_state.config = build_config()

    tab_live, tab_analytics, tab_about = st.tabs(["🎥 Live Detection", "📊 Analytics", "ℹ️ About"])

    # ── LIVE DETECTION TAB ──────────────────────
    with tab_live:
        col_video, col_metrics = st.columns([3, 2], gap="large")

        with col_video:
            video_placeholder = st.empty()

            # Cockpit Control Panel Status Bar
            is_active = st.session_state.recording
            status_text = "MONITORING ACTIVE" if is_active else "SYSTEM READY"
            status_class = "status-live" if is_active else "status-idle"
            st.markdown(f"""
            <div class="cockpit-bar">
                <div class="cockpit-title">⚡ AI DRIVER MONITOR CONTROLS</div>
                <div class="cockpit-status {status_class}">
                    <span class="pulse-dot"></span> {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            ctrl1, ctrl2, ctrl3 = st.columns([1.35, 0.85, 1.1], gap="small")
            with ctrl1:
                start_btn = st.button("▶  START DETECTION", width='stretch', key="btn_start")
            with ctrl2:
                stop_btn = st.button("⏹  STOP", width='stretch', key="btn_stop")
            with ctrl3:
                reset_btn = st.button("🔄  RESET", width='stretch', key="btn_reset")
            st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        with col_metrics:
            metrics_placeholder = st.empty()
            audio_placeholder = st.empty()

        # Button actions
        if start_btn:
            st.session_state.recording = True
            st.session_state.session_log = []
            st.session_state.session_start_time = time.time()
            st.session_state.total_frames = 0

        if stop_btn:
            st.session_state.recording = False

        if reset_btn:
            st.session_state.recording = False
            st.session_state.session_log = []
            st.session_state.session_start_time = None
            st.session_state.total_frames = 0
            if st.session_state.state_machine:
                st.session_state.state_machine.reset()

        # Idle placeholder
        if not st.session_state.recording:
            video_placeholder.markdown("""
            <div style="height:380px;display:flex;align-items:center;justify-content:center;
                        background:#131b2a;border-radius:12px;border:1px solid #1f2d3d;">
                <div style="text-align:center;color:#37474f;">
                    <div style="font-size:3.5rem;">🚗</div>
                    <div style="font-size:1.1rem;margin-top:10px;color:#546e7a;font-weight:600;">DrowseGuard AI Ready</div>
                    <div style="font-size:0.85rem;margin-top:6px;">Select a source and press ▶ Start Detection</div>
                </div>
            </div>""", unsafe_allow_html=True)
            metrics_placeholder.markdown("""
            <div style="height:200px;display:flex;align-items:center;justify-content:center;
                        background:#131b2a;border-radius:12px;border:1px solid #1f2d3d;">
                <div style="text-align:center;color:#37474f;font-size:0.9rem;">
                    Metrics appear here during detection
                </div>
            </div>""", unsafe_allow_html=True)

        # ── DETECTION LOOP ───────────────────────
        if st.session_state.recording:
            detector = load_detector()
            audio_mgr = load_audio()

            if st.session_state.state_machine is None:
                st.session_state.state_machine = DrowsinessStateMachine(st.session_state.config)
            else:
                st.session_state.state_machine.update_config(st.session_state.config)

            hud = HUDVisualizer()
            video_path = None

            if source == "Webcam (Live)":
                cap_source = 0
            elif source == "Upload Video File" and uploaded_file is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded_file.read())
                tmp.close()
                video_path = tmp.name
                cap_source = video_path
            elif source == "Sample Demo Video":
                demo_path = "sample_driver.mp4"
                if not os.path.exists(demo_path):
                    with st.spinner("Generating sample demo video…"):
                        generate_synthetic_driver_video(demo_path, duration_sec=15, fps=30)
                cap_source = demo_path
            else:
                st.warning("Please upload a video file first.")
                st.session_state.recording = False
                st.stop()

            cap = cv2.VideoCapture(cap_source)
            if not cap.isOpened():
                st.error("❌ Failed to open video source.")
                st.session_state.recording = False
                st.stop()

            is_webcam = (cap_source == 0)
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_delay = 1.0 / vid_fps if not is_webcam else 0.0
            LOG_EVERY_N = 3
            frame_count = 0

            while st.session_state.recording:
                ret, frame = cap.read()

                # ── Video End Handling ──────────────
                if not ret:
                    if is_webcam:
                        st.warning("⚠️ Webcam disconnected.")
                    else:
                        # Video file finished — STOP, do not loop
                        st.session_state.recording = False
                        video_placeholder.markdown("""
                        <div style="height:380px;display:flex;align-items:center;justify-content:center;
                                    background:#131b2a;border-radius:12px;border:1px solid #0a3d1a;">
                            <div style="text-align:center;">
                                <div style="font-size:3rem;">✅</div>
                                <div style="font-size:1.1rem;margin-top:10px;color:#00e676;font-weight:700;">
                                    Video Analysis Complete
                                </div>
                                <div style="font-size:0.85rem;margin-top:6px;color:#546e7a;">
                                    Check the 📊 Analytics tab for the full session report
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    break
                # ────────────────────────────────────

                frame_count += 1
                st.session_state.total_frames = frame_count

                annotated, state = process_frame(
                    frame, detector, st.session_state.state_machine, hud, st.session_state.config
                )

                if frame_count % LOG_EVERY_N == 0:
                    st.session_state.session_log.append({
                        "timestamp": state.timestamp,
                        "state_label": state.state_label,
                        "left_ear": state.left_ear,
                        "right_ear": state.right_ear,
                        "avg_ear": state.avg_ear,
                        "mar": state.mar,
                        "drowsiness_score": state.drowsiness_score,
                        "eye_closed_duration": state.eye_closed_duration,
                        "total_blinks": state.total_blinks,
                        "total_yawns": state.total_yawns,
                        "pitch": state.pitch,
                        "yaw": state.yaw,
                        "roll": state.roll,
                    })

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width='stretch')

                with metrics_placeholder.container():
                    render_metric_cards(state)

                # Audio alert with cooldown
                now = time.time()
                if st.session_state.config.enable_audio_alerts:
                    if (now - st.session_state.last_alert_time) >= st.session_state.alert_cooldown:
                        if state.trigger_sleep_alert or state.state_label == "SLEEPING":
                            audio_placeholder.markdown(audio_mgr.get_html5_audio_tag("critical"), unsafe_allow_html=True)
                            audio_mgr.play_desktop_async("critical")
                            st.session_state.last_alert_time = now
                        elif state.trigger_drowsy_alert or state.state_label == "DROWSY":
                            audio_placeholder.markdown(audio_mgr.get_html5_audio_tag("warning"), unsafe_allow_html=True)
                            audio_mgr.play_desktop_async("warning")
                            st.session_state.last_alert_time = now

                if not is_webcam and frame_delay > 0:
                    time.sleep(max(0, frame_delay - 0.01))
                else:
                    time.sleep(0.005)

            cap.release()
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception:
                    pass

    # ── ANALYTICS TAB ───────────────────────────
    with tab_analytics:
        render_analytics_tab()

    # ── ABOUT TAB ───────────────────────────────
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
