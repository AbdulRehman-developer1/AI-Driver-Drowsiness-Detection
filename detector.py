"""
MediaPipe FaceLandmarker wrapper and landmark extractor.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"


class FaceLandmarkerDetector:
    """
    Encapsulates MediaPipe Face Landmarker detection and landmark extraction.
    Downloads face_landmarker.task model automatically if missing.
    """
    
    def __init__(self, model_path: str = MODEL_FILENAME, num_faces: int = 1):
        self.model_path = model_path
        self.num_faces = num_faces
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe FaceLandmarker model to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model downloaded successfully.")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process BGR frame and extract facial landmarks.
        
        Returns:
            (face_detected, landmarks_pixel, landmarks_normalized)
            - landmarks_pixel: np.ndarray of shape (478, 3) in pixel coordinates (x_px, y_px, z)
            - landmarks_normalized: np.ndarray of shape (478, 3) normalized (0..1)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, None, None
            
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return False, None, None
            
        face_landmarks = detection_result.face_landmarks[0]
        
        landmarks_norm = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks
        ], dtype=np.float64)
        
        landmarks_px = np.array([
            [lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks
        ], dtype=np.float64)
        
        return True, landmarks_px, landmarks_norm

    def close(self):
        """Release landmarker resources."""
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
