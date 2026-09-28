"""
Audio Alert Synthesizer and Audio Stream Manager for Streamlit & Local Playback.
"""

import os
import io
import base64
import threading
import numpy as np
from scipy.io import wavfile

# Try importing winsound for Windows desktop audio
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AudioAlertManager:
    """
    Generates synthetic alarm sound signals (WAV) and provides audio playback
    mechanisms for both Streamlit Web UI (HTML5 Auto-play) and Desktop Python runtime.
    """
    
    def __init__(self, cache_dir: str = ".audio_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.warning_path = os.path.join(self.cache_dir, "warning_alert.wav")
        self.critical_path = os.path.join(self.cache_dir, "critical_alert.wav")
        
        self.warning_base64 = ""
        self.critical_base64 = ""
        
        self._generate_sounds()
        self.last_played_time = 0.0

    def _generate_sounds(self):
        """Synthesize warning beep and critical siren sound files."""
        sr = 22050
        
        # 1. Warning Alert (Double beep sound, 0.8 sec)
        duration = 0.8
        t = np.linspace(0, duration, int(sr * duration), False)
        # 880 Hz tone pulsed at 6 Hz
        pulse = (np.sin(2 * np.pi * 6 * t) > 0).astype(float)
        tone_warning = np.sin(2 * np.pi * 880 * t) * pulse * 0.7
        audio_warning = (tone_warning * 32767).astype(np.int16)
        wavfile.write(self.warning_path, sr, audio_warning)
        
        with open(self.warning_path, "rb") as f:
            self.warning_base64 = base64.b64encode(f.read()).decode("utf-8")
            
        # 2. Critical Sleep Siren Alert (Urgent alternating dual tone, 1.2 sec)
        duration_crit = 1.2
        t_crit = np.linspace(0, duration_crit, int(sr * duration_crit), False)
        # Alternating siren between 1200 Hz and 700 Hz
        freq = np.where(np.sin(2 * np.pi * 5 * t_crit) > 0, 1200, 700)
        tone_crit = np.sin(2 * np.pi * freq * t_crit) * 0.9
        audio_crit = (tone_crit * 32767).astype(np.int16)
        wavfile.write(self.critical_path, sr, audio_crit)
        
        with open(self.critical_path, "rb") as f:
            self.critical_base64 = base64.b64encode(f.read()).decode("utf-8")

    def get_html5_audio_tag(self, alert_type: str = "warning") -> str:
        """
        Generate hidden HTML5 audio autoplay tag for Streamlit web browser audio alert.
        """
        b64_data = self.critical_base64 if alert_type == "critical" else self.warning_base64
        html_code = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/wav;base64,{b64_data}" type="audio/wav">
        </audio>
        """
        return html_code

    def play_desktop_async(self, alert_type: str = "warning"):
        """Play sound locally using winsound on Windows in a background thread."""
        if not HAS_WINSOUND:
            return
            
        def _play():
            try:
                path = self.critical_path if alert_type == "critical" else self.warning_path
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass
                
        threading.Thread(target=_play, daemon=True).start()
