import urllib.request
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import mediapipe as mp

#1. Page Configuration
st.set_page_config(
    page_title="GazeMate AI | Mobile Tracking",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Light & Sky-Blue Custom CSS Theme
st.markdown("""
    <style>
        /* Main background - Light Theme */
        .stApp {
            background-color: #F4F8FA;
            color: #1E293B;
        }
        
        /* Metric cards in Sky-Blue style */
        .metric-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 16px;
            border: 2px solid #BAE6FD;
            text-align: center;
            color: #0F172A;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        /* Gradient Sky-Blue Title */
        .title-text {
            font-size: 2.2rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #0284C7, #38BDF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Info guide box - Sky Blue Accent */
        .info-box {
            background-color: #E0F2FE;
            border-left: 5px solid #0284C7;
            padding: 16px;
            border-radius: 8px;
            color: #0369A1 !important;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        /* Visible Sky-Blue Action Button */
        div[data-testid="stActionButton"] button,
        .element-container button {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            border: none !important;
        }
        
        div[data-testid="stActionButton"] button:hover {
            background-color: #0369A1 !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Model Buffer Downloader
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

@st.cache_resource
def get_model_buffer():
    req = urllib.request.urlopen(MODEL_URL)
    return req.read()

# Safe initialization block
GLOBAL_DETECTOR = None
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_buffer = get_model_buffer()
    base_options = python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    GLOBAL_DETECTOR = vision.FaceLandmarker.create_from_options(options)
except Exception:
    GLOBAL_DETECTOR = None

# 4. WebRTC Video Processor
class EyeGazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = GLOBAL_DETECTOR

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.detector is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)

            if result.face_landmarks:
                h, w, _ = img.shape
                for landmark in result.face_landmarks[0]:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # Draw Sky-Blue tracking points
                    cv2.circle(img, (cx, cy), 1, (248, 189, 56), -1)

        return frame.from_ndarray(img, format="bgr24")

# 5. Header Layout
col1, col2 = st.columns([1, 4])
with col1:
    st.title("👁️")
with col2:
    st.markdown('<p class="title-text">GazeMate AI</p>', unsafe_allow_html=True)
    st.caption("Real-time Mobile Web Eye-Gaze Tracking Subsystem")

st.markdown("---")

# 6. Light Theme Quick Start Guide
st.markdown("""
<div class="info-box">
    <strong>📱 Quick Start Guide:</strong><br>
    1. Position your face clearly in front of your device camera.<br>
    2. Click <strong>START</strong> below and grant camera permissions.<br>
    3. Ensure good lighting for optimal landmark detection accuracy.
</div>
""", unsafe_allow_html=True)

# 7. Live WebRTC Feed
st.subheader("📹 Live Camera Feed")

if GLOBAL_DETECTOR is None:
    st.warning("⚠️ Streamlit Cloud is currently running Python 3.14. Please set Python Version to 3.11 in your App Settings under Manage App.")

webrtc_streamer(
    key="gazemate-mobile-stream",
    video_processor_factory=EyeGazeProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": "user"}
        },
        "audio": False
    },
    async_processing=True
)

st.markdown("---")

# 8. Metric Cards (Sky Blue / White Theme)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric-card"><strong>Status</strong><br><span style="color:#0284C7; font-weight: bold;">Ready</span></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><strong>Platform</strong><br>Mobile Web</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><strong>Backend</strong><br>MediaPipe Tasks</div>', unsafe_allow_html=True)

# 9. Sidebar Controls
with st.sidebar:
    st.header("⚙️ App Controls")
    st.success("MediaPipe Engine Active")
    st.info("Resolution: Standard Mobile WebRTC")
    st.markdown("---")
    st.header("💻 Desktop Version")
    st.write("Need system-wide desktop cursor control?")
    st.markdown("[👉 Download Windows Executable (.exe)](https://github.com/vennelakonduru/GazeMate/releases)")
    st.markdown("---")
    st.caption("GazeMate Project • Dual-Platform Deployment")
