import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.inference.detector import EmotionDetector

st.set_page_config(page_title="Face Emotion Detection", layout="wide")

st.title("Face Emotion Detection")
st.sidebar.title("Settings")

# model_name = st.sidebar.selectbox("Model", ["resnet18", "mobilenet"])
model_name = "resnet18"
use_webcam = st.sidebar.checkbox("Use Webcam")

@st.cache_resource
def load_detector(name):
    model_path = "experiments/exp1/best_model.pth"
    if not os.path.exists(model_path):
        model_path = None
    return EmotionDetector(model_path=model_path, model_name=name, device='cpu')

detector = load_detector(model_name)

if use_webcam:
    st.header("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    
    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Could not open webcam. Please check permissions and ensure no other app is using it.")
        else:
            st.success("Webcam initialized successfully.")
            
            while run:
                ret, frame = camera.read()
                if not ret or frame is None:
                    st.error("Failed to read frame from webcam.")
                    break
                    
                # Pass BGR frame to detector
                frame, detections = detector.detect_emotions(frame)
                
                # Convert to RGB for Streamlit display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                FRAME_WINDOW.image(frame)
            
            camera.release()
else:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
        processed_img, detections = detector.detect_emotions(img_bgr)
        
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)
        
        st.write("Detections:")
        st.json(detections)
