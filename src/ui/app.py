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
    st.header("Webcam Capture")
    st.write("Click the button below to take a picture. (Note: Live video processing is not supported on Streamlit Cloud without additional setup).")
    
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Run detection
        processed_img, detections = detector.detect_emotions(cv2_img)
        
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)
        st.write("Detections:")
        st.json(detections)
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
