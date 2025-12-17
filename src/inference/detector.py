import cv2
import torch
import numpy as np
import os
from PIL import Image
from src.data.transforms import get_val_transforms
from src.models.factory import get_model

class EmotionDetector:
    def __init__(self, model_path, model_name='resnet18', device='cpu'):
        self.device = torch.device(device)
        self.model = get_model(model_name, num_classes=7, pretrained=False)
        
        # Load weights
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_val_transforms()
        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        
        # OpenCV Haar Cascade Face Detection (more reliable for cloud deployment)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_emotions(self, frame):
        """
        Detects faces and predicts emotions in a frame.
        Returns the frame with annotations and list of results.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            print(f"Detected {len(faces)} faces")
        else:
            print("No faces detected")
        
        detections = []
        
        for (x, y, w_box, h_box) in faces:
            # Extract face region
            face_roi = frame[y:y+h_box, x:x+w_box]
            
            # Preprocess
            try:
                # Convert to RGB and resize
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                resized_face = cv2.resize(face_rgb, (48, 48))
                img_tensor = self.transform(resized_face).unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                emotion = self.emotions[pred.item()]
                confidence = conf.item()
                
                detections.append({
                    'bbox': (x, y, w_box, h_box),
                    'emotion': emotion,
                    'confidence': confidence
                })
                
                # Draw on frame
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
                        
        return frame, detections
