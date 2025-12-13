import cv2
import mediapipe as mp
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
        
        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def detect_emotions(self, frame):
        """
        Detects faces and predicts emotions in a frame.
        Returns the frame with annotations and list of results.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            print(f"Detected {len(results.detections)} faces")
        else:
            print("No faces detected")
        
        detections = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
                # Ensure bbox is within frame
                x, y = max(0, x), max(0, y)
                w_box, h_box = min(w - x, w_box), min(h - y, h_box)
                
                if w_box > 0 and h_box > 0:
                    face_roi = rgb_frame[y:y+h_box, x:x+w_box]
                    
                    # Preprocess
                    try:
                        # Resize using OpenCV to keep it as numpy array
                        resized_face = cv2.resize(face_roi, (48, 48))
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
