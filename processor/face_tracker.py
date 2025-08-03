import cv2
import numpy as np
from typing import Dict, Optional, Tuple

class FaceBodyTracker:
    def __init__(self):
        # Use OpenCV's built-in face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.prev_landmarks = None
        self.tracking_confidence = 1.0
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = {
            'face_landmarks': None,
            'face_mask': None,
            'body_landmarks': None,
            'body_mask': None,
            'combined_mask': None,
            'tracking_confidence': 0.0
        }
        
        # Detect faces using OpenCV
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Create face mask
            results['face_mask'] = self._create_simple_face_mask(frame.shape, x, y, w, h)
            results['combined_mask'] = results['face_mask']
            
            # Create simple landmarks for the face center and edges
            results['face_landmarks'] = self._create_simple_landmarks(x, y, w, h)
            
            results['tracking_confidence'] = 0.8
        else:
            results['tracking_confidence'] = 0.0
            
        return results
    
    def _create_simple_face_mask(self, frame_shape: Tuple, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Create a focused face and upper body mask"""
        frame_h, frame_w = frame_shape[:2]
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        # Create a more conservative face-focused mask
        center = (x + w // 2, y + h // 2)
        
        # Make the mask more focused - smaller width, include some upper body height
        face_width = int(w * 0.45)  # Reduced from 0.6 to be more conservative
        face_height = int(h * 0.55)  # Reduced from 0.8 to focus on face
        
        # Add upper body area by extending the mask downward
        upper_body_extension = int(h * 0.4)  # Extend down for upper body/shoulders
        total_height = face_height + upper_body_extension
        
        # Adjust center to account for upper body extension
        extended_center = (center[0], center[1] + upper_body_extension // 2)
        
        # Create face + upper body ellipse
        cv2.ellipse(mask, extended_center, (face_width, total_height), 0, 0, 360, 255, -1)
        
        # Add a separate, smaller ellipse focused just on the face for more definition
        cv2.ellipse(mask, center, (face_width, face_height), 0, 0, 360, 255, -1)
        
        # Lighter blur for more defined edges while keeping some softness
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _create_simple_landmarks(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Create simple landmarks for depth estimation"""
        landmarks = []
        
        # Create a grid of points across the face
        for i in range(5):
            for j in range(5):
                px = x + (w * i / 4)
                py = y + (h * j / 4)
                # Simulate depth (closer at center)
                pz = 0.5 + 0.3 * np.exp(-((i-2)**2 + (j-2)**2) / 4)
                landmarks.append([px, py, pz])
        
        return np.array(landmarks)