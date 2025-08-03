import cv2
import numpy as np
from typing import Dict, Optional, Tuple

class FaceBodyTracker:
    def __init__(self):
        # Use OpenCV's built-in face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enhanced tracking for professional quality
        self.prev_landmarks = None
        self.prev_face_bbox = None
        self.tracking_confidence = 1.0
        self.frame_count = 0
        self.smooth_factor = 0.3  # For temporal smoothing
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        
        results = {
            'face_landmarks': None,
            'face_mask': None,
            'body_landmarks': None,
            'body_mask': None,
            'combined_mask': None,
            'tracking_confidence': 0.0
        }
        
        # Enhanced face detection with multiple scales for better accuracy
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # More sensitive detection
            minNeighbors=4,   # Slightly less strict for moving faces
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        current_face = None
        
        if len(faces) > 0:
            # If we have previous face, find the closest one for temporal consistency
            if self.prev_face_bbox is not None:
                current_face = self._find_closest_face(faces, self.prev_face_bbox)
            else:
                # Use the largest face for initial detection
                current_face = max(faces, key=lambda x: x[2] * x[3])
            
            x, y, w, h = current_face
            
            # Apply temporal smoothing for professional stability
            if self.prev_face_bbox is not None:
                prev_x, prev_y, prev_w, prev_h = self.prev_face_bbox
                # Smooth the face box position to reduce jitter
                x = int(prev_x * self.smooth_factor + x * (1 - self.smooth_factor))
                y = int(prev_y * self.smooth_factor + y * (1 - self.smooth_factor))
                w = int(prev_w * self.smooth_factor + w * (1 - self.smooth_factor))
                h = int(prev_h * self.smooth_factor + h * (1 - self.smooth_factor))
            
            # Update tracking state
            self.prev_face_bbox = (x, y, w, h)
            
            # Create professional-quality face mask with better precision
            results['face_mask'] = self._create_adaptive_face_mask(frame.shape, x, y, w, h)
            results['combined_mask'] = results['face_mask']
            
            # Create enhanced landmarks
            results['face_landmarks'] = self._create_enhanced_landmarks(x, y, w, h)
            
            # Higher confidence for tracked faces
            confidence = 0.9 if self.prev_face_bbox is not None else 0.8
            results['tracking_confidence'] = confidence
            self.tracking_confidence = confidence
            
        else:
            # No face detected - use previous frame's data if available (short-term tracking)
            if self.prev_face_bbox is not None and self.frame_count < 10:
                # Use previous mask for a few frames to handle brief occlusions
                x, y, w, h = self.prev_face_bbox
                results['face_mask'] = self._create_adaptive_face_mask(frame.shape, x, y, w, h)
                results['combined_mask'] = results['face_mask']
                results['face_landmarks'] = self._create_enhanced_landmarks(x, y, w, h)
                results['tracking_confidence'] = 0.6  # Lower confidence for predicted frames
                self.tracking_confidence = 0.6
            else:
                # Reset tracking
                self.prev_face_bbox = None
                self.frame_count = 0
                results['tracking_confidence'] = 0.0
                self.tracking_confidence = 0.0
            
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
    
    def _find_closest_face(self, faces, prev_bbox):
        """Find the face closest to the previous frame's face for temporal consistency."""
        prev_x, prev_y, prev_w, prev_h = prev_bbox
        prev_center = (prev_x + prev_w//2, prev_y + prev_h//2)
        
        closest_face = None
        min_distance = float('inf')
        
        for face in faces:
            x, y, w, h = face
            center = (x + w//2, y + h//2)
            distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_face = face
        
        return closest_face
    
    def _create_adaptive_face_mask(self, frame_shape: Tuple, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Create an adaptive, professional-quality face mask that follows facial movement."""
        frame_h, frame_w = frame_shape[:2]
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        # More precise face area calculation
        center = (x + w // 2, y + h // 2)
        
        # Adaptive sizing based on face detection confidence
        face_width = int(w * 0.6)   # More generous face coverage
        face_height = int(h * 0.7)  # Better vertical coverage
        
        # Enhanced upper body detection
        upper_body_extension = int(h * 0.5)  # More conservative upper body
        total_height = face_height + upper_body_extension
        
        # Better center positioning for upper body
        extended_center = (center[0], center[1] + upper_body_extension // 3)
        
        # Create main elliptical mask with better proportions
        cv2.ellipse(mask, extended_center, (face_width, total_height), 0, 0, 360, 255, -1)
        
        # Add additional face-specific ellipse for better definition
        face_only_width = int(w * 0.5)
        face_only_height = int(h * 0.6)
        cv2.ellipse(mask, center, (face_only_width, face_only_height), 0, 0, 360, 255, -1)
        
        # Professional edge smoothing with adaptive blur
        blur_size = max(5, min(21, int(w * 0.05)))  # Adaptive blur based on face size
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # Enhance edges for better blending
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def _create_enhanced_landmarks(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Create enhanced landmarks with better face mapping."""
        landmarks = []
        
        # Create more detailed facial landmark grid
        for i in range(7):  # Increased resolution
            for j in range(7):
                px = x + (w * i / 6)
                py = y + (h * j / 6)
                
                # Enhanced depth calculation with more realistic face curvature
                center_x, center_y = 3, 3
                dist_from_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                # More natural depth falloff
                max_dist = np.sqrt(center_x**2 + center_y**2)
                normalized_dist = dist_from_center / max_dist
                
                # Natural face depth curve (closer at center, further at edges)
                pz = 0.3 + 0.5 * np.exp(-normalized_dist * 2.5)
                landmarks.append([px, py, pz])
        
        return np.array(landmarks)