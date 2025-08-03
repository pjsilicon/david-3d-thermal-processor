import cv2
import numpy as np
from typing import Optional
import os

# Try to import GPU accelerated version
try:
    from .gpu_depth_estimator import GPUDepthEstimator, FaceDepthProcessor
    GPU_AVAILABLE = True
    print("🚀 GPU depth estimation available!")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"⚠️ GPU depth estimation not available: {e}")
    print("📱 Using CPU-only geometric depth estimation")

class DepthEstimator:
    def __init__(self, mode='fast', use_gpu=True):
        """
        Depth estimator with optional GPU acceleration
        Modes: 'fast', 'medium', 'quality'
        use_gpu: Try to use GPU-accelerated AI models if available
        """
        self.mode = mode
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Initialize appropriate estimator
        if self.use_gpu:
            try:
                self.gpu_estimator = GPUDepthEstimator(mode=mode)
                self.face_processor = FaceDepthProcessor()
                print(f"✅ Using GPU depth estimation ({mode} mode)")
            except Exception as e:
                print(f"⚠️ GPU estimator failed to initialize: {e}")
                self.use_gpu = False
        
        if not self.use_gpu:
            print(f"📱 Using CPU geometric depth estimation ({mode} mode)")
        
        # For temporal consistency (CPU mode)
        self.last_depth = None
        self.frame_counter = 0
        self.process_every_n_frames = 1
        
    def estimate_frame(self, frame: np.ndarray, mask: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate depth map for frame using best available method"""
        if self.use_gpu:
            try:
                # Use GPU-accelerated AI depth estimation
                depth_map = self.gpu_estimator.estimate_frame(frame, mask)
                
                # Enhance with face landmarks if available
                if face_landmarks is not None:
                    depth_map = self.face_processor.enhance_face_depth(depth_map, face_landmarks)
                
                return depth_map
                
            except Exception as e:
                print(f"⚠️ GPU depth estimation failed: {e}, falling back to CPU")
                self.use_gpu = False  # Disable GPU for future frames
        
        # Fallback to CPU geometric estimation
        return self._estimate_geometric_depth(frame, mask)
    
    def _estimate_geometric_depth(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Geometric depth estimation using face shape assumptions"""
        h, w = frame.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        # Find face region
        rows, cols = np.where(mask > 128)
        if len(rows) == 0:
            return depth_map
            
        # Get face bounding box
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Face center
        center_y = (min_row + max_row) / 2
        center_x = (min_col + max_col) / 2
        
        # Create depth based on distance from center (nose is closest)
        for y in range(h):
            for x in range(w):
                if mask[y, x] > 128:
                    # Distance from center
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    face_radius = max(max_row - min_row, max_col - min_col) / 2
                    
                    # Normalize distance
                    norm_dist = min(dist_from_center / face_radius, 1.0)
                    
                    # Create depth profile (closer at center, farther at edges)
                    depth_value = 1.0 - (norm_dist ** 0.7) * 0.7
                    depth_map[y, x] = depth_value
        
        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (31, 31), 0)
        
        # Apply mask again after smoothing
        depth_map = depth_map * (mask > 128).astype(np.float32)
        
        return depth_map
        
    def _gaussian_2d(self, height, width, amplitude):
        """Create 2D Gaussian blob"""
        y, x = np.ogrid[:height, :width]
        cy, cx = height / 2, width / 2
        
        gaussian = amplitude * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * (min(height, width) / 4)**2)
        )
        
        return gaussian
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from the depth estimator"""
        if self.use_gpu and hasattr(self, 'gpu_estimator'):
            return self.gpu_estimator.get_performance_stats()
        else:
            return {
                'device': 'CPU',
                'model_mode': self.mode,
                'type': 'geometric'
            }