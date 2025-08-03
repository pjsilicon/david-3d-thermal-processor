import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import warnings
from transformers import pipeline
from PIL import Image
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class EnhancedSurfaceNormalEstimator:
    """Enhanced surface normal estimation with better depth detail extraction"""
    
    def __init__(self, mode='fast'):
        """Initialize enhanced surface normal estimator"""
        self.mode = mode
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Enhanced Surface Normal Estimator using: {self.device}")
        
        # Initialize depth estimation model
        self._load_depth_model()
        
        # Performance tracking
        self.inference_times = []
        
    def _load_depth_model(self):
        """Load depth estimation model"""
        print(f"📦 Loading Intel DPT-Large for enhanced normal estimation...")
        
        try:
            self.depth_pipeline = pipeline(
                task="depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device.type == "mps" else -1,
                torch_dtype=torch.float32
            )
            print(f"✅ Depth model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load depth model: {e}")
            raise RuntimeError("Could not load depth estimation model")
    
    def estimate_surface_normals(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate surface normals with enhanced depth processing
        """
        start_time = time.time()
        
        # Convert frame to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get depth map
        with torch.no_grad():
            try:
                # Get raw depth from model
                depth_result = self.depth_pipeline(pil_image)
                depth_map = np.array(depth_result['predicted_depth'])
                
                # Resize depth to match input frame
                h, w = frame.shape[:2]
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Enhance depth details before normal calculation
                depth_enhanced = self._enhance_depth_details(depth_map, mask)
                
                # Convert enhanced depth to surface normals
                normal_map = self._depth_to_normals_enhanced(depth_enhanced)
                
                # Apply mask and create final result
                if mask is not None:
                    result = self._apply_normal_map_to_face(frame, normal_map, mask)
                else:
                    result = (normal_map * 255).astype(np.uint8)
                
                # Track performance
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                if len(self.inference_times) % 10 == 0:
                    avg_time = np.mean(self.inference_times[-10:])
                    fps = 1.0 / avg_time
                    print(f"⚡ Enhanced Normal: {fps:.1f} FPS ({avg_time*1000:.1f}ms per frame)")
                
                return result
                
            except Exception as e:
                print(f"⚠️ Enhanced normal estimation failed: {e}")
                return frame
    
    def _enhance_depth_details(self, depth_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhance depth map to extract more geometric details
        This is crucial for getting the 3D appearance
        """
        depth = depth_map.astype(np.float32)
        
        # If mask provided, focus enhancement on face area
        if mask is not None:
            mask_binary = (mask > 128).astype(np.float32)
            # Extract face region depth
            face_depth = depth * mask_binary
            
            # Find valid depth range in face area
            valid_depths = face_depth[face_depth > 0]
            if len(valid_depths) > 0:
                depth_min = np.percentile(valid_depths, 1)
                depth_max = np.percentile(valid_depths, 99)
            else:
                depth_min = depth.min()
                depth_max = depth.max()
        else:
            depth_min = depth.min()
            depth_max = depth.max()
        
        # Normalize and stretch contrast
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.ones_like(depth) * 0.5
        
        # Apply multiple enhancement techniques
        
        # 1. Histogram equalization for better contrast
        depth_uint8 = (depth * 255).astype(np.uint8)
        depth_eq = cv2.equalizeHist(depth_uint8).astype(np.float32) / 255.0
        
        # 2. Unsharp masking to enhance details
        depth_blur = cv2.GaussianBlur(depth_eq, (0, 0), 2.0)
        depth_unsharp = cv2.addWeighted(depth_eq, 1.5, depth_blur, -0.5, 0)
        
        # 3. Detail enhancement using bilateral filter
        depth_detail = cv2.bilateralFilter(depth_unsharp, 9, 50, 50)
        
        # 4. Edge-aware enhancement
        edges = cv2.Canny((depth_detail * 255).astype(np.uint8), 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_mask = (edges_dilated > 0).astype(np.float32)
        
        # Enhance edges in depth map
        depth_enhanced = depth_detail * (1 + edge_mask * 0.3)
        
        # Apply mask if provided
        if mask is not None:
            depth_enhanced = depth_enhanced * mask_binary
        
        return np.clip(depth_enhanced, 0, 1)
    
    def _depth_to_normals_enhanced(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert enhanced depth to surface normals with better detail preservation
        """
        # Scale depth for gradient calculation
        # Higher scale = more pronounced normals
        depth_scaled = depth_map * 100.0
        
        # Multi-scale gradient calculation for better detail capture
        # Small scale for fine details
        grad_x_fine = cv2.Sobel(depth_scaled, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_fine = cv2.Sobel(depth_scaled, cv2.CV_64F, 0, 1, ksize=3)
        
        # Large scale for overall shape
        grad_x_coarse = cv2.Sobel(depth_scaled, cv2.CV_64F, 1, 0, ksize=7)
        grad_y_coarse = cv2.Sobel(depth_scaled, cv2.CV_64F, 0, 1, ksize=7)
        
        # Combine scales
        grad_x = grad_x_fine * 0.7 + grad_x_coarse * 0.3
        grad_y = grad_y_fine * 0.7 + grad_y_coarse * 0.3
        
        # Create normal vectors
        h, w = depth_map.shape
        normals = np.zeros((h, w, 3), dtype=np.float64)
        
        # Set components
        normals[:, :, 0] = -grad_x  # X (red)
        normals[:, :, 1] = grad_y   # Y (green)
        normals[:, :, 2] = 2.0      # Z (blue) - smaller value for more color variation
        
        # Normalize
        norm = np.sqrt(normals[:, :, 0]**2 + normals[:, :, 1]**2 + normals[:, :, 2]**2)
        norm = np.maximum(norm, 1e-8)
        
        normals[:, :, 0] /= norm
        normals[:, :, 1] /= norm
        normals[:, :, 2] /= norm
        
        # Convert to RGB visualization
        normal_rgb = self._normals_to_rgb(normals)
        
        return normal_rgb
    
    def _normals_to_rgb(self, normals: np.ndarray) -> np.ndarray:
        """
        Convert normal vectors to RGB visualization
        with enhanced color mapping for better 3D appearance
        """
        # Map from [-1, 1] to [0, 1]
        normal_rgb = np.zeros_like(normals)
        
        # Enhanced mapping with contrast adjustment
        normal_rgb[:, :, 0] = (normals[:, :, 0] + 1.0) * 0.5  # Red: X-axis
        normal_rgb[:, :, 1] = (normals[:, :, 1] + 1.0) * 0.5  # Green: Y-axis
        normal_rgb[:, :, 2] = (normals[:, :, 2] + 1.0) * 0.5  # Blue: Z-axis
        
        # Apply color enhancement
        # Stretch each channel for more vivid colors
        for i in range(3):
            channel = normal_rgb[:, :, i]
            # Find valid range
            valid_pixels = channel[channel > 0]
            if len(valid_pixels) > 0:
                p_low = np.percentile(valid_pixels, 10)
                p_high = np.percentile(valid_pixels, 90)
                if p_high > p_low:
                    channel = (channel - p_low) / (p_high - p_low)
                    channel = np.clip(channel, 0, 1)
                    
                    # Apply gamma for better visual appearance
                    gamma = [0.7, 0.8, 1.2][i]  # Different gamma per channel
                    channel = np.power(channel, gamma)
                    
                    normal_rgb[:, :, i] = channel
        
        # Boost saturation
        normal_rgb_uint8 = (normal_rgb * 255).astype(np.uint8)
        hsv = cv2.cvtColor(normal_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)  # Increase saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Slight brightness boost
        normal_rgb_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        normal_rgb = normal_rgb_uint8.astype(np.float32) / 255.0
        
        return normal_rgb
    
    def _apply_normal_map_to_face(self, frame: np.ndarray, normal_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply the normal map to the face area with proper masking
        """
        # Create binary mask
        mask_binary = (mask > 128).astype(np.float32)
        
        # Slightly erode mask to avoid edge artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
        
        # Create soft edges
        mask_soft = cv2.GaussianBlur(mask_eroded, (9, 9), 0)
        
        # Apply normal map to face area
        result = frame.copy().astype(np.float32)
        normal_rgb = (normal_map * 255).astype(np.float32)
        
        # Blend with soft mask
        for i in range(3):
            result[:, :, i] = result[:, :, i] * (1 - mask_soft) + normal_rgb[:, :, i] * mask_soft
        
        return result.astype(np.uint8)
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        recent_times = self.inference_times[-20:] if len(self.inference_times) > 20 else self.inference_times
        return {
            'average_fps': 1.0 / np.mean(recent_times),
            'average_ms': np.mean(recent_times) * 1000,
            'device': str(self.device),
            'model_mode': self.mode,
            'total_frames': len(self.inference_times)
        }