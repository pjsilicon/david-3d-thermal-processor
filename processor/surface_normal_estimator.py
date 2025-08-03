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

class SurfaceNormalEstimator:
    """Surface normal estimation using depth-to-normal conversion with M4 GPU acceleration"""
    
    def __init__(self, mode='fast'):
        """
        Initialize surface normal estimator
        Modes: 'fast', 'medium', 'quality' - all use depth-to-normal conversion
        """
        self.mode = mode
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Surface Normal Estimator using: {self.device}")
        
        # Initialize depth estimation model for normal calculation
        self._load_depth_model()
        
        # Temporal smoothing
        self.temporal_buffer = []
        self.max_buffer_size = 3
        
        # Performance tracking
        self.inference_times = []
        
    def _load_depth_model(self):
        """Load depth estimation model for normal calculation"""
        print(f"📦 Loading Intel DPT-Large for normal estimation...")
        
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
        Estimate surface normals from frame using depth-to-normal conversion
        
        Args:
            frame: Input RGB frame (H, W, 3)
            mask: Optional mask to focus on specific regions
            
        Returns:
            normal_map: Surface normal map (H, W, 3) with RGB encoding of normals
        """
        start_time = time.time()
        
        # Convert frame to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get depth map
        with torch.no_grad():
            try:
                depth_result = self.depth_pipeline(pil_image)
                depth_map = np.array(depth_result['predicted_depth'])
                
                # Resize depth to match input frame
                h, w = frame.shape[:2]
                depth_map = cv2.resize(depth_map, (w, h))
                
                # Convert depth to surface normals - only for the face area
                if mask is not None:
                    # Create a masked depth map for better normal calculation
                    mask_binary = (mask > 128).astype(np.float32)
                    masked_depth = depth_map * mask_binary
                    
                    # Convert to normals
                    normal_map = self._depth_to_normals(masked_depth)
                    
                    # Create the final output - face replaced with normals, background unchanged
                    result = frame.copy()
                    mask_3ch = np.stack([mask_binary] * 3, axis=-1).astype(bool)
                    result[mask_3ch] = (normal_map * 255).astype(np.uint8)[mask_3ch]
                    
                    # No temporal smoothing for now to see raw results
                    # normal_map = self._apply_temporal_smoothing(normal_map)
                else:
                    # No mask, process entire frame
                    normal_map = self._depth_to_normals(depth_map)
                    result = (normal_map * 255).astype(np.uint8)
                
                # Track performance
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                if len(self.inference_times) % 10 == 0:
                    avg_time = np.mean(self.inference_times[-10:])
                    fps = 1.0 / avg_time
                    print(f"⚡ Surface Normal: {fps:.1f} FPS ({avg_time*1000:.1f}ms per frame)")
                
                return result
                
            except Exception as e:
                print(f"⚠️ Surface normal estimation failed: {e}")
                return frame  # Return original frame on failure
    
    def _depth_to_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to surface normals using gradient computation
        This creates the 3D geometric visualization effect
        """
        # Ensure depth map is float32 and normalize it
        depth = depth_map.astype(np.float32)
        
        # Normalize depth to a consistent range
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Apply slight smoothing but preserve details
        depth = cv2.bilateralFilter(depth, 9, 75, 75)
        
        # Scale depth for better gradient calculation
        # This is crucial - we need sufficient depth variation
        depth_scaled = depth * 50.0  # Scale up to get meaningful gradients
        
        # Calculate gradients with multiple scales for better detail
        # Use larger kernel for smoother normals
        grad_x = cv2.Sobel(depth_scaled, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_scaled, cv2.CV_64F, 0, 1, ksize=5)
        
        # Create surface normal vectors
        h, w = depth.shape
        
        # Initialize normal vectors
        normals = np.zeros((h, w, 3), dtype=np.float64)
        
        # The normal is computed as the cross product of the tangent vectors
        # dz/dx and dz/dy give us the rate of change
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        
        # X component (red channel) - left/right surface orientation
        normals[:, :, 0] = -grad_x
        
        # Y component (green channel) - up/down surface orientation  
        normals[:, :, 1] = grad_y  # Positive for proper orientation
        
        # Z component (blue channel) - forward orientation
        # Use a smaller value to make X and Y components more prominent
        normals[:, :, 2] = 1.0
        
        # Normalize the normal vectors to unit length
        norm = np.sqrt(normals[:, :, 0]**2 + normals[:, :, 1]**2 + normals[:, :, 2]**2)
        norm = np.maximum(norm, 1e-8)  # Avoid division by zero
        
        normals[:, :, 0] /= norm
        normals[:, :, 1] /= norm  
        normals[:, :, 2] /= norm
        
        # Convert from [-1, 1] to [0, 1] range for RGB visualization
        # This mapping is crucial for the colorful appearance
        normal_map = np.zeros_like(normals)
        normal_map[:, :, 0] = (normals[:, :, 0] + 1.0) * 0.5  # Red: left/right
        normal_map[:, :, 1] = (normals[:, :, 1] + 1.0) * 0.5  # Green: up/down
        normal_map[:, :, 2] = (normals[:, :, 2] + 1.0) * 0.5  # Blue: forward
        
        # Enhance contrast for more vivid colors
        normal_map = self._enhance_normal_contrast(normal_map)
        
        return np.clip(normal_map, 0, 1)
    
    def _enhance_normal_contrast(self, normal_map: np.ndarray) -> np.ndarray:
        """
        Enhance the contrast and color vibrancy of the normal map
        to create the characteristic DAViD-style appearance
        """
        enhanced = normal_map.copy()
        
        # Apply contrast enhancement to each channel
        # This makes the colors more vivid and distinct
        for i in range(3):
            channel = enhanced[:, :, i]
            
            # Stretch the histogram for better contrast
            p_low, p_high = np.percentile(channel[channel > 0], [5, 95])
            channel = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            
            # Apply gamma correction for better visual appearance
            # Different gamma for each channel to enhance color separation
            if i == 0:  # Red channel (X-axis)
                channel = np.power(channel, 0.8)
            elif i == 1:  # Green channel (Y-axis)
                channel = np.power(channel, 0.9)
            else:  # Blue channel (Z-axis)
                channel = np.power(channel, 1.1)
            
            enhanced[:, :, i] = channel
        
        # Increase overall saturation
        # Convert to HSV to manipulate saturation
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        hsv = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Increase saturation
        hsv[:, :, 1] = hsv[:, :, 1] * 1.5  # Increase saturation by 50%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to RGB
        enhanced_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
        
        return enhanced
    
    def _apply_temporal_smoothing(self, normal_map: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce flicker"""
        self.temporal_buffer.append(normal_map.copy())
        
        # Keep buffer size manageable
        if len(self.temporal_buffer) > self.max_buffer_size:
            self.temporal_buffer.pop(0)
        
        if len(self.temporal_buffer) > 1:
            # Weighted average favoring recent frames
            weights = np.linspace(0.3, 1.0, len(self.temporal_buffer))
            weights = weights / weights.sum()
            
            smoothed = np.zeros_like(normal_map)
            for i, weight in enumerate(weights):
                smoothed += self.temporal_buffer[i] * weight
            
            return smoothed
        
        return normal_map
    
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

class SurfaceNormalRenderer:
    """Specialized renderer for 3D surface normal visualization"""
    
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def create_3d_normal_overlay(
        self,
        frame: np.ndarray,
        normal_map: np.ndarray,
        mask: np.ndarray,
        face_landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create complete face replacement with 3D normal map visualization
        This replaces the face entirely with the geometric normal map
        """
        h, w = frame.shape[:2]
        
        # Ensure normal_map is the same size as frame
        if normal_map.shape[:2] != (h, w):
            normal_map = cv2.resize(normal_map, (w, h))
        
        # The normal_map is already the complete 3D visualization
        # It should already contain the face area with proper normals
        # So we just return it directly (it's already been masked in estimate_surface_normals)
        
        return normal_map
    
    def _create_soft_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create soft feathered mask for seamless integration"""
        mask_float = mask.astype(np.float32) / 255.0
        
        # Slight expansion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_float = cv2.dilate(mask_float, kernel, iterations=1)
        
        # Soft feathering
        mask_float = cv2.GaussianBlur(mask_float, (15, 15), 0)
        
        return mask_float
    
    def _add_3d_glow_effect(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add subtle glow effect to enhance 3D appearance"""
        # Create edge glow
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edge_glow = cv2.GaussianBlur(edges.astype(np.float32), (7, 7), 0) / 255.0
        
        # Add subtle white glow around edges
        glow_color = np.array([255, 255, 255])  # White glow
        for i in range(3):
            image[:, :, i] += edge_glow * glow_color[i] * 0.1
        
        return image