import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
from collections import deque

class DepthHologramGenerator:
    """Creates DAVID-style blue holographic depth overlay"""
    
    def __init__(self):
        self.temporal_buffer = deque(maxlen=5)  # For temporal smoothing
        
    def create_holographic_overlay(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        mask: np.ndarray,
        face_landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create the EXACT blue holographic effect from DAVID
        
        This is the most important function - it creates the signature look!
        """
        h, w = frame.shape[:2]
        
        # Step 1: Process depth map
        depth_norm = self._process_depth(depth_map)
        
        # Step 2: Temporal smoothing
        depth_norm = self._apply_temporal_smoothing(depth_norm)
        
        # Step 3: Create the blue-cyan gradient (DAVID signature look)
        overlay = self._create_blue_cyan_overlay(depth_norm, h, w)
        
        # Step 4: Add holographic edges
        overlay = self._add_holographic_edges(overlay, depth_norm)
        
        # Step 5: Create soft mask
        mask_soft = self._create_soft_mask(mask)
        
        # Step 6: Apply holographic blending
        result = self._blend_holographic(frame, overlay, mask_soft)
        
        return result
    
    def _process_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Process and normalize depth map"""
        depth_norm = depth_map.copy()
        
        # Remove outliers
        depth_norm = np.clip(depth_norm, 
                            np.percentile(depth_norm[depth_norm > 0], 5),
                            np.percentile(depth_norm[depth_norm > 0], 95))
        
        # Normalize to 0-1
        depth_min = depth_norm[depth_norm > 0].min() if np.any(depth_norm > 0) else 0
        depth_max = depth_norm.max()
        
        if depth_max > depth_min:
            depth_norm = (depth_norm - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.ones_like(depth_norm) * 0.5
            
        # Invert so closer is brighter (DAVID style)
        depth_norm = 1.0 - depth_norm
        
        # Enhance contrast with gamma correction
        depth_norm = np.power(depth_norm, 0.7)
        
        return depth_norm
    
    def _apply_temporal_smoothing(self, depth_norm: np.ndarray) -> np.ndarray:
        """Smooth depth across frames to prevent flicker"""
        self.temporal_buffer.append(depth_norm)
        
        if len(self.temporal_buffer) > 1:
            # Weighted average, favoring recent frames
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[-len(self.temporal_buffer):]
            weights = weights / weights.sum()
            
            smoothed = np.zeros_like(depth_norm)
            for i, weight in enumerate(weights):
                smoothed += self.temporal_buffer[i] * weight
            
            return smoothed
        
        return depth_norm
    
    def _create_blue_cyan_overlay(self, depth_norm: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create the signature DAVID blue-cyan gradient"""
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        
        # DAVID color mapping (this is the key!)
        # Blue channel: Always high, slightly modulated
        overlay[:, :, 0] = 180 + depth_norm * 75  # 180-255
        
        # Green channel: Creates cyan for close depths
        overlay[:, :, 1] = depth_norm * 200  # 0-200
        
        # Red channel: Subtle highlights only
        overlay[:, :, 2] = np.power(depth_norm, 2) * 100  # 0-100, quadratic
        
        return overlay
    
    def _add_holographic_edges(self, overlay: np.ndarray, depth_norm: np.ndarray) -> np.ndarray:
        """Add glowing edges for holographic effect"""
        # Detect edges in depth
        depth_edges = cv2.Sobel(depth_norm.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
        depth_edges = np.abs(depth_edges)
        depth_edges = gaussian_filter(depth_edges, sigma=1.0)
        
        # Create edge glow
        edge_glow = np.zeros_like(overlay)
        edge_glow[:, :, 0] = 255  # Bright blue
        edge_glow[:, :, 1] = 255  # Cyan
        edge_glow[:, :, 2] = 200  # Slight white
        
        # Add to overlay
        edge_mask = (depth_edges > 0.1).astype(np.float32)
        edge_mask = gaussian_filter(edge_mask, sigma=2.0)
        overlay += edge_glow * edge_mask[:, :, np.newaxis] * 0.3
        
        return overlay
    
    def _create_soft_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create soft feathered mask"""
        mask_float = mask.astype(np.float32) / 255.0
        
        # Expand slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_float = cv2.dilate(mask_float, kernel, iterations=1)
        
        # Heavy feathering for professional look
        mask_float = gaussian_filter(mask_float, sigma=10.0)
        
        return mask_float
    
    def _blend_holographic(self, frame: np.ndarray, overlay: np.ndarray, 
                          mask: np.ndarray) -> np.ndarray:
        """Apply DAVID-style holographic blending"""
        overlay_norm = overlay / 255.0
        frame_float = frame.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        # Additive blend for glow
        additive = frame_float + overlay_norm * mask_3ch * 0.4
        
        # Overlay blend for depth
        overlay_blend = np.where(
            frame_float < 0.5,
            2 * frame_float * overlay_norm,
            1 - 2 * (1 - frame_float) * (1 - overlay_norm)
        )
        
        # Combine blend modes (DAVID uses this mix)
        result = frame_float * (1 - mask_3ch * 0.6) + \
                 (additive * 0.3 + overlay_blend * 0.7) * mask_3ch * 0.6
        
        # Slight blue color correction
        result[:, :, 0] *= 1.05  # Boost blue
        result[:, :, 1] *= 0.98  # Slight green reduction
        result[:, :, 2] *= 0.95  # Slight red reduction
        
        # Convert back
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result