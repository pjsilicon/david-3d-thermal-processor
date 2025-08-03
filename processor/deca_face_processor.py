"""
DECA-based 3D Face Processor for UV Map Visualization
Creates the UV-mapped 3D face effect similar to Microsoft DAViD demo
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Dict, Tuple
import warnings

# Try to import FLAME model components
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesVertex
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    warnings.warn("pytorch3d not available. Some features will be limited.")

class DECAFaceProcessor:
    """
    DECA-inspired face processor that creates UV-mapped 3D face visualization
    """
    def __init__(self, mode='fast'):
        self.mode = mode
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🎯 DECA Face Processor using: {self.device}")
        
        # Initialize FLAME-like face model parameters
        self.n_shape_params = 100
        self.n_exp_params = 50
        self.image_size = 224
        
        # Pre-compute UV coordinates for face mesh
        self._init_uv_coordinates()
        
    def _init_uv_coordinates(self):
        """Initialize UV coordinates for face mesh visualization"""
        # Create a simple grid of UV coordinates
        # In a real DECA implementation, these would come from FLAME model
        n_vertices = 5023  # Standard FLAME model vertex count
        
        # Create UV coordinates that map to face regions
        grid_size = int(np.ceil(np.sqrt(n_vertices)))
        u = np.linspace(0, 1, grid_size)
        v = np.linspace(0, 1, grid_size)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Flatten and create UV coordinates
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        
        # Take only the required number of vertices
        self.uv_coords = np.stack([u_flat[:n_vertices], v_flat[:n_vertices]], axis=1)
        
        # Create face-specific UV mapping that emphasizes facial features
        # Center UV coordinates around face region
        center_u, center_v = 0.5, 0.5
        for i in range(len(self.uv_coords)):
            u, v = self.uv_coords[i]
            # Apply radial mapping for better face coverage
            dist = np.sqrt((u - center_u)**2 + (v - center_v)**2)
            angle = np.arctan2(v - center_v, u - center_u)
            
            # Remap to emphasize face regions
            new_dist = dist ** 0.7  # Compress outer regions
            self.uv_coords[i, 0] = center_u + new_dist * np.cos(angle)
            self.uv_coords[i, 1] = center_v + new_dist * np.sin(angle)
            
    def process_face(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process face to create UV-mapped 3D visualization
        
        Args:
            frame: Input RGB frame
            mask: Optional face mask
            
        Returns:
            UV-mapped face visualization
        """
        h, w = frame.shape[:2]
        
        # Create UV-mapped colors based on coordinates
        # This creates the characteristic colorful 3D face effect
        uv_face = self._generate_uv_face(frame, mask)
        
        return uv_face
    
    def _generate_uv_face(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate UV-mapped face visualization"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        if mask is None:
            # Create a default face mask
            center_x, center_y = w // 2, h // 2
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = dist_from_center <= min(h, w) * 0.3
            mask = mask.astype(np.float32)
        else:
            mask = (mask > 128).astype(np.float32)
            
        # Create UV coordinate grid for the image
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Normalize coordinates to [0, 1]
        x_norm = x_coords / w
        y_norm = y_coords / h
        
        # Find face region bounds
        if mask.sum() > 0:
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # Remap coordinates within face region
            face_width = x_max - x_min
            face_height = y_max - y_min
            
            # Create local UV coordinates within face region
            u_local = (x_coords - x_min) / face_width
            v_local = (y_coords - y_min) / face_height
            
            # Apply face-specific UV mapping
            # Create depth-like effect based on position
            center_u = 0.5
            center_v = 0.5
            
            # Distance from face center
            dist_u = u_local - center_u
            dist_v = v_local - center_v
            dist = np.sqrt(dist_u**2 + dist_v**2)
            
            # Create 3D-like UV colors
            # Red channel: horizontal position (left/right facing)
            # Green channel: vertical position (up/down facing)  
            # Blue channel: depth/forward facing
            
            # Add face geometry simulation
            # Nose region (center, protruding)
            nose_mask = (np.abs(dist_u) < 0.1) & (np.abs(v_local - 0.5) < 0.2)
            
            # Eye regions (sunken)
            eye_mask = ((np.abs(dist_u - 0.2) < 0.1) | (np.abs(dist_u + 0.2) < 0.1)) & \
                      (np.abs(v_local - 0.35) < 0.1)
            
            # Mouth region
            mouth_mask = (np.abs(dist_u) < 0.2) & (np.abs(v_local - 0.65) < 0.1)
            
            # Create UV colors with geometric features
            r_channel = np.clip(0.5 + dist_u * 2.0, 0, 1)
            g_channel = np.clip(0.5 + (v_local - 0.5) * 2.0, 0, 1)
            b_channel = np.clip(0.5 + (1 - dist) * 0.8, 0, 1)
            
            # Enhance geometric features
            b_channel[nose_mask] = np.clip(b_channel[nose_mask] + 0.3, 0, 1)
            b_channel[eye_mask] = np.clip(b_channel[eye_mask] - 0.2, 0, 1)
            g_channel[mouth_mask] = np.clip(g_channel[mouth_mask] + 0.1, 0, 1)
            
            # Add subtle gradient variations for smoother appearance
            angle = np.arctan2(dist_v, dist_u)
            r_channel += 0.1 * np.sin(angle * 2)
            g_channel += 0.1 * np.cos(angle * 3)
            
            # Create smooth transitions
            r_channel = cv2.GaussianBlur(r_channel, (5, 5), 0)
            g_channel = cv2.GaussianBlur(g_channel, (5, 5), 0)
            b_channel = cv2.GaussianBlur(b_channel, (5, 5), 0)
            
            # Combine into UV face
            uv_face = np.stack([b_channel, g_channel, r_channel], axis=-1)
            uv_face = (uv_face * 255).astype(np.uint8)
            
            # Apply mask with smooth blending
            mask_3ch = np.stack([mask] * 3, axis=-1)
            
            # Smooth mask edges
            mask_smooth = cv2.GaussianBlur(mask, (21, 21), 0)
            mask_smooth_3ch = np.stack([mask_smooth] * 3, axis=-1)
            
            # Blend with original
            result = (uv_face * mask_smooth_3ch + result * (1 - mask_smooth_3ch)).astype(np.uint8)
            
        return result
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'device': str(self.device),
            'mode': self.mode,
            'method': 'DECA-style UV mapping'
        }


class DECARenderer:
    """
    Renderer for DECA-style UV face visualization
    """
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def render_uv_face(self, 
                      original_frame: np.ndarray,
                      uv_face: np.ndarray,
                      mask: np.ndarray,
                      face_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render UV face with optional enhancements
        
        Args:
            original_frame: Original input frame
            uv_face: UV-mapped face
            mask: Face mask
            face_landmarks: Optional face landmarks for enhanced rendering
            
        Returns:
            Rendered frame with UV face
        """
        # The UV face is already blended in the processor
        # This renderer can add additional effects if needed
        result = uv_face.copy()
        
        # Optional: Add contour lines for better 3D effect
        if face_landmarks is not None and len(face_landmarks) > 0:
            # Draw subtle contour lines to enhance 3D appearance
            h, w = result.shape[:2]
            
            # Convert landmarks to image coordinates if needed
            if face_landmarks.shape[1] == 3:
                # 3D landmarks, use only x, y
                pts = face_landmarks[:, :2].astype(np.int32)
            else:
                pts = face_landmarks.astype(np.int32)
                
            # Draw face contour
            if len(pts) >= 17:
                # Jawline
                jawline = pts[:17]
                for i in range(len(jawline) - 1):
                    cv2.line(result, tuple(jawline[i]), tuple(jawline[i+1]), 
                            (100, 100, 100), 1, cv2.LINE_AA)
                            
        return result