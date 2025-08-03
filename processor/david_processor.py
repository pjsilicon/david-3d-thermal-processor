import sys
import os
import cv2
import numpy as np

# Add the DaviD runtime to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../DaviD/DaviD/runtime'))

from multi_task_estimator import MultiTaskEstimator
from visualize import visualize_relative_depth_map, visualize_normal_maps, visualize_foreground

class DaviDProcessor:
    """
    DaviD processor that creates stunning 3D depth overlays on faces using Microsoft's DaviD models.
    This provides the high-quality depth and surface normal estimation you're looking for.
    """
    
    def __init__(self, multitask_model_path):
        print("🎨 Initializing DaviD processor with multitask model...")
        self.multitask_estimator = MultiTaskEstimator(
            onnx_model=multitask_model_path, is_inverse_depth=False
        )
        
        # Effect parameters - configurable through UI
        self.normal_weight = 0.85      # How much surface normal (blue/cyan thermal) effect
        self.blend_strength = 0.9      # Overall effect intensity
        self.depth_contribution = 0.15 # How much depth info to blend in
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        print("✅ DaviD processor ready!")
    
    def configure_effect(self, normal_weight=None, blend_strength=None, depth_contribution=None):
        """
        Configure the 3D effect parameters.
        
        Args:
            normal_weight: 0.0-1.0, how much surface normal effect (blue/cyan thermal look)
            blend_strength: 0.0-1.0, overall effect intensity
            depth_contribution: 0.0-1.0, how much depth information to include
        """
        if normal_weight is not None:
            self.normal_weight = max(0.0, min(1.0, normal_weight))
        if blend_strength is not None:
            self.blend_strength = max(0.0, min(1.0, blend_strength))
        if depth_contribution is not None:
            self.depth_contribution = max(0.0, min(1.0, depth_contribution))
            
        print(f"🎨 DaviD effect configured: normal={self.normal_weight:.2f}, intensity={self.blend_strength:.2f}, depth={self.depth_contribution:.2f}")

    def process_frame(self, frame, mask=None):
        """
        Process a frame with DaviD models to create 3D depth overlay.
        
        Args:
            frame: Input frame (BGR image)
            mask: Optional mask to focus processing on specific regions
            
        Returns:
            Processed frame with stunning 3D depth overlay
        """
        import time
        start_time = time.time()
        
        try:
            # Get DaviD model predictions
            results = self.multitask_estimator.estimate_all_tasks(frame)
            
            # Extract the different outputs
            depth_map = results.get('depth', None)
            surface_normals = results.get('normal', None) 
            foreground_mask = results.get('foreground', None)
            
            # Create the final 3D depth visualization
            if depth_map is not None and surface_normals is not None:
                # First, create DaviD visualizations using their own foreground mask
                # This ensures proper depth/normal visualization quality
                depth_vis = visualize_relative_depth_map(frame, depth_map, foreground_mask)
                normal_vis = visualize_normal_maps(frame, surface_normals, foreground_mask)
                
                # Then we'll apply our precise face tracking mask for focused targeting
                
                # For the stunning blue/cyan thermal effect, prioritize surface normals
                # This creates the dramatic thermal-like 3D depth look you want
                normal_weight = self.normal_weight  # Use configured thermal effect weight
                depth_weight = self.depth_contribution
                combined_vis = cv2.addWeighted(normal_vis, normal_weight, depth_vis, depth_weight, 0)
                
                # Apply the 3D effect using our precise face tracking mask for focused targeting
                if mask is not None:
                    # Use the precise face tracking mask - this ensures effect only on face/upper body
                    face_mask = mask
                    print(f"🎯 Using face tracking mask for precise targeting")
                elif foreground_mask is not None:
                    # Fallback to DaviD's foreground mask, but make it more conservative
                    face_mask = foreground_mask
                    print(f"⚠️ Using DaviD foreground mask as fallback")
                else:
                    # No mask available - apply to entire frame (not ideal)
                    face_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                    print(f"❌ No mask available - applying to entire frame")
                
                # Normalize mask to 0-1 range
                if face_mask.dtype == np.uint8:
                    mask_norm = face_mask.astype(np.float32) / 255.0
                else:
                    mask_norm = face_mask.astype(np.float32)
                
                # Create 3-channel mask
                mask_3ch = np.stack([mask_norm] * 3, axis=-1)
                
                # Apply the thermal effect only where the face tracking mask indicates
                blend_strength = self.blend_strength  # Use configured effect intensity
                result = (combined_vis.astype(np.float32) * mask_3ch * blend_strength + 
                         frame.astype(np.float32) * (1 - mask_3ch * blend_strength)).astype(np.uint8)
                    
                # Track performance
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.frame_count += 1
                
                return result
                
            elif depth_map is not None:
                # Fallback to depth-only visualization
                final_mask = foreground_mask if foreground_mask is not None else mask
                result = visualize_relative_depth_map(frame, depth_map, final_mask)
                
                # Track performance
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.frame_count += 1
                
                return result
                
            else:
                print("⚠️ DaviD model failed to produce depth estimates")
                return frame
                
        except Exception as e:
            print(f"❌ DaviD processing failed: {e}")
            return frame

    def get_performance_stats(self):
        """Return performance statistics"""
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            avg_fps = 0
            
        return {
            'model_type': 'DaviD_MultiTask',
            'features': ['depth', 'surface_normals', 'foreground_segmentation'],
            'average_processing_time': avg_time,
            'average_fps': avg_fps,
            'frames_processed': self.frame_count
        }
