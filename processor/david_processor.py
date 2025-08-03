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
        self.face_focus = 0.75         # How tightly to focus on face vs upper body (0=full body, 1=face only)
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        print("✅ DaviD processor ready!")
    
    def configure_effect(self, normal_weight=None, blend_strength=None, depth_contribution=None, face_focus=None):
        """
        Configure the 3D effect parameters.
        
        Args:
            normal_weight: 0.0-1.0, how much surface normal effect (blue/cyan thermal look)
            blend_strength: 0.0-1.0, overall effect intensity
            depth_contribution: 0.0-1.0, how much depth information to include
            face_focus: 0.0-1.0, how tightly to focus on face vs upper body
        """
        if normal_weight is not None:
            self.normal_weight = max(0.0, min(1.0, normal_weight))
        if blend_strength is not None:
            self.blend_strength = max(0.0, min(1.0, blend_strength))
        if depth_contribution is not None:
            self.depth_contribution = max(0.0, min(1.0, depth_contribution))
        if face_focus is not None:
            self.face_focus = max(0.0, min(1.0, face_focus))
            
        print(f"🎨 DaviD effect configured: normal={self.normal_weight:.2f}, intensity={self.blend_strength:.2f}, depth={self.depth_contribution:.2f}, focus={self.face_focus:.2f}")

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
                
                # Trust DaviD's high-quality foreground segmentation completely
                if foreground_mask is not None:
                    # Use DaviD's mask directly - it's trained on humans and works great!
                    face_mask = foreground_mask.copy()
                    
                    # Optional: Apply face_focus as smooth boundary adjustment
                    if self.face_focus < 0.95:  # Only modify if user wants boundary adjustment
                        # Keep mask in float32 for quality preservation
                        if face_mask.dtype == np.uint8:
                            mask_float = face_mask.astype(np.float32) / 255.0
                        else:
                            mask_float = face_mask.astype(np.float32)
                        
                        # Use high-quality Gaussian-based boundary adjustment instead of morphological ops
                        if self.face_focus > 0.5:
                            # Tighter boundaries: apply slight inward gradient
                            blur_radius = int(2 + (self.face_focus - 0.5) * 6)  # 2-5 pixels
                            mask_blur = cv2.GaussianBlur(mask_float, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
                            # Create inward gradient by mixing original with blurred
                            mask_float = mask_float * 0.7 + mask_blur * 0.3
                            mask_float = np.power(mask_float, 1.2)  # Slight contrast boost for tighter edges
                        else:
                            # More generous coverage: expand boundaries smoothly
                            blur_radius = int(2 + (0.5 - self.face_focus) * 6)  # 2-5 pixels
                            mask_blur = cv2.GaussianBlur(mask_float, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
                            # Expand by mixing with blurred version
                            mask_float = np.maximum(mask_float, mask_blur * 0.8)
                        
                        # Keep as float32 - no conversion back to uint8 yet
                        face_mask = np.clip(mask_float, 0, 1)
                        print(f"✨ Using DaviD segmentation with smooth boundary adjustment (focus={self.face_focus:.2f})")
                    else:
                        # Convert to float32 for consistency
                        if face_mask.dtype == np.uint8:
                            face_mask = face_mask.astype(np.float32) / 255.0
                        print(f"✨ Using DaviD's high-quality foreground segmentation directly")
                        
                elif mask is not None:
                    # Fallback to face tracking mask if DaviD segmentation unavailable
                    if mask.dtype == np.uint8:
                        face_mask = mask.astype(np.float32) / 255.0
                    else:
                        face_mask = mask.astype(np.float32)
                    print(f"⚠️ DaviD foreground unavailable, using face tracking mask as fallback")
                else:
                    # Last resort - apply to entire frame
                    face_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32)
                    print(f"❌ No mask available - applying to entire frame")
                
                # Ensure mask is in 0-1 float32 range (should already be from above)
                if face_mask.dtype == np.uint8:
                    mask_norm = face_mask.astype(np.float32) / 255.0
                else:
                    mask_norm = np.clip(face_mask.astype(np.float32), 0, 1)
                
                # Create 3-channel mask
                mask_3ch = np.stack([mask_norm] * 3, axis=-1)
                
                # Apply the thermal effect with high-precision blending
                blend_strength = self.blend_strength  # Use configured effect intensity
                
                # Keep everything in float32 for quality
                frame_float = frame.astype(np.float32) / 255.0
                combined_vis_float = combined_vis.astype(np.float32) / 255.0
                
                # Professional gamma-correct blending
                gamma = 2.2
                frame_linear = np.power(frame_float, gamma)
                overlay_linear = np.power(combined_vis_float, gamma)
                
                # High-quality alpha blending in linear space
                blend_mask = mask_3ch * blend_strength
                blended_linear = frame_linear * (1 - blend_mask) + overlay_linear * blend_mask
                
                # Convert back to sRGB and clamp
                result_float = np.power(np.clip(blended_linear, 0, 1), 1/gamma)
                result = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
                    
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
