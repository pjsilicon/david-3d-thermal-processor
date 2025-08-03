import os
import time
from typing import Optional
import numpy as np
from tqdm import tqdm
import cv2

from .video_handler import VideoHandler
from .face_tracker import FaceBodyTracker
from .depth_estimator import DepthEstimator
from .depth_hologram_generator import DepthHologramGenerator
from .surface_normal_estimator import SurfaceNormalEstimator, SurfaceNormalRenderer
try:
    from .enhanced_surface_normal_estimator import EnhancedSurfaceNormalEstimator
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

try:
    from .deca_face_processor import DECAFaceProcessor, DECARenderer
    DECA_AVAILABLE = True
except ImportError:
    DECA_AVAILABLE = False

try:
    from .david_processor import DaviDProcessor
    DAVID_AVAILABLE = True
except ImportError:
    DAVID_AVAILABLE = False

class HolographicOverlayProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.tracker = FaceBodyTracker()
        
        # Choose processing mode
        processing_mode = config.get('processing_mode', '3d_normal')  # '3d_normal', 'hologram', or 'david'
        depth_mode = config.get('depth_mode', 'fast')  # fast/medium/quality
        use_gpu = config.get('use_gpu', True)  # Enable GPU acceleration by default
        
        if processing_mode == 'david' and DAVID_AVAILABLE:
            print("🎨 Using DaviD-style UV Face Mapping")
            # Make sure to provide the correct path to the multitask model
            self.david_processor = DaviDProcessor(multitask_model_path='DaviD/models/david/multitask-vitl16_384.onnx')
            
            # Configure DaviD effect parameters from UI
            normal_weight = config.get('david_normal_weight', 0.85)
            blend_strength = config.get('david_blend_strength', 0.9) 
            depth_contribution = config.get('david_depth_contribution', 0.15)
            self.david_processor.configure_effect(normal_weight, blend_strength, depth_contribution)
            
            self.use_surface_normals = False  # DaviD handles its own processing
            self.use_david = True
        elif processing_mode == '3d_normal':
            print("🎯 Using 3D Surface Normal Mode")
            # Prefer DECA-style UV mapping over other methods
            if DECA_AVAILABLE:
                print("🎨 Using DECA-style UV Face Mapping")
                self.surface_normal_estimator = DECAFaceProcessor(mode=depth_mode)
                self.surface_normal_renderer = DECARenderer()
            elif ENHANCED_AVAILABLE:
                print("✨ Using Enhanced Surface Normal Estimator")
                self.surface_normal_estimator = EnhancedSurfaceNormalEstimator(mode=depth_mode)
                self.surface_normal_renderer = SurfaceNormalRenderer()
            else:
                print("📊 Using Basic Surface Normal Estimator")
                self.surface_normal_estimator = SurfaceNormalEstimator(mode=depth_mode)
                self.surface_normal_renderer = SurfaceNormalRenderer()
            self.use_surface_normals = True
            self.use_david = False
        else:
            print("🌈 Using Holographic Overlay Mode")
            self.depth_estimator = DepthEstimator(mode=depth_mode, use_gpu=use_gpu)
            self.hologram_generator = DepthHologramGenerator()
            self.use_surface_normals = False
            self.use_david = False
        
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> dict:
        """Main processing pipeline"""
        start_time = time.time()
        stats = {
            'input_path': input_path,
            'output_path': output_path,
            'frames_processed': 0,
            'processing_time': 0,
            'average_fps': 0,
            'tracking_loss_frames': 0
        }
        
        # Initialize video handler
        video = VideoHandler(input_path, output_path)
        writer = video.create_writer()
        
        # Extract audio for later
        audio_path = video.extract_audio()
        
        # Process frames
        with tqdm(total=video.total_frames, desc="Creating holographic overlay") as pbar:
            for frame_idx, frame in video.read_frames():
                # Track face and body
                tracking_results = self.tracker.process_frame(frame)
                
                # Check tracking quality
                if tracking_results['tracking_confidence'] < 0.3:
                    stats['tracking_loss_frames'] += 1
                    # Use previous mask if available
                    if hasattr(self, 'last_good_mask'):
                        tracking_results['combined_mask'] = self.last_good_mask
                    else:
                        writer.write(frame)
                        continue
                
                # Save good mask
                if tracking_results['combined_mask'] is not None:
                    self.last_good_mask = tracking_results['combined_mask'].copy()
                
                # Process frame with either surface normals or holographic overlay
                if tracking_results['combined_mask'] is not None:
                    if self.use_david:
                        print(f"🎨 Using DaviD processor for frame {frame_idx}")
                        processed_frame = self.david_processor.process_frame(frame, tracking_results['combined_mask'])
                        
                        # Save sample frames for DaviD mode
                        if frame_idx == 0:
                            print("🎨 Saving DaviD sample frames...")
                            cv2.imwrite('sample_first_frame.png', processed_frame)
                            cv2.imwrite('sample_david_3d.png', processed_frame)
                            cv2.imwrite('sample_mask.png', tracking_results['combined_mask'])
                    elif self.use_surface_normals:
                        # Check if we're using DECA processor
                        if hasattr(self.surface_normal_estimator, 'process_face'):
                            # DECA-style UV face processing
                            processed_frame = self.surface_normal_estimator.process_face(
                                frame, 
                                tracking_results['combined_mask']
                            )
                            
                            # Apply DECA rendering
                            processed_frame = self.surface_normal_renderer.render_uv_face(
                                frame,
                                processed_frame,
                                tracking_results['combined_mask'],
                                tracking_results.get('face_landmarks')
                            )
                        else:
                            # Traditional surface normal estimation
                            processed_frame = self.surface_normal_estimator.estimate_surface_normals(
                                frame, 
                                tracking_results['combined_mask']
                            )
                            
                            # Apply 3D normal rendering (complete face replacement)
                            processed_frame = self.surface_normal_renderer.create_3d_normal_overlay(
                                frame,
                                processed_frame,
                                tracking_results['combined_mask'],
                                tracking_results.get('face_landmarks')
                            )
                        
                        # Save sample frames
                        if frame_idx == 0:
                            cv2.imwrite('sample_first_frame.png', processed_frame)
                            cv2.imwrite('sample_3d_normals.png', processed_frame)
                            cv2.imwrite('sample_mask.png', tracking_results['combined_mask'])
                    else:
                        # Original holographic overlay mode
                        depth_map = self.depth_estimator.estimate_frame(
                            frame, 
                            tracking_results['combined_mask'],
                            tracking_results.get('face_landmarks')
                        )
                        
                        processed_frame = self.hologram_generator.create_holographic_overlay(
                            frame,
                            depth_map,
                            tracking_results['combined_mask'],
                            tracking_results.get('face_landmarks')
                        )
                        
                        # Save sample frames
                        if frame_idx == 0:
                            cv2.imwrite('sample_first_frame.png', processed_frame)
                            cv2.imwrite('sample_depth.png', (depth_map * 255).astype(np.uint8))
                            cv2.imwrite('sample_mask.png', tracking_results['combined_mask'])
                else:
                    processed_frame = frame
                
                # Write frame
                writer.write(processed_frame)
                stats['frames_processed'] += 1
                
                # Update progress
                pbar.update(1)
                if progress_callback:
                    progress = (frame_idx + 1) / video.total_frames
                    progress_callback(progress)
        
        # Cleanup
        video.cap.release()
        writer.release()
        
        # Remux with audio
        if audio_path and os.path.exists(audio_path):
            video.remux_with_audio(output_path, audio_path)
        
        # Calculate stats
        stats['processing_time'] = time.time() - start_time
        stats['average_fps'] = stats['frames_processed'] / stats['processing_time']
        
        # Add processing performance stats
        if self.use_david:
            david_stats = self.david_processor.get_performance_stats()
            stats.update({f'david_{k}': v for k, v in david_stats.items()})
        elif self.use_surface_normals:
            normal_stats = self.surface_normal_estimator.get_performance_stats()
            stats.update({f'normal_{k}': v for k, v in normal_stats.items()})
        else:
            depth_stats = self.depth_estimator.get_performance_stats()
            stats.update({f'depth_{k}': v for k, v in depth_stats.items()})
        
        return stats