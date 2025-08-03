import os
import time
import json
from typing import Optional
import numpy as np
from tqdm import tqdm
import cv2

from .video_handler import VideoHandler
from .face_tracker import FaceBodyTracker
from .depth_estimator import DepthEstimator
from .depth_hologram_generator import DepthHologramGenerator
from .surface_normal_estimator import SurfaceNormalEstimator, SurfaceNormalRenderer
# Import all processors - GPU acceleration is required
from .enhanced_surface_normal_estimator import EnhancedSurfaceNormalEstimator
from .deca_face_processor import DECAFaceProcessor, DECARenderer

# DaviD processor is required for GPU acceleration
from .david_processor import DaviDProcessor

class HolographicOverlayProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.tracker = FaceBodyTracker()
        
        # Choose processing mode
        processing_mode = config.get('processing_mode', '3d_normal')  # '3d_normal', 'hologram', or 'david'
        depth_mode = config.get('depth_mode', 'fast')  # fast/medium/quality
        use_gpu = config.get('use_gpu', True)  # Enable GPU acceleration by default
        
        # All modes use GPU acceleration - no CPU fallbacks
        if processing_mode == 'david':
            print("🎨 Using DaviD 3D Thermal (GPU Accelerated)")
            self.david_processor = DaviDProcessor(multitask_model_path='DaviD/models/david/multitask-vitl16_384.onnx')
            
            # Configure DaviD effect parameters from UI
            normal_weight = config.get('david_normal_weight', 0.85)
            blend_strength = config.get('david_blend_strength', 0.9) 
            depth_contribution = config.get('david_depth_contribution', 0.15)
            face_focus = config.get('david_face_focus', 0.75)
            self.david_processor.configure_effect(normal_weight, blend_strength, depth_contribution, face_focus)
            
            self.use_surface_normals = False
            self.use_david = True
            
        elif processing_mode == '3d_normal':
            print("🎯 Using 3D Surface Normal (GPU Accelerated)")
            # Use DECA for high-quality GPU-accelerated face processing
            self.surface_normal_estimator = DECAFaceProcessor(mode=depth_mode)
            self.surface_normal_renderer = DECARenderer()
            self.use_surface_normals = True
            self.use_david = False
            
        else:  # holographic mode
            print("🌈 Using Holographic Depth Overlay (GPU Accelerated)")
            # Force GPU usage - no CPU fallback
            self.depth_estimator = DepthEstimator(mode=depth_mode, use_gpu=True)
            self.hologram_generator = DepthHologramGenerator()
            self.use_surface_normals = False
            self.use_david = False
        
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None,
        progress_file: Optional[str] = None
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
        
        # Initialize video handler with quality settings
        video_quality = self.config.get('video_quality', 'high')  # draft/high/cinema
        video = VideoHandler(input_path, output_path)
        writer = video.create_writer(quality=video_quality)
        
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
                current_progress = (frame_idx + 1) / video.total_frames
                
                # Calculate progress data
                progress_data = {
                    'progress': current_progress,
                    'current_frame': frame_idx + 1,
                    'total_frames': video.total_frames,
                    'fps': stats['frames_processed'] / (time.time() - start_time) if stats['frames_processed'] > 0 else 0,
                    'eta_seconds': int((video.total_frames - frame_idx - 1) / max(0.1, stats['frames_processed'] / (time.time() - start_time))) if stats['frames_processed'] > 0 else 0
                }
                
                # Call progress callback with full data (every 5 frames to avoid overwhelming)
                if progress_callback and frame_idx % 5 == 0:
                    progress_callback(progress_data)
                
                # Legacy progress file support (will be removed later)
                if progress_file and frame_idx % 10 == 0:
                    try:
                        with open(progress_file, 'w') as f:
                            f.write(json.dumps(progress_data))
                    except Exception as e:
                        print(f"Failed to write progress: {e}")
        
        # Cleanup
        video.cap.release()
        writer.release()
        
        # Validate output video was created successfully
        is_valid, validation_message = video.validate_output()
        if not is_valid:
            raise Exception(f"Video validation failed: {validation_message}")
        
        print(f"✅ {validation_message}")
        
        # Remux with audio
        if audio_path and os.path.exists(audio_path):
            print("🎵 Adding audio track...")
            video.remux_with_audio(output_path, audio_path)
            
            # Validate again after audio remux
            is_valid, validation_message = video.validate_output()
            if not is_valid:
                raise Exception(f"Video validation failed after audio remux: {validation_message}")
        
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