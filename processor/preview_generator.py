import cv2
import numpy as np
import os
from .video_handler import VideoHandler
# Import the same processor as main pipeline for consistency
from .main_pipeline import HolographicOverlayProcessor

class PreviewGenerator:
    """
    Generates high-quality preview images to show what the full video processing will look like.
    Uses the same processing pipeline as main video processing for consistency.
    """
    
    def __init__(self, config):
        self.config = config
        # Use the same processor as main pipeline for consistency
        self.processor = HolographicOverlayProcessor(config)
    
    def generate_preview_at_position(self, video_path, output_path, frame_position=0.5):
        """
        Generate a preview image at a specific position in the video.
        Uses the same processing pipeline as main video processing for consistency.
        
        Args:
            video_path: Path to input video
            output_path: Path where preview image will be saved
            frame_position: Position in video (0.0 to 1.0)
            
        Returns:
            bool: True if preview generated successfully
        """
        try:
            print(f"🎭 Generating preview for {video_path} at position {frame_position}")
            
            # Use the same VideoHandler as main processing for consistency
            video = VideoHandler(video_path, output_path)
            
            if video.total_frames == 0:
                print("❌ Video has no frames")
                return False
            
            print(f"📹 Video has {video.total_frames} frames")
            
            # Calculate target frame position
            target_frame = int(video.total_frames * frame_position)
            
            # Seek to target frame
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = video.cap.read()
            
            if not ret or frame is None:
                print(f"❌ Could not read frame at position {target_frame}")
                video.cap.release()
                return False
            
            print(f"📍 Processing frame {target_frame}")
            
            # Use the same tracking and processing as main pipeline
            tracking_results = self.processor.tracker.process_frame(frame)
            confidence = tracking_results.get('tracking_confidence', 0.0)
            
            print(f"📍 Face tracking confidence: {confidence:.2f}")
            
            # Process frame if we have face detection
            processed_frame = frame
            if tracking_results['combined_mask'] is not None:
                if self.processor.use_david:
                    processed_frame = self.processor.david_processor.process_frame(
                        frame, tracking_results['combined_mask']
                    )
                    print("✅ Applied DaviD thermal effect")
                elif self.processor.use_surface_normals:
                    # Process with surface normals
                    depth_map = self.processor.surface_normal_estimator.estimate_depth_and_normals(frame)
                    processed_frame = self.processor.surface_normal_renderer.render(frame, depth_map, tracking_results['combined_mask'])
                    print("✅ Applied surface normal effect")
                else:
                    # Process with holographic overlay
                    depth_map = self.processor.depth_estimator.estimate_depth(frame)
                    processed_frame = self.processor.hologram_generator.apply_holographic_overlay(
                        frame, depth_map, tracking_results['combined_mask']
                    )
                    print("✅ Applied holographic overlay")
            else:
                print("⚠️ No face detected, using original frame")
            
            video.cap.release()
            
            # Save the processed frame as preview
            success = cv2.imwrite(output_path, processed_frame)
            if success:
                print(f"✅ Preview saved to {output_path}")
                return True
            else:
                print(f"❌ Failed to save preview to {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Preview generation failed: {str(e)}")
            return False
    
    def generate_preview(self, video_path, output_path):
        """
        Generate a preview image from the middle of the video.
        Uses the same processing pipeline as main video processing for consistency.
        
        Args:
            video_path: Path to input video
            output_path: Path where preview image will be saved
            
        Returns:
            bool: True if preview generated successfully
        """
        return self.generate_preview_at_position(video_path, output_path, 0.5)