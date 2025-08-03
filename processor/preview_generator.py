import cv2
import numpy as np
import os
from .video_handler import VideoHandler
from .face_tracker import FaceBodyTracker
from .david_processor import DaviDProcessor

class PreviewGenerator:
    """
    Generates high-quality preview images to show what the full video processing will look like.
    Perfect for checking mask accuracy and effect quality before committing to full processing.
    """
    
    def __init__(self, config):
        self.config = config
        self.processing_mode = config.get('processing_mode', 'david')
        
        # Initialize processors
        self.face_tracker = FaceBodyTracker()
        
        if self.processing_mode == 'david':
            # Initialize DaviD processor for high-quality preview
            multitask_model_path = 'DaviD/models/david/multitask-vitl16_384.onnx'
            if os.path.exists(multitask_model_path):
                self.david_processor = DaviDProcessor(multitask_model_path)
                
                # Configure with user settings
                normal_weight = config.get('david_normal_weight', 0.85)
                blend_strength = config.get('david_blend_strength', 0.9)
                depth_contribution = config.get('david_depth_contribution', 0.15)
                face_focus = config.get('david_face_focus', 0.75)
                
                self.david_processor.configure_effect(
                    normal_weight, blend_strength, depth_contribution, face_focus
                )
            else:
                raise Exception("DaviD model not found")
    
    def generate_preview(self, video_path, output_path):
        """
        Generate a preview image from the video showing the thermal effect.
        Uses multiple frames and selects the best one for preview.
        
        Args:
            video_path: Path to input video
            output_path: Path where preview image will be saved
            
        Returns:
            bool: True if preview generated successfully
        """
        try:
            print(f"🎭 Generating preview for {video_path}")
            
            # Use cv2.VideoCapture directly for preview (simpler than VideoHandler)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("❌ Invalid video file")
                return False
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print("❌ Video has no frames")
                cap.release()
                return False
            
            print(f"📹 Video has {total_frames} frames")
            
            # Select multiple frames for analysis (beginning, middle, 3/4 point)
            frame_positions = [
                int(total_frames * 0.1),   # 10% into video (skip intro)
                int(total_frames * 0.4),   # 40% into video
                int(total_frames * 0.6),   # 60% into video
                int(total_frames * 0.8),   # 80% into video
            ]
            
            best_frame = None
            best_confidence = 0.0
            best_processed = None
            
            print(f"🔍 Analyzing frames at positions: {frame_positions}")
            
            for frame_pos in frame_positions:
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                    
                # Get face tracking results
                tracking_results = self.face_tracker.process_frame(frame)
                confidence = tracking_results.get('tracking_confidence', 0.0)
                
                print(f"📍 Frame {frame_pos}: confidence {confidence:.2f}")
                
                # Process frame if we have good face detection
                if confidence > 0.5 and confidence > best_confidence:
                    if self.processing_mode == 'david':
                        processed_frame = self.david_processor.process_frame(
                            frame, tracking_results.get('combined_mask')
                        )
                        
                        # Save both original and processed for comparison
                        best_frame = frame
                        best_processed = processed_frame
                        best_confidence = confidence
                        
                        print(f"✨ New best frame found at position {frame_pos} (confidence: {confidence:.2f})")
            
            cap.release()
            
            if best_processed is not None:
                # Create side-by-side comparison for preview
                comparison = self._create_comparison_image(best_frame, best_processed)
                
                # Save preview
                cv2.imwrite(output_path, comparison)
                print(f"💾 Preview saved to {output_path}")
                
                return True
            else:
                print("❌ No suitable frame found for preview")
                return False
                
        except Exception as e:
            print(f"❌ Preview generation failed: {e}")
            return False
    
    def _create_comparison_image(self, original, processed):
        """
        Create a side-by-side comparison image showing before/after effect.
        
        Args:
            original: Original frame
            processed: Processed frame with thermal effect
            
        Returns:
            numpy.ndarray: Comparison image
        """
        h, w = original.shape[:2]
        
        # Resize if images are too large for preview
        max_width = 800
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            original = cv2.resize(original, (new_w, new_h))
            processed = cv2.resize(processed, (new_w, new_h))
            h, w = new_h, new_w
        
        # Create side-by-side comparison
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        
        # Add original frame on left
        comparison[:, :w] = original
        
        # Add separator line
        comparison[:, w:w+20] = (40, 40, 40)  # Dark gray separator
        
        # Add processed frame on right
        comparison[:, w+20:] = processed
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # "Original" label
        cv2.putText(comparison, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
        
        # "DaviD Thermal Effect" label
        cv2.putText(comparison, "DaviD Thermal 3D", (w + 30, 30), font, font_scale, (0, 168, 255), thickness)
        
        # Add confidence indicator
        confidence_text = f"Face Detection: {self.face_tracker.tracking_confidence:.1%}"
        cv2.putText(comparison, confidence_text, (10, h - 15), font, 0.5, (200, 200, 200), 1)
        
        return comparison