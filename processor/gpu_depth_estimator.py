import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Union
import warnings
from transformers import pipeline, AutoImageProcessor, AutoModel
from PIL import Image
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class GPUDepthEstimator:
    """High-quality depth estimation using AI models with M4 GPU acceleration"""
    
    def __init__(self, mode='fast'):
        """
        Initialize GPU-accelerated depth estimator
        Modes: 
        - 'fast': MiDaS small model (fastest, good quality)
        - 'medium': MiDaS v3 DPT hybrid (balanced)
        - 'quality': DepthAnything large (highest quality)
        """
        self.mode = mode
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 GPU Depth Estimator using: {self.device}")
        
        # Set dtype based on device - MPS requires float16 for compatibility
        self.torch_dtype = torch.float16 if self.device.type == "mps" else torch.float32
        print(f"📊 Using dtype: {self.torch_dtype}")
        
        # Model configurations - all use Intel/dpt-large for M4 compatibility
        self.model_configs = {
            'fast': {
                'model_name': 'Intel/dpt-large',
                'description': 'DPT Large - M4 GPU optimized'
            },
            'medium': {
                'model_name': 'Intel/dpt-large',
                'description': 'DPT Large - M4 GPU optimized'
            },
            'quality': {
                'model_name': 'Intel/dpt-large',
                'description': 'DPT Large - M4 GPU optimized'
            }
        }
        
        self._load_model()
        
        # Temporal smoothing
        self.temporal_buffer = []
        self.max_buffer_size = 2  # Reduced for faster processing
        
        # Performance tracking
        self.inference_times = []
        
    def _load_model(self):
        """Load the depth estimation model onto M4 GPU"""
        config = self.model_configs[self.mode]
        print(f"📦 Loading {config['description']}...")
        
        try:
            # Use transformers pipeline with proper MPS compatibility
            self.depth_pipeline = pipeline(
                task="depth-estimation",
                model=config['model_name'],
                device=0 if self.device.type == "mps" else -1,  # Use GPU if available
                torch_dtype=self.torch_dtype  # Use the class dtype setting
            )
            
            # Ensure model and all parameters are on correct dtype
            if self.device.type == "mps":
                self.depth_pipeline.model = self.depth_pipeline.model.to(self.device, dtype=self.torch_dtype)
            
            print(f"✅ Model loaded successfully on {self.device} with dtype {self.torch_dtype}")
            
        except Exception as e:
            print(f"⚠️  Failed to load {config['model_name']}: {e}")
            print("🔄 Falling back to CPU-optimized MiDaS...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a simpler model if the main one fails"""
        try:
            # Try multiple fallback models
            fallback_models = [
                "Intel/dpt-large",
                "vinvino02/glpn-nyu", 
                "sayakpaul/glpn-nyu-finetuned-diode-221214-123047"
            ]
            
            for model_name in fallback_models:
                try:
                    self.depth_pipeline = pipeline(
                        task="depth-estimation",
                        model=model_name,
                        device=-1,  # CPU fallback
                        torch_dtype=torch.float32
                    )
                    print(f"✅ Fallback model '{model_name}' loaded on CPU")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load fallback model '{model_name}': {e}")
                    continue
            
            raise RuntimeError("Could not load any fallback depth estimation model")
            
        except Exception as e:
            print(f"❌ Complete model loading failure: {e}")
            raise RuntimeError("Could not load any depth estimation model")
    
    def estimate_frame(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate depth map for a frame using AI model
        
        Args:
            frame: Input RGB frame (H, W, 3)
            mask: Optional mask to focus on specific regions
            
        Returns:
            depth_map: Normalized depth map (H, W) where closer = higher values
        """
        start_time = time.time()
        
        # Convert frame to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run depth estimation
        with torch.no_grad():
            try:
                # Process inputs with proper dtype handling for MPS
                if self.device.type == "mps":
                    # Manual preprocessing for better control
                    inputs = self.depth_pipeline.image_processor(pil_image, return_tensors="pt")
                    # Convert inputs to correct dtype and device
                    inputs = {k: v.to(self.device, dtype=self.torch_dtype) if torch.is_tensor(v) else v 
                             for k, v in inputs.items()}
                    
                    # Run model directly
                    outputs = self.depth_pipeline.model(**inputs)
                    depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
                else:
                    # Use standard pipeline for CPU
                    result = self.depth_pipeline(pil_image)
                    depth_map = np.array(result['predicted_depth'])
                
                # Normalize depth map
                depth_map = self._process_depth_output(depth_map, frame.shape[:2])
                
                # Apply mask if provided
                if mask is not None:
                    mask_norm = (mask > 128).astype(np.float32)
                    depth_map = depth_map * mask_norm
                
                # Temporal smoothing
                depth_map = self._apply_temporal_smoothing(depth_map)
                
                # Track performance
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                if len(self.inference_times) % 10 == 0:
                    avg_time = np.mean(self.inference_times[-10:])
                    fps = 1.0 / avg_time
                    print(f"⚡ GPU Depth: {fps:.1f} FPS ({avg_time*1000:.1f}ms per frame)")
                
                return depth_map
                
            except Exception as e:
                print(f"⚠️ GPU inference failed: {e}")
                return self._fallback_geometric_depth(frame, mask)
    
    def _process_depth_output(self, raw_depth: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Process raw model output into usable depth map"""
        # Resize to match input frame
        depth_resized = cv2.resize(raw_depth, (target_shape[1], target_shape[0]))
        
        # Normalize to 0-1 range (closer = higher values)
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()
        
        if depth_max > depth_min:
            # Invert so closer objects have higher values
            depth_norm = 1.0 - (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.ones_like(depth_resized) * 0.5
        
        # Apply gamma correction for better contrast
        depth_norm = np.power(depth_norm, 0.8)
        
        # Smooth the depth map
        depth_norm = cv2.GaussianBlur(depth_norm, (5, 5), 0)
        
        return depth_norm.astype(np.float32)
    
    def _apply_temporal_smoothing(self, depth_map: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce flicker"""
        self.temporal_buffer.append(depth_map.copy())
        
        # Keep buffer size manageable
        if len(self.temporal_buffer) > self.max_buffer_size:
            self.temporal_buffer.pop(0)
        
        if len(self.temporal_buffer) > 1:
            # Weighted average favoring recent frames
            weights = np.linspace(0.3, 1.0, len(self.temporal_buffer))
            weights = weights / weights.sum()
            
            smoothed = np.zeros_like(depth_map)
            for i, weight in enumerate(weights):
                smoothed += self.temporal_buffer[i] * weight
            
            return smoothed
        
        return depth_map
    
    def _fallback_geometric_depth(self, frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Fallback to geometric depth if AI model fails"""
        print("🔄 Using geometric fallback depth estimation")
        h, w = frame.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        if mask is not None:
            rows, cols = np.where(mask > 128)
            if len(rows) > 0:
                # Face center
                center_y = (rows.min() + rows.max()) / 2
                center_x = (cols.min() + cols.max()) / 2
                
                # Create depth based on distance from center
                y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                face_radius = max(rows.max() - rows.min(), cols.max() - cols.min()) / 2
                
                if face_radius > 0:
                    norm_dist = np.clip(dist_from_center / face_radius, 0, 1)
                    depth_map = 1.0 - (norm_dist ** 0.7) * 0.7
                    depth_map = cv2.GaussianBlur(depth_map, (31, 31), 0)
                    depth_map = depth_map * (mask > 128).astype(np.float32)
        
        return depth_map
    
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

class FaceDepthProcessor:
    """Specialized face depth processing with M4 optimization"""
    
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def enhance_face_depth(self, depth_map: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Enhance depth map specifically for faces using landmarks"""
        if face_landmarks is None:
            return depth_map
            
        enhanced = depth_map.copy()
        h, w = depth_map.shape
        
        # Convert landmarks to pixel coordinates if needed
        if face_landmarks.max() <= 1.0:
            face_landmarks = face_landmarks * np.array([w, h])
        
        # Define face depth profile based on landmarks
        landmark_depths = {
            'nose_tip': 1.0,      # Closest point
            'nose_bridge': 0.95,   
            'eyes': 0.85,         
            'cheeks': 0.75,       
            'forehead': 0.7,      
            'chin': 0.8,          
            'jaw': 0.65           # Furthest points
        }
        
        # Apply depth enhancement around key landmarks
        if len(face_landmarks) >= 68:  # Standard 68-point face landmarks
            # Nose tip (landmark 30)
            nose_tip = face_landmarks[30].astype(int)
            self._apply_depth_point(enhanced, nose_tip, landmark_depths['nose_tip'], radius=15)
            
            # Eyes (landmarks 36-47)
            for eye_idx in range(36, 48):
                if eye_idx < len(face_landmarks):
                    eye_point = face_landmarks[eye_idx].astype(int)
                    self._apply_depth_point(enhanced, eye_point, landmark_depths['eyes'], radius=10)
        
        return enhanced
    
    def _apply_depth_point(self, depth_map: np.ndarray, point: np.ndarray, depth_value: float, radius: int):
        """Apply depth value around a specific point"""
        h, w = depth_map.shape
        y, x = point[1], point[0]
        
        # Ensure point is within bounds
        if 0 <= x < w and 0 <= y < h:
            # Create circular mask
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            distance = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            mask = distance <= radius
            
            # Smooth falloff
            falloff = np.exp(-distance / (radius * 0.5))
            
            # Blend with existing depth
            depth_map[mask] = depth_map[mask] * 0.7 + depth_value * falloff[mask] * 0.3