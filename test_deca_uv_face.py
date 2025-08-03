"""
Test DECA-style UV Face Mapping
"""

import os
import cv2
import numpy as np
from processor.deca_face_processor import DECAFaceProcessor, DECARenderer
from processor.face_tracker import FaceBodyTracker

def test_deca_uv_face():
    """Test DECA-style UV face mapping on a single image"""
    
    print("🎯 Testing DECA-style UV Face Mapping")
    print("=====================================")
    
    # Load test image
    test_image_path = 'uploads/test.mp4'
    if os.path.exists(test_image_path):
        # Extract first frame from video
        cap = cv2.VideoCapture(test_image_path)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("❌ Failed to read video frame")
            return
    else:
        # Create a test image
        print("⚠️  No test video found, creating synthetic test image")
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Draw a face-like shape
        cv2.ellipse(image, (320, 240), (100, 130), 0, 0, 360, (200, 180, 160), -1)
        # Add features
        cv2.circle(image, (290, 220), 15, (50, 50, 50), -1)  # Left eye
        cv2.circle(image, (350, 220), 15, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(image, (320, 280), (30, 20), 0, 0, 180, (100, 50, 50), -1)  # Mouth
    
    # Initialize components
    print("\n📦 Initializing components...")
    tracker = FaceBodyTracker()
    processor = DECAFaceProcessor(mode='fast')
    renderer = DECARenderer()
    
    # Process image
    print("\n🔍 Detecting face...")
    tracking_results = tracker.process_frame(image)
    
    if tracking_results['combined_mask'] is not None:
        print("✅ Face detected!")
        
        # Process face with DECA-style UV mapping
        print("\n🎨 Applying UV face mapping...")
        uv_face = processor.process_face(image, tracking_results['combined_mask'])
        
        # Render final result
        result = renderer.render_uv_face(
            image,
            uv_face,
            tracking_results['combined_mask'],
            tracking_results.get('face_landmarks')
        )
        
        # Save results
        cv2.imwrite('test_original.jpg', image)
        cv2.imwrite('test_uv_face.jpg', result)
        cv2.imwrite('test_mask.jpg', tracking_results['combined_mask'])
        
        # Create comparison image
        comparison = np.hstack([image, result])
        cv2.imwrite('test_deca_comparison.jpg', comparison)
        
        print("\n✅ Results saved:")
        print("   - test_original.jpg (input image)")
        print("   - test_uv_face.jpg (UV-mapped face)")
        print("   - test_mask.jpg (face mask)")
        print("   - test_deca_comparison.jpg (side-by-side comparison)")
        
        # Get performance stats
        stats = processor.get_performance_stats()
        print(f"\n📊 Performance stats:")
        print(f"   - Device: {stats['device']}")
        print(f"   - Method: {stats['method']}")
        print(f"   - Mode: {stats['mode']}")
        
        print("\n🎯 DECA-style UV face mapping test complete!")
        print("\n💡 What you should see:")
        print("   ✅ Face replaced with colorful UV mapping")
        print("   ✅ Red/pink tones = left/right facing surfaces")
        print("   ✅ Green tones = up/down facing surfaces")
        print("   ✅ Blue/purple tones = forward facing/depth")
        print("   ✅ Smooth gradients representing 3D geometry")
        print("   ✅ Clear geometric features (nose, eyes, mouth)")
        
    else:
        print("❌ No face detected in image")

if __name__ == "__main__":
    test_deca_uv_face()