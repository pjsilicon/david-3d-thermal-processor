"""
Process video with DECA-style UV Face Mapping
"""

import os
from processor.main_pipeline import HolographicOverlayProcessor

# Input and output paths
input_video_path = 'uploads/test.mp4'
output_video_path = 'outputs/test_deca_uv_face.mp4'

# Create directories if needed
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print(f"🎯 Processing {input_video_path} with DECA-style UV Face Mapping...")
print("=" * 60)
print("✨ Key features:")
print("   - UV-mapped 3D face visualization")
print("   - Colorful representation of face geometry")  
print("   - Similar to Microsoft DAViD demo effect")
print("   - Smooth gradients showing 3D structure")

# Configuration for DECA UV face mapping
config = {
    'processing_mode': '3d_normal',  # This will use DECA processor
    'depth_mode': 'fast',
    'intensity': 0.8,
    'use_gpu': True
}

print(f"\n⚙️  Settings: {config}")

# Initialize processor
processor = HolographicOverlayProcessor(config)

# Process video
print("\n🔄 Processing video...")
stats = processor.process_video(input_video_path, output_video_path)

print("\n✅ DECA-style UV Face Processing Complete!")
print("=" * 60)
print(f"📊 Frames processed: {stats['frames_processed']}")
print(f"⏱️  Total time: {stats['processing_time']:.1f}s")
print(f"🚀 Average FPS: {stats['average_fps']:.1f}")
print(f"❗ Tracking loss frames: {stats['tracking_loss_frames']}")

if 'processing_method' in stats:
    print(f"🎨 Processing method: {stats['processing_method']}")

# Get file size
output_file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
print(f"\n🎉 Output saved to: {output_video_path}")
print(f"📁 File size: {output_file_size_mb:.1f} MB")

print(f"\n🎥 To preview your DECA-style UV face video:")
print(f"   open {output_video_path}")

print("\n🌟 What you should see:")
print("   ✅ Face replaced with smooth UV-mapped visualization")
print("   ✅ Colorful gradient effect showing 3D geometry")
print("   ✅ Red/pink = left/right facing surfaces")
print("   ✅ Green = up/down facing surfaces")
print("   ✅ Blue/purple = forward facing/depth")
print("   ✅ Clear geometric features (nose, eyes, mouth, cheeks)")
print("   ✅ Smooth transitions as the face moves")

print("\nThis creates a DAViD-style 3D face visualization! 🎯")