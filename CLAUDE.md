# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
DaviD 3D Thermal Face Processor - A Flask web application that creates 3D thermal effects on faces in videos using Microsoft's DaviD (Depth and Visual Understanding in Dense prediction) models.

## Key Commands

### Setup
```bash
# Automated setup
./setup.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download DaviD models if not present
mkdir -p DaviD/models/david
curl https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx -o DaviD/models/david/multitask-vitl16_384.onnx
```

### Running the Application
```bash
# Start web server
python3 app.py
# Access at http://127.0.0.1:5001

# Run DaviD demo standalone
cd DAViD/DAViD
python demo.py --image path/to/input.jpg --multitask-model models/multitask.onnx
```

### Testing
No formal test suite exists. Test manually by:
1. Starting the web server
2. Uploading a test video through the web interface
3. Verifying processing completes without errors

## Architecture Overview

### Core Processing Pipeline
The video processing flow follows this architecture:

1. **Web Interface** (`app.py`, `templates/index.html`)
   - Flask server handles uploads and parameter configuration
   - Real-time parameter adjustment interface
   - Progress tracking and time estimation

2. **Main Pipeline** (`processor/main_pipeline.py`)
   - Orchestrates the entire processing workflow
   - Manages temporary files and cleanup
   - Handles progress reporting

3. **Processing Components** (`processor/`)
   - `video_handler.py`: Video I/O, frame extraction/compilation
   - `face_tracker.py`: Face and upper body detection using OpenCV
   - `david_processor.py`: DaviD model integration for thermal effects
   - `depth_estimator.py`: Depth map generation
   - `surface_normal_estimator.py`: 3D surface normal calculations
   - `depth_hologram_generator.py`: Alternative holographic overlay

4. **DaviD Models** (`DAViD/DAViD/`)
   - `runtime/estimation.py`: Core multi-task estimation
   - ONNX models for depth, surface normals, and segmentation
   - GPU-accelerated inference with ONNX Runtime

### Key Design Patterns
- **Pipeline Architecture**: Each processor is a separate module that can be tested independently
- **Frame-by-Frame Processing**: Videos are processed as individual frames for memory efficiency
- **Parameter Injection**: All processing parameters flow from web interface through the pipeline
- **Temporary File Management**: Uses unique session IDs to prevent conflicts

### Processing Modes
1. **david**: Primary mode using DaviD models for blue/cyan thermal effects
2. **surface_normal**: Geometric UV mapping visualization
3. **holographic_depth**: Alternative depth-based overlay

### Important Files
- `processor/david_processor.py`: Core thermal effect implementation
- `processor/main_pipeline.py`: Pipeline orchestration and progress tracking
- `app.py`: Web server and API endpoints
- `DAViD/DAViD/runtime/estimation.py`: DaviD model inference

### Performance Considerations
- Optimized for Apple M4 GPU but works on CPU
- Memory-efficient frame processing to handle large videos
- Progress estimation based on frame count and processing speed
- Cleanup of temporary files after processing