# DaviD 3D Thermal Face Processor

A sophisticated Flask web application that applies stunning 3D thermal effects to faces in videos using Microsoft's DaviD (Depth and Visual Understanding in Dense prediction) models.

![DaviD Thermal Effect](https://img.shields.io/badge/Effect-3D%20Thermal%20Mapping-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey)
![AI](https://img.shields.io/badge/AI-Microsoft%20DaviD-orange)

## ✨ Features

### 🎨 **DaviD Thermal 3D Effect**
- **Blue/Cyan Thermal Visualization**: Stunning surface normal mapping that creates a thermal camera-like effect
- **Precise Face Targeting**: Advanced face tracking ensures effects only apply to face and upper body
- **Real-time Parameter Control**: Adjust thermal intensity, depth contribution, and targeting precision
- **High-Quality Processing**: Uses Microsoft's state-of-the-art DaviD models for depth and surface normal estimation

### 🔧 **Configurable Parameters**
- **Thermal Effect (85%)**: Controls the intensity of the blue/cyan thermal effect
- **Effect Intensity (90%)**: Overall strength of the 3D overlay
- **Depth Contribution (15%)**: Amount of geometric depth information to blend
- **Face Focus Precision (75%)**: How tightly the effect targets face and upper body

### 🚀 **Processing Modes**
1. **DaviD Thermal 3D**: Primary mode using Microsoft DaviD models
2. **3D Surface Normal**: Geometric UV mapping visualization  
3. **Holographic Depth**: Alternative depth-based overlay

### ⏱️ **Smart Processing**
- **Time Estimates**: Real-time processing time predictions based on file size
- **Progress Tracking**: Live progress updates during processing
- **Performance Stats**: Detailed processing statistics and FPS metrics
- **GPU Acceleration**: Optimized for Apple M4 and other GPU hardware

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenCV
- ONNX Runtime
- Flask and dependencies

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/david-3d-thermal-processor.git
cd david-3d-thermal-processor
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download DaviD Models**
```bash
mkdir -p DaviD/models/david
curl https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx -o DaviD/models/david/multitask-vitl16_384.onnx
```

5. **Run the application**
```bash
python3 app.py
```

6. **Open in browser**
Navigate to `http://127.0.0.1:5001`

## 🎯 Usage

### Web Interface
1. **Upload Video**: Drag & drop or click to select video files
2. **Choose Processing Mode**: Select "DaviD Thermal 3D" for the thermal effect
3. **Adjust Parameters**: 
   - Increase "Thermal Effect" for more blue/cyan intensity
   - Adjust "Effect Intensity" for stronger overlay
   - Fine-tune "Face Focus" for precise targeting
4. **Process**: Click "Process Video" and monitor progress
5. **Download**: Get your processed video with thermal effects

### API Usage
```bash
curl -X POST -F "video=@input.mp4" \
     -F "processing_mode=david" \
     -F "david_normal_weight=0.85" \
     -F "david_blend_strength=0.9" \
     -F "david_depth_contribution=0.15" \
     -F "david_face_focus=0.75" \
     http://127.0.0.1:5001/upload
```

## 🏗️ Architecture

### Core Components
- **`app.py`**: Flask web server and API endpoints
- **`processor/david_processor.py`**: DaviD model integration and thermal effect generation
- **`processor/face_tracker.py`**: Precise face and upper body detection
- **`processor/main_pipeline.py`**: Video processing orchestration
- **`templates/index.html`**: Web interface with real-time controls

### Processing Pipeline
1. **Video Input**: Upload and frame extraction
2. **Face Tracking**: Detect and track face/upper body regions
3. **DaviD Analysis**: Depth, surface normal, and foreground estimation
4. **Thermal Visualization**: Generate blue/cyan thermal effects
5. **Precise Masking**: Apply effects only to targeted face/body areas
6. **Video Output**: Render final video with thermal effects

## 🎨 Technical Details

### DaviD Integration
- **Multi-task Model**: Simultaneous depth, surface normal, and segmentation
- **ONNX Runtime**: Optimized inference on CPU and GPU
- **Visualization Pipeline**: Custom thermal effect rendering using DaviD outputs

### Face Tracking
- **OpenCV Detection**: Robust face detection across video frames
- **Conservative Masking**: Precise elliptical masks for face and upper body
- **Tracking Continuity**: Maintains effect consistency across frames

### Performance Optimization
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Memory Management**: Efficient frame processing and cleanup
- **Progress Monitoring**: Real-time processing statistics

## 📊 Performance

- **Processing Speed**: ~0.27 FPS on typical hardware
- **Time Estimate**: ~0.5 minutes per MB of video
- **Memory Usage**: Optimized for standard desktop/laptop systems
- **Quality**: High-resolution thermal effects with precise targeting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft DaviD Team**: For the incredible depth and surface normal estimation models
- **OpenCV Community**: For robust computer vision tools
- **Flask Team**: For the excellent web framework

## 🔗 Links

- [Microsoft DaviD Paper](https://arxiv.org/abs/2309.04508)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Built with ❤️ for creating stunning 3D thermal face effects**