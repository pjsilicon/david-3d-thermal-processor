#!/bin/bash

echo "🚀 Setting up DAVID-Style Holographic Overlay Processor..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Test the installation
echo "🧪 Running basic tests..."
python test_hologram.py test_face.jpg

# Check if test passed
if [ -f "test_hologram_output.jpg" ]; then
    echo "✅ Test passed! Holographic effect is working."
    echo "📁 Generated files:"
    echo "   - test_hologram_output.jpg (holographic result)"
    echo "   - test_depth.jpg (depth map)"
    echo "   - test_mask.jpg (face mask)"
else
    echo "❌ Test failed. Check the error messages above."
    exit 1
fi

echo ""
echo "🎉 Setup complete! You can now:"
echo "   1. Run the web app: python app.py"
echo "   2. Open http://localhost:5000 in your browser"
echo "   3. Upload a video and apply the holographic effect"
echo ""
echo "💡 Tips:"
echo "   - Use 'Fast' mode for real-time processing"
echo "   - Works best with clear face videos"
echo "   - M4 MacBook optimized for performance"