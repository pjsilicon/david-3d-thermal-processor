from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import uuid
import threading
import logging
import traceback
from processor.main_pipeline import HolographicOverlayProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Configure logging for better error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Global progress tracker for polling-based updates
progress_tracker = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    logger.info("Received video upload request")
    
    if 'video' not in request.files:
        logger.warning("No video file in upload request")
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file and allowed_file(file.filename):
        logger.info(f"Processing upload for file: {file.filename}")
        video_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        file.save(input_path)
        
        # Get processing options
        options = {
            'processing_mode': request.form.get('processing_mode', 'david'),  # Default to DaviD mode
            'depth_mode': request.form.get('depth_mode', 'fast'),
            'intensity': float(request.form.get('intensity', 0.6)),
            'video_quality': request.form.get('video_quality', 'high'),  # draft/high/cinema
            # DaviD-specific parameters
            'david_normal_weight': float(request.form.get('david_normal_weight', 0.85)),
            'david_blend_strength': float(request.form.get('david_blend_strength', 0.9)),
            'david_depth_contribution': float(request.form.get('david_depth_contribution', 0.15)),
            'david_face_focus': float(request.form.get('david_face_focus', 0.75))
        }
        
        # Process video in a thread to avoid blocking
        # Use timestamp to ensure unique filenames
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_id}_{timestamp}_hologram.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Get file size for initial estimate (typical: 0.3 FPS processing)
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        estimated_seconds = file_size_mb * 30  # ~30 seconds per MB at 0.3 FPS
        
        if estimated_seconds > 60:
            estimated_time = f"~{int(estimated_seconds / 60)} minutes"
        else:
            estimated_time = f"~{int(estimated_seconds)} seconds"
        
        # Initialize progress tracking with estimate
        progress_tracker[video_id] = {
            'status': 'processing',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'fps': 0,
            'eta_seconds': estimated_seconds,
            'eta_formatted': estimated_time,
            'message': f'Starting processing... Estimated time: {estimated_time}'
        }
        
        # Start processing in background using threading
        thread = threading.Thread(
            target=process_video_async, 
            args=(video_id, input_path, output_path, options, timestamp),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'video_id': video_id,
            'message': 'Processing started. Listen for WebSocket updates.'
        })
    
    return jsonify({'error': 'Invalid file'}), 400

def process_video_async(video_id, input_path, output_path, options, timestamp):
    """Process video asynchronously with file-based progress tracking"""
    import time
    # Use the same timestamp as the output file for consistency
    progress_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{video_id}_{timestamp}_progress.json")
    start_time = time.time()
    
    logger.info(f"Starting video processing for {video_id}")
    logger.info(f"Input: {input_path}, Output: {output_path}")
    logger.info(f"Options: {options}")
    
    try:
        # Write initial progress file
        initial_progress = {
            'status': 'starting',
            'current_frame': 0,
            'total_frames': 0,
            'progress': 0,
            'fps': 0,
            'time_elapsed': 0,
            'time_remaining': 0,
            'eta_formatted': 'Calculating...',
            'message': 'Analyzing video...',
            'last_update': time.time(),
            'started_at': start_time
        }
        with open(progress_file, 'w') as f:
            json.dump(initial_progress, f)
        
        # Create progress callback that writes to file
        def emit_progress(progress_data):
            current_time = time.time()
            elapsed = current_time - start_time
            current_frame = progress_data.get('current_frame', 0)
            total_frames = progress_data.get('total_frames', 0)
            
            # Calculate actual FPS and time remaining
            if current_frame > 0 and elapsed > 0:
                actual_fps = current_frame / elapsed
                frames_remaining = total_frames - current_frame
                time_remaining = frames_remaining / actual_fps if actual_fps > 0 else 0
                
                # Format time remaining
                if time_remaining > 60:
                    eta_formatted = f"{int(time_remaining / 60)} min {int(time_remaining % 60)} sec"
                else:
                    eta_formatted = f"{int(time_remaining)} sec"
            else:
                actual_fps = 0
                time_remaining = 0
                eta_formatted = "Calculating..."
            
            # Update progress file
            file_progress = {
                'status': 'processing',
                'current_frame': current_frame,
                'total_frames': total_frames,
                'progress': progress_data.get('progress', 0),
                'fps': actual_fps,
                'time_elapsed': elapsed,
                'time_remaining': time_remaining,
                'eta_formatted': eta_formatted,
                'message': f"Processing frame {current_frame}/{total_frames} ({int(progress_data.get('progress', 0) * 100)}%)",
                'last_update': current_time,
                'started_at': start_time
            }
            
            # Write to file
            with open(progress_file, 'w') as f:
                json.dump(file_progress, f)
            
            # Also update in-memory tracker for compatibility
            progress_tracker[video_id] = file_progress
        
        logger.info(f"Initializing processor for {video_id}")
        processor = HolographicOverlayProcessor(options)
        logger.info(f"Starting video processing for {video_id}")
        stats = processor.process_video(input_path, output_path, progress_callback=emit_progress)
        logger.info(f"Processing completed for {video_id}. Stats: {stats}")
        
        # Validate output file was created successfully
        if not os.path.exists(output_path):
            raise Exception(f"Output file was not created: {output_path}")
        
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            raise Exception(f"Output file is empty: {output_path}")
        
        logger.info(f"Output file created successfully: {output_path} ({output_size} bytes)")
        
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)
            logger.info(f"Cleaned up input file: {input_path}")
        
        # Update progress file for completion
        completion_data = {
            'status': 'complete',
            'output': os.path.basename(output_path),
            'stats': stats,
            'time_elapsed': time.time() - start_time,
            'message': 'Processing complete!',
            'last_update': time.time()
        }
        with open(progress_file, 'w') as f:
            json.dump(completion_data, f)
        
        # Update in-memory tracker
        progress_tracker[video_id] = completion_data
        
    except Exception as e:
        # Log detailed error information
        logger.error(f"Processing failed for {video_id}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Create detailed error message
        error_message = f"Processing failed: {str(e)}"
        if "CUDA" in str(e) or "GPU" in str(e):
            error_message += " (GPU-related error - check system configuration)"
        elif "model" in str(e).lower():
            error_message += " (Model loading error - check model files)"
        elif "codec" in str(e).lower() or "video" in str(e).lower():
            error_message += " (Video encoding error - check input file format)"
        
        # Update progress file for error
        error_data = {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'detailed_message': error_message,
            'time_elapsed': time.time() - start_time,
            'message': error_message,
            'last_update': time.time(),
            'traceback': traceback.format_exc()
        }
        
        if os.path.exists(progress_file):
            with open(progress_file, 'w') as f:
                json.dump(error_data, f)
        
        # Update in-memory tracker
        progress_tracker[video_id] = error_data
        
        # Cleanup on error
        if os.path.exists(input_path):
            os.remove(input_path)

# SocketIO handlers removed - using simple polling-based progress tracking

@app.route('/preview', methods=['POST'])
def preview_video():
    logger.info("Received preview request")
    
    if 'video' not in request.files:
        logger.warning("No video file in preview request")
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file and allowed_file(file.filename):
        logger.info(f"Processing preview for file: {file.filename}")
        preview_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{preview_id}_{filename}")
        file.save(input_path)
        
        # Get processing options for preview
        options = {
            'processing_mode': request.form.get('processing_mode', 'david'),
            'david_normal_weight': float(request.form.get('david_normal_weight', 0.85)),
            'david_blend_strength': float(request.form.get('david_blend_strength', 0.9)),
            'david_depth_contribution': float(request.form.get('david_depth_contribution', 0.15)),
            'david_face_focus': float(request.form.get('david_face_focus', 0.75))
        }
        
        # Get frame position (0.0 to 1.0)
        frame_position = float(request.form.get('frame_position', 0.5))
        
        try:
            # Generate preview image from middle frame
            from processor.preview_generator import PreviewGenerator
            preview_gen = PreviewGenerator(options)
            # Use timestamp for unique preview filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_filename = f"{preview_id}_{timestamp}_preview.jpg"
            preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
            
            success = preview_gen.generate_preview_at_position(input_path, preview_path, frame_position)
            
            # Cleanup input file
            os.remove(input_path)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'preview': preview_filename,
                    'message': 'Preview generated successfully'
                })
            else:
                return jsonify({'error': 'Failed to generate preview'}), 500
                
        except Exception as e:
            # Cleanup on error
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'error': f'Preview generation failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Determine if this is a download or stream request
    as_attachment = request.args.get('download', 'false').lower() == 'true'
    
    # For video streaming, use appropriate mimetype
    if filename.endswith('.mp4'):
        return send_file(
            file_path,
            mimetype='video/mp4',
            as_attachment=as_attachment,
            download_name=filename if as_attachment else None
        )
    elif filename.endswith(('.jpg', '.jpeg')):
        return send_file(
            file_path,
            mimetype='image/jpeg',
            as_attachment=as_attachment,
            download_name=filename if as_attachment else None
        )
    else:
        return send_file(
            file_path,
            as_attachment=as_attachment,
            download_name=filename if as_attachment else None
        )

@app.route('/progress/<video_id>')
def get_progress(video_id):
    """Get processing progress for a video"""
    # First check in-memory tracker
    if video_id in progress_tracker:
        return jsonify(progress_tracker[video_id])
    
    # Then check progress file
    progress_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{video_id}_progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                # Check if progress is stale (no update for 30 seconds)
                if 'last_update' in progress_data:
                    import time
                    if time.time() - progress_data['last_update'] > 30 and progress_data.get('status') == 'processing':
                        progress_data['message'] += ' (Processing may have stalled)'
                return jsonify(progress_data)
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Failed to read progress: {str(e)}'}), 500
    
    return jsonify({'status': 'not_found', 'error': 'No progress information available'}), 404

def check_gpu_availability():
    """Check if GPU acceleration is available for all processing modes"""
    try:
        print("🔍 Checking GPU availability...")
        
        # Test DaviD processor
        from processor.david_processor import DaviDProcessor
        test_david = DaviDProcessor(multitask_model_path='DaviD/models/david/multitask-vitl16_384.onnx')
        print("✅ DaviD GPU acceleration: Available")
        
        # Test depth estimator
        from processor.depth_estimator import DepthEstimator
        test_depth = DepthEstimator(mode='fast', use_gpu=True)
        print("✅ Depth estimation GPU acceleration: Available")
        
        # Test DECA processor
        from processor.deca_face_processor import DECAFaceProcessor
        test_deca = DECAFaceProcessor(mode='fast')
        print("✅ DECA GPU acceleration: Available")
        
        print("🚀 All GPU processors initialized successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ GPU acceleration not available: Missing dependencies - {e}")
        return False
    except Exception as e:
        print(f"❌ GPU acceleration failed: {e}")
        return False

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Verify GPU availability at startup
    if not check_gpu_availability():
        print("⚠️ WARNING: GPU acceleration not available. Performance will be severely impacted.")
        print("Please check your system configuration and ensure all dependencies are installed.")
    
    # Run with regular Flask (SocketIO removed for simplicity)
    app.run(debug=True, port=5001)  # Use port 5001 to avoid AirPlay conflict