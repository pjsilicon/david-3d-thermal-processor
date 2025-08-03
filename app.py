from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import uuid
from processor.main_pipeline import HolographicOverlayProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file and allowed_file(file.filename):
        video_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        file.save(input_path)
        
        # Get processing options
        options = {
            'processing_mode': request.form.get('processing_mode', 'david'),  # Default to DaviD mode
            'depth_mode': request.form.get('depth_mode', 'fast'),
            'intensity': float(request.form.get('intensity', 0.6)),
            'use_gpu': request.form.get('use_gpu', 'true').lower() == 'true',
            # DaviD-specific parameters
            'david_normal_weight': float(request.form.get('david_normal_weight', 0.85)),
            'david_blend_strength': float(request.form.get('david_blend_strength', 0.9)),
            'david_depth_contribution': float(request.form.get('david_depth_contribution', 0.15)),
            'david_face_focus': float(request.form.get('david_face_focus', 0.75))
        }
        
        # Process video (in production, use Celery for async)
        output_filename = f"{video_id}_hologram.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        progress_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{video_id}_progress.json")
        
        processor = HolographicOverlayProcessor(options)
        stats = processor.process_video(input_path, output_path, progress_file=progress_file)
        
        # Cleanup
        os.remove(input_path)
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            
        return jsonify({
            'status': 'complete',
            'output': output_filename,
            'stats': stats,
            'video_id': video_id
        })
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/progress/<video_id>')
def get_progress(video_id):
    progress_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{video_id}_progress.json")
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.loads(f.read())
                return jsonify(progress_data)
        except:
            return jsonify({'error': 'Failed to read progress'}), 500
    else:
        return jsonify({'error': 'Progress not found'}), 404

@app.route('/preview', methods=['POST'])
def preview_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file and allowed_file(file.filename):
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
        
        try:
            # Generate preview image from middle frame
            from processor.preview_generator import PreviewGenerator
            preview_gen = PreviewGenerator(options)
            preview_filename = f"{preview_id}_preview.jpg"
            preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
            
            success = preview_gen.generate_preview(input_path, preview_path)
            
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

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5001)  # Use port 5001 to avoid AirPlay conflict