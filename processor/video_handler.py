import cv2
import ffmpeg
import numpy as np
import subprocess
from typing import Tuple, Generator, Optional
import os

class VideoHandler:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(input_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
            
    def create_writer(self, quality='high') -> cv2.VideoWriter:
        """
        Create a reliable video writer with multiple fallback options.
        
        Args:
            quality: 'draft' (fast), 'high' (H.264 CRF 23), 'cinema' (H.264 CRF 18)
        """
        # List of codecs to try in order of preference
        codec_options = [
            ('avc1', 'H.264 (preferred)'),
            ('h264', 'H.264 (alt)'),
            ('mp4v', 'MPEG-4'),
            ('XVID', 'Xvid'),
            ('MJPG', 'Motion JPEG')
        ]
        
        writer = None
        successful_codec = None
        
        for codec, codec_name in codec_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(
                    self.output_path, fourcc, self.fps,
                    (self.width, self.height), True
                )
                
                if test_writer.isOpened():
                    writer = test_writer
                    successful_codec = codec_name
                    break
                else:
                    test_writer.release()
                    
            except Exception as e:
                print(f"Failed to create writer with {codec}: {e}")
                continue
        
        if writer is None:
            raise Exception("Failed to create video writer with any codec")
        
        quality_msg = {
            'cinema': '🎬 Cinema quality',
            'high': '📹 High quality', 
            'draft': '⚡ Draft quality'
        }.get(quality, '📹 Standard quality')
        
        print(f"{quality_msg} encoding using {successful_codec}")
        return writer
    
    def create_ffmpeg_writer(self, quality='high'):
        """
        Create an FFmpeg-based writer for superior quality control.
        This bypasses OpenCV's limitations for maximum quality.
        """
        if quality == 'cinema':
            # Near-lossless with CRF 18
            crf = '18'
            preset = 'slower'  # Best compression efficiency
        elif quality == 'high':
            # High quality with CRF 23
            crf = '23' 
            preset = 'slow'    # Good compression efficiency
        else:
            # Draft with CRF 28
            crf = '28'
            preset = 'fast'    # Fast encoding
            
        # Create FFmpeg command for high-quality H.264
        self.ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo', 
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-vcodec', 'libx264',
            '-crf', crf,
            '-preset', preset,
            '-pix_fmt', 'yuv420p',  # Ensure compatibility
            self.output_path
        ]
        
        # Start FFmpeg process
        self.ffmpeg_process = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"🎬 Using FFmpeg H.264 encoding: CRF {crf}, preset {preset}")
        return self  # Return self to maintain VideoWriter-like interface
    
    def write_ffmpeg_frame(self, frame):
        """Write frame to FFmpeg process"""
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process.poll() is None:
            self.ffmpeg_process.stdin.write(frame.tobytes())
        
    def release_ffmpeg(self):
        """Close FFmpeg process"""
        if hasattr(self, 'ffmpeg_process'):
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        
    def extract_audio(self) -> Optional[str]:
        audio_path = self.output_path.replace('.mp4', '_audio.aac')
        try:
            stream = ffmpeg.input(self.input_path)
            stream = ffmpeg.output(stream.audio, audio_path, acodec='aac')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            return audio_path
        except:
            return None
            
    def remux_with_audio(self, video_path: str, audio_path: str):
        temp_output = video_path.replace('.mp4', '_temp.mp4')
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        out = ffmpeg.output(
            video, audio, temp_output, 
            vcodec='copy', acodec='copy'
        )
        ffmpeg.run(out, overwrite_output=True)
        os.replace(temp_output, video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    def validate_output(self) -> tuple[bool, str]:
        """
        Validate that the output video file was created successfully.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            if not os.path.exists(self.output_path):
                return False, f"Output file does not exist: {self.output_path}"
            
            file_size = os.path.getsize(self.output_path)
            if file_size == 0:
                return False, f"Output file is empty: {self.output_path}"
            
            # Try to open the video file to validate it
            test_cap = cv2.VideoCapture(self.output_path)
            if not test_cap.isOpened():
                test_cap.release()
                return False, f"Output file cannot be opened as video: {self.output_path}"
            
            # Check if we can read at least one frame
            ret, frame = test_cap.read()
            test_cap.release()
            
            if not ret or frame is None:
                return False, f"Output file contains no readable video data: {self.output_path}"
            
            return True, f"Output file validated successfully: {self.output_path} ({file_size} bytes)"
            
        except Exception as e:
            return False, f"Error validating output file: {str(e)}"