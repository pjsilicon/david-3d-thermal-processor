import cv2
import ffmpeg
import numpy as np
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
            
    def create_writer(self, codec='mp4v') -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(
            self.output_path, fourcc, self.fps, 
            (self.width, self.height)
        )
        
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