import time
import numpy as np
from scipy.io import wavfile
from pathlib import Path


class SmartAudioLogger:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.folder = Path(__file__).resolve().parent.parent.parent / 'data' / 'recorded_alerts'
        self.folder.mkdir(parents=True, exist_ok=True)

    def save_event(self, audio_buffer):
        if not audio_buffer:
            return

        full_audio = np.concatenate(audio_buffer)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.folder / f"live_drone_{timestamp}.wav"

        wavfile.write(str(filename), self.sample_rate, full_audio)
        print(f"Збережено: {filename.name}")