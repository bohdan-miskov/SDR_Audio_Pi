import sys
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from pathlib import Path
from src.ml.detect import DroneDetector
from src.utils.audio_logger import SmartAudioLogger


class Config: pass


setattr(sys.modules['__main__'], 'Config', Config)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
BUFFER_SECONDS = 5.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_CHUNKS = int(BUFFER_SECONDS / CHUNK_DURATION)


def main():
    src_dir = Path(__file__).resolve().parent.parent
    m_path = src_dir / 'models' / 'conv.keras'
    p_path = src_dir / 'models' / 'conv.p'

    detector = DroneDetector(model_path=str(m_path), pickle_path=str(p_path))
    logger = SmartAudioLogger(sample_rate=SAMPLE_RATE)

    audio_buffer = deque(maxlen=MAX_CHUNKS)
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    print("Запуск мікрофона...")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=audio_callback):
            while True:
                chunk = audio_queue.get()
                audio_buffer.append(chunk)

                result = detector.predict(chunk, rate=SAMPLE_RATE)

                if result['class'] == 'drone' and result['confidence'] > 75.0:
                    print(f"ДРОН ({result['confidence']:.1f}%)")
                    logger.save_event(list(audio_buffer))
                    audio_buffer.clear()

    except KeyboardInterrupt:
        print("\nЗупинено.")


if __name__ == "__main__":
    main()