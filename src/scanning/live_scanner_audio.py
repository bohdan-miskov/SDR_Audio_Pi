import sys
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from pathlib import Path

from src.ml.detect import DroneDetector
from src.utils.audio_logger import SmartAudioLogger
from src.triangulation.compute_correlation import get_delays
from src.triangulation.tdoa import calculate_position

class Config: pass  # noqa: E701
setattr(sys.modules['__main__'], 'Config', Config)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
BUFFER_SECONDS = 5.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_CHUNKS = int(BUFFER_SECONDS / CHUNK_DURATION)

# Приклад для 4-х мікрофонів, розставлених квадратом 10х10 см:
MIC_X = [0.0, 0.1, 0.0, -0.1]
MIC_Y = [0.0, 0.0, 0.1, 0.0]
MIC_Z = [0.0, 0.0, 0.0, 0.0]

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
        audio_queue.put(indata.copy())

    print("Запуск системи (ШІ + Логер + Тріангуляція)...")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=4, blocksize=CHUNK_SIZE, callback=audio_callback):
            while True:
                chunk = audio_queue.get()
                audio_buffer.append(chunk)

                mono_chunk = chunk[:, 0]
                result = detector.predict(mono_chunk, rate=SAMPLE_RATE)

                if result['class'] == 'drone' and result['confidence'] > 75.0:
                    print(f"\n🚨 ДРОН ({result['confidence']:.1f}%)")

                    full_multichannel_audio = np.concatenate(list(audio_buffer))
                    try:
                        delays = get_delays(full_multichannel_audio, sample_rate=SAMPLE_RATE)
                        x, y, z = calculate_position(delays, MIC_X, MIC_Y, MIC_Z)
                        print(f"🎯 Координати цілі: X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m")
                    except Exception as e:
                        print(f"⚠️ Помилка розрахунку координат: {e}")

                    logger.save_event(list(audio_buffer))
                    
                    audio_buffer.clear()

    except KeyboardInterrupt:
        print("\nЗупинено.")

if __name__ == "__main__":
    main()