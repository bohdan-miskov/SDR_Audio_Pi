import sys
import queue
import numpy as np
import sounddevice as sd
import logging
import uuid
from datetime import datetime
from collections import deque
from pathlib import Path

from src.ml.detect import DroneDetector
from src.utils.audio_logger import SmartAudioLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LiveScanner")

class Config: pass
setattr(sys.modules['__main__'], 'Config', Config)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
BUFFER_SECONDS = 5.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_CHUNKS = int(BUFFER_SECONDS / CHUNK_DURATION)

# НАЛАШТУВАННЯ РАДАРА: 4 з 5 останніх кадрів мають бути дроном
WINDOW_SIZE = 5
REQUIRED_DRONE_FRAMES = 4

def main(detection_queue=None):
    """
    detection_queue: об'єкт queue.Queue(), через який ми передаватимемо 
    пакети з тривогою нашому TCP-серверу для відправки на графічний інтерфейс.
    """
    src_dir = Path(__file__).resolve().parent.parent
    m_path = src_dir / 'models' / 'convn.keras'
    p_path = src_dir / 'models' / 'convn.p'

    detector = DroneDetector(model_path=str(m_path), pickle_path=str(p_path))
    audio_logger = SmartAudioLogger(sample_rate=SAMPLE_RATE)

    audio_buffer = deque(maxlen=MAX_CHUNKS)
    audio_queue = queue.Queue()
    
    # Історія останніх 5 кадрів
    prediction_history = deque(maxlen=WINDOW_SIZE)

    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Статус аудіопотоку: {status}")
        audio_queue.put(indata.copy())

    logger.info("Запуск системи (ШІ + Логер з ковзним вікном)...")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, dtype='int16', callback=audio_callback):
            while True:
                chunk = audio_queue.get()
                audio_buffer.append(chunk)

                mono_chunk = chunk[:, 0]
                result = detector.predict(mono_chunk, rate=SAMPLE_RATE)
                volume = np.max(np.abs(mono_chunk))
                
                # Якщо це дрон з впевненістю > 60%, записуємо 1, інакше 0
                is_drone = 1 if (result['class'] == 'drone' and result['confidence'] > 60.0) else 0
                prediction_history.append(is_drone)

                # Рахуємо, скільки дронів було за останні 5 кадрів
                drones_in_window = sum(prediction_history)

                logger.info(f"Аналіз: {result['class']} ({result['confidence']:.1f}%) | Дронів у вікні: {drones_in_window}/{WINDOW_SIZE} | Гучність: {volume}")

                # Якщо набрали 4 підтвердження з 5 можливих
                if drones_in_window >= REQUIRED_DRONE_FRAMES:
                    logger.warning(f"🚨🚨🚨 ДРОН ПІДТВЕРДЖЕНО ({result['confidence']:.1f}%) 🚨🚨🚨")

                    # Формуємо пакет даних для відправки на інтерфейс
                    detection_payload = {
                        "action": "detection",
                        "data": {
                            "id": str(uuid.uuid4()),
                            "type": "SOUND",
                            "name": result['class'],
                            "object_class": "UAV",
                            "confidence": float(result['confidence']),
                            "timestamp": datetime.now().isoformat(),
                            "distance_km": 0.5,
                            "angle": 0.0,
                            "frequency_hz": 0.0
                        }
                    }

                    if detection_queue is not None:
                        detection_queue.put(detection_payload)

                    audio_logger.save_event(list(audio_buffer))
                    audio_buffer.clear()
                    prediction_history.clear() 
    except KeyboardInterrupt:
        logger.info("Сканування зупинено користувачем.")

if __name__ == "__main__":
    main()