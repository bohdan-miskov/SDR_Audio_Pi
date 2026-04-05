import os
import sys
from pathlib import Path

# глушимо зайві логи tensorflow, щоб не спамив у консоль малинки
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf

    print(f"✅ TF завантажено. Версія: {tf.__version__}")
except Exception as e:
    print(f"❌ Помилка TF: {e}")

import yaml
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import RadarWindow
from src.core.processing_worker import ProcessingWorker
from src.ml.detect import DroneDetector


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # ініціалізуємо графіку (PyQt6)
    app = QApplication(sys.argv)

    # збираємо правильні абсолютні шляхи до моделей
    base_dir = Path(__file__).parent
    m_path = str(base_dir / config['paths']['model_keras'])
    p_path = str(base_dir / config['paths']['model_pickle'])

    detector = DroneDetector(model_path=m_path, pickle_path=p_path)

    window = RadarWindow(config)
    worker = ProcessingWorker(config, detector)

    # підключаємо сигнали (UI <-> Worker)
    worker.audio_data_ready.connect(window.update_plot)
    worker.goertzel_ready.connect(window.update_goertzel_status)
    worker.detection_result_ready.connect(
        lambda res, vol: window.update_detection(res['class'] == 'drone', res['confidence'])
    )

    window.rec_button.clicked.connect(worker.trigger_manual_record)

    # поїхали!
    worker.start()
    window.show()

    exit_code = app.exec()

    # правильно глушимо мікрофон і потік при закритті хрестиком
    worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    # милиця для старого pickle, щоб він нормально розпакував налаштування
    class Config: pass


    setattr(sys.modules['__main__'], 'Config', Config)

    main()