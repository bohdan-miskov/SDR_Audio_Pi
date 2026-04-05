import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.audio_stream import AudioStream
from src.core.audio_logger import SmartAudioLogger
from src.dsp.goertzel import GoertzelFilter


class ProcessingWorker(QThread):
    audio_data_ready = pyqtSignal(np.ndarray)
    detection_result_ready = pyqtSignal(dict, float)
    goertzel_ready = pyqtSignal(dict)

    def __init__(self, config, detector):
        super().__init__()
        self.config = config
        self.detector = detector
        self.running = True

        root = Path(__file__).resolve().parent.parent.parent
        log_dir = root / config['paths']['log_dir']

        self.mic = AudioStream(
            rate=config['device']['rate'],
            chunk_size=config['device']['chunk_size']
        )

        self.logger = SmartAudioLogger(
            save_dir=str(log_dir),
            rate=config['device']['rate'],
            history_seconds=config['logging']['history_seconds']
        )

        self.goertzel = GoertzelFilter(sample_rate=config['device']['rate'])

    def run(self):
        print("Потік обробки запущено. Чекаю на звук...")
        while self.running:
            audio_chunk = self.mic.get_audio_chunk()
            self.audio_data_ready.emit(audio_chunk)

            freq_results = self.goertzel.analyze_drone_harmonics(
                audio_chunk,
                self.config['detection']['goertzel_frequencies']
            )
            self.goertzel_ready.emit(freq_results)

            amp = np.max(np.abs(audio_chunk))

            if amp > 0.001:
                result = self.detector.predict(audio_chunk, rate=self.config['device']['rate'])

                print(f"ШІ каже: {result['class']} ({result['confidence']}%) | Гучність: {amp:.4f}")

                is_drone = (result['class'] == 'drone' and
                            result['confidence'] > self.config['detection']['threshold_confidence'])

                if is_drone:
                    print("ТРИВОГА! Нейромережа впізнала дрон!")
                    self.logger.trigger_record(self.config['logging']['post_record_seconds'])

                self.detection_result_ready.emit(result, amp)

            self.logger.feed_audio(audio_chunk)

    def trigger_manual_record(self):
        self.logger.trigger_record(self.config['logging']['post_record_seconds'])

    def stop(self):
        self.running = False
        self.mic.close()
        self.wait()