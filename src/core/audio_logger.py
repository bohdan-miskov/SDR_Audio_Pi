import numpy as np
import soundfile as sf
from collections import deque
from datetime import datetime
from pathlib import Path


class SmartAudioLogger:
    def __init__(self, save_dir="src/data/recorded_alerts", rate=16000, history_seconds=3, max_storage_mb=500):
        self.save_dir = Path(save_dir)
        self.rate = rate
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.history_buffer = deque(maxlen=history_seconds)
        self.is_recording = False
        self.frames_to_record = []
        self.target_record_chunks = 0
        self.current_chunks = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def feed_audio(self, audio_chunk):
        if not self.is_recording:
            self.history_buffer.append(audio_chunk)
        else:
            self.frames_to_record.append(audio_chunk)
            self.current_chunks += 1
            if self.current_chunks >= self.target_record_chunks:
                self._save_to_disk()

    def trigger_record(self, post_record_seconds=7):
        if self.is_recording:
            return

        print("[РЕЄСТРАТОР] Запис пішов...")
        self.is_recording = True
        self.target_record_chunks = post_record_seconds
        self.current_chunks = 0
        self.frames_to_record = list(self.history_buffer)

    def _cleanup_old_files(self):
        all_files = sorted(
            self.save_dir.rglob('*.flac'),
            key=lambda x: x.stat().st_mtime
        )

        total_size = sum(f.stat().st_size for f in all_files)

        while total_size > self.max_storage_bytes and all_files:
            oldest = all_files.pop(0)
            total_size -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            print(f"[ОЧИЩЕННЯ] Видалено: {oldest.name}")

    def _save_to_disk(self):
        today = datetime.now().strftime('%Y-%m-%d')
        daily_path = self.save_dir / today
        daily_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%H-%M-%S')
        filename = f"drone_alert_{timestamp}.flac"
        filepath = daily_path / filename

        full_audio = np.concatenate(self.frames_to_record)

        try:
            sf.write(str(filepath), full_audio, self.rate, format='FLAC', subtype='PCM_16')

            size_kb = filepath.stat().st_size / 1024
            print(f"[ЗБЕРЕЖЕНО] {filename} ({size_kb:.1f} КБ)")

            self._cleanup_old_files()
        except Exception as e:
            print(f"[ПОМИЛКА ЗАПИСУ] {e}")
        finally:
            self.is_recording = False
            self.frames_to_record.clear()