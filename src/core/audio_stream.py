import pyaudio
import numpy as np

class AudioStream:
    def __init__(self, rate=16000, chunk_size=2048, device_index=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size
        )

    def get_audio_chunk(self):
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            audio_data /= 32768.0

            max_amp = np.max(np.abs(audio_data))
            if max_amp > 0.05:
                audio_data /= max_amp

            return audio_data

        except Exception:
            return np.zeros(self.chunk_size, dtype=np.float32)

    def close(self):
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()