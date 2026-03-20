"""
SimulationGenerator — генератор тестових IQ-сигналів.
Використовує scipy.fft замість numpy.fft для кращої продуктивності.
"""
import numpy as np
import random
import time
from scipy.fft import fft, ifft, fftfreq, next_fast_len

from core.config import SAMPLE_RATE, BUFFER_SIZE


class SimulationGenerator:
    """Генерує реалістичний шум та тестові сигнали дронів."""

    def __init__(self):
        self._targets = [
            {"freq": 433e6,  "type": "LORA 433",  "bw": 0.2e6,  "power": 25, "style": "hop"},
            {"freq": 868e6,  "type": "ELRS 868",  "bw": 0.5e6,  "power": 30, "style": "hop"},
            {"freq": 915e6,  "type": "CROSS 915", "bw": 0.5e6,  "power": 30, "style": "hop"},
            {"freq": 1200e6, "type": "VIDEO 1.2", "bw": 8.0e6,  "power": 20, "style": "plateau"},
            {"freq": 2400e6, "type": "WIFI 2.4",  "bw": 20.0e6, "power": 15, "style": "arch"},
            {"freq": 5800e6, "type": "VIDEO 5.8", "bw": 14.0e6, "power": 25, "style": "plateau"},
        ]

        self._hop_states: dict[int, dict] = {
            i: {"last_hop": 0.0, "curr_freq": t["freq"]}
            for i, t in enumerate(self._targets)
        }

        # Попередньо обчислюємо оптимальну довжину FFT для BUFFER_SIZE
        self._N_fft = next_fast_len(BUFFER_SIZE)

    @property
    def targets(self) -> list[dict]:
        """Список симульованих цілей (read-only)."""
        return list(self._targets)

    def get_iq_samples(self, center_freq: float, gain_level: float) -> np.ndarray:
        """
        Генерує IQ-буфер розміром BUFFER_SIZE.
        Використовує scipy.fft для ifft зворотного перетворення.
        """
        # Частотна сітка (scipy.fft.fftfreq)
        freq_bins = fftfreq(self._N_fft, d=1.0 / SAMPLE_RATE) + center_freq

        # Базовий шум (~-90 dB)
        base_noise_amp = 0.0001 * (10 ** (gain_level / 100.0))
        spectrum = (
            np.random.normal(0, 1, self._N_fft)
            + 1j * np.random.normal(0, 1, self._N_fft)
        ) * base_noise_amp

        for i, t in enumerate(self._targets):
            if abs(center_freq - t["freq"]) > 60e6:
                continue

            sig_amp = base_noise_amp * (10 ** (t["power"] / 20.0))

            if t["style"] == "hop":
                state = self._hop_states[i]
                if time.time() - state["last_hop"] > 0.1:
                    state["last_hop"] = time.time()
                    state["curr_freq"] = t["freq"] + random.randint(-5, 5) * 1e6
                mask = np.abs(freq_bins - state["curr_freq"]) < t["bw"] / 2

            elif t["style"] == "plateau":
                mask = np.abs(freq_bins - t["freq"]) < t["bw"] / 2

            elif t["style"] == "arch":
                mask = np.abs(freq_bins - t["freq"]) < t["bw"] / 2

            else:
                continue

            if not np.any(mask):
                continue

            n = int(np.sum(mask))
            noise_chunk = (
                np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
            ) * sig_amp

            if t["style"] == "arch":
                indices = np.where(mask)[0]
                x = (freq_bins[indices] - t["freq"]) / (t["bw"] / 2)
                envelope = np.sqrt(np.maximum(0, 1 - x ** 2))
                noise_chunk *= envelope

            spectrum[mask] += noise_chunk

        # scipy.fft.ifft (швидше та точніше для оптимізованих розмірів)
        result = ifft(spectrum)
        # Повертаємо рівно BUFFER_SIZE семплів
        return result[:BUFFER_SIZE].astype(np.complex64)
