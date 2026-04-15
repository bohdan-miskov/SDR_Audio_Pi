"""
FFTProcessor — FFT, кроп, згладжування та маска шуму.
Використовує scipy.fft для кращої продуктивності та scipy.signal
для вікон та згладжування.
"""
import numpy as np
from scipy.fft import fft, fftshift, fftfreq, next_fast_len
from scipy.signal.windows import blackman
from scipy.signal import fftconvolve

from core.config import SAMPLE_RATE


class FFTProcessor:
    """
    Обчислює спектральну щільність потужності з IQ-семплів.
    Включає: FFT з вікном Blackman, кроп країв, MA-згладжування, маску шуму.

    Переваги scipy.fft над numpy.fft:
    - next_fast_len() — автоматичний підбір оптимального розміру FFT
    - Планувальник (pyfftw-сумісний backend)
    - Кращі чисельні характеристики
    """

    # Константи обробки
    CROP_PERCENT: float = 0.15     # Частка семплів, що відкидається з кожного боку
    SMOOTH_KERNEL: int = 5         # Розмір ковзного середнього (семплів)
    MASK_MARGIN_DB: float = 8.0    # Запас маски над поточним рівнем шуму (дБ)

    def __init__(self):
        self._mask_enabled: bool = False
        self._mask_spectrum: np.ndarray | None = None
        self._mask_freqs: np.ndarray | None = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def mask_enabled(self) -> bool:
        return self._mask_enabled

    @property
    def mask_spectrum(self) -> np.ndarray | None:
        return self._mask_spectrum

    @property
    def mask_freqs(self) -> np.ndarray | None:
        return self._mask_freqs

    # ── Public API ───────────────────────────────────────────────────────────

    def enable_mask(self) -> None:
        """Вмикає маску (захопиться при першому process())."""
        self._mask_enabled = True
        self._mask_spectrum = None

    def clear_mask(self) -> None:
        """Вимикає та скидає маску."""
        self._mask_enabled = False
        self._mask_spectrum = None
        self._mask_freqs = None

    def process(self, iq_samples: np.ndarray, center_freq: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Повний конвеєр обробки: FFT → кроп → згладжування → маска.

        Returns:
            (freqs, psd) — обрізані масиви частот (Гц) та рівнів (dB).
        """
        N = len(iq_samples)

        # Оптимальна довжина FFT (2^n або 2^n * 3^m * 5^k — швидша scipy.fft)
        N_fft = next_fast_len(N)

        # 1. Вікно Blackman (scipy.signal.windows) + FFT (scipy.fft)
        window = blackman(N)
        fft_raw = fft(iq_samples * window, n=N_fft)
        fft_shifted = fftshift(fft_raw) / N
        psd = 20 * np.log10(np.abs(fft_shifted) + 1e-15)

        # 2. Згладжування через fftconvolve (швидше за np.convolve для великих масивів)
        kernel = np.ones(self.SMOOTH_KERNEL) / self.SMOOTH_KERNEL
        psd = fftconvolve(psd, kernel, mode='same')

        # 3. Частотна сітка (scipy.fft.fftfreq)
        freqs = fftshift(fftfreq(N_fft, d=1.0 / SAMPLE_RATE)) + center_freq

        # 4. Кроп країв
        cut_idx = int(N_fft * self.CROP_PERCENT)
        if cut_idx > 0 and N_fft > 2 * cut_idx:
            psd = psd[cut_idx:-cut_idx]
            freqs = freqs[cut_idx:-cut_idx]

        # 5. Маска шуму
        if self._mask_enabled:
            if self._mask_spectrum is None:
                self._mask_spectrum = psd.copy() + self.MASK_MARGIN_DB
                self._mask_freqs = freqs.copy()
                print(f"[FFT] Mask captured (+{self.MASK_MARGIN_DB} dB margin), N_fft={N_fft}")
            psd = psd - self._mask_spectrum
            psd[psd < 0] = 0.0

        return freqs, psd
