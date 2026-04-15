"""
AmplitudeSyncDetector — синхронізація антенного циклу методом Amplitude Blanking.

Принцип:
  Мультиплексор щотіку вмикає антени за циклом:
      ANT_A → ANT_B → ANT_C → BLANK → ANT_A → ...
  У крок BLANK всі антени відключені — сигнал різко падає.
  Цей «провал» амплітуди є стабільним маркером початку нового циклу.

  DSP знаходить ці провали у IQ-потоці, вирівнює дані та повертає
  IQ-семпли для кожної антени окремо.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from core.config import SAMPLE_RATE, ANTENNA_STEP_MS, N_ANTENNAS


class AmplitudeSyncDetector:
    """
    Знаходить в IQ-буфері маркерні «провали» амплітуди (BLANK-кроки)
    та нарізає дані на сегменти по антенах.

    Параметри (задаються у core/config/scan_settings.py):
        ANTENNA_STEP_MS  — тривалість одного кроку (мс)
        N_ANTENNAS       — кількість активних антен (без BLANK)

    Таймінг:
        step_samples  = SAMPLE_RATE * ANTENNA_STEP_MS / 1000
        cycle_samples = step_samples * (N_ANTENNAS + 1)   # +1 = BLANK
    """

    # Частка від step_samples, що вважається «провалом» (поріг нижче медіани)
    BLANK_DEPTH_FACTOR: float = 0.4   # у BLANK сигнал < 40 % від норми
    # Частка step_samples для згладжування амплітуди
    SMOOTH_WINDOW_FACTOR: float = 0.05

    def __init__(self):
        self._step_samples: int = int(SAMPLE_RATE * ANTENNA_STEP_MS / 1000)
        self._n_antennas: int = N_ANTENNAS
        self._cycle_samples: int = self._step_samples * (self._n_antennas + 1)
        self._smooth_win: int = max(3, int(self._step_samples * self.SMOOTH_WINDOW_FACTOR))

        # Стан для diagnostics / GUI
        self._last_markers: list[int] = []
        self._last_offsets: list[int] = []
        self._sync_quality: float = 0.0   # 0..1 (1 = ідеальна синхронізація)

        print(
            f"[AmpSync] step={self._step_samples} samp, "
            f"cycle={self._cycle_samples} samp, "
            f"antennas={self._n_antennas}"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def step_samples(self) -> int:
        """Кількість семплів на один крок (одну антену або BLANK)."""
        return self._step_samples

    @property
    def cycle_samples(self) -> int:
        """Кількість семплів у повному циклі (N антен + 1 BLANK)."""
        return self._cycle_samples

    @property
    def last_markers(self) -> list[int]:
        """Індекси знайдених BLANK-маркерів у останньому буфері."""
        return list(self._last_markers)

    @property
    def sync_quality(self) -> float:
        """Якість синхронізації 0..1 (1 = ідеальна рівномірність маркерів)."""
        return self._sync_quality

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, iq_samples: np.ndarray) -> dict[str, np.ndarray] | None:
        """
        Основний метод. Знаходить BLANK-маркери та нарізає IQ на антенні сегменти.

        Args:
            iq_samples: Сирі IQ-семпли з PlutoSDR (complex64).

        Returns:
            Словник {'ANT_A': ndarray, 'ANT_B': ndarray, 'ANT_C': ndarray}
            або None — якщо маркери не знайдені (менше 2).
        """
        if len(iq_samples) < self._cycle_samples * 2:
            return None   # Замало даних для хоч одного повного циклу

        markers = self._find_blank_markers(iq_samples)
        self._last_markers = markers
        self._update_sync_quality(markers)

        if len(markers) < 2:
            return None

        return self._slice_antennas(iq_samples, markers)

    def detect_only(self, iq_samples: np.ndarray) -> list[int]:
        """
        Тільки шукає маркери, без нарізки (корисно для діагностики).
        """
        markers = self._find_blank_markers(iq_samples)
        self._last_markers = markers
        self._update_sync_quality(markers)
        return markers

    # ── Private: Marker detection ─────────────────────────────────────────────

    def _find_blank_markers(self, iq_samples: np.ndarray) -> list[int]:
        """
        Знаходить позиції BLANK-провалів в IQ-буфері.

        Алгоритм:
          1. Обчислюємо обгортку (envelope) = |IQ|
          2. Згладжуємо uniform_filter1d для прибирання шуму
          3. Визначаємо поріг: медіана * BLANK_DEPTH_FACTOR
          4. Інвертуємо і шукаємо піки (= мінімуми вихідного сигналу)
             з мінімальною відстанню 80 % від cycle_samples
        """
        amplitude = np.abs(iq_samples)
        smoothed = uniform_filter1d(amplitude.astype(float), size=self._smooth_win)

        median_amp = float(np.median(smoothed))
        if median_amp < 1e-10:
            return []   # Немає сигналу взагалі

        blank_threshold = median_amp * self.BLANK_DEPTH_FACTOR

        # Шукаємо мінімуми через інвертований сигнал
        inverted = -smoothed
        min_distance = int(self._cycle_samples * 0.8)

        # height: піки повинні бути вищими за -blank_threshold (тобто нижчими за blank_threshold)
        peaks, props = find_peaks(
            inverted,
            height=-blank_threshold,
            distance=min_distance,
            prominence=median_amp * 0.3,
        )

        return peaks.tolist()

    # ── Private: Antenna slicing ──────────────────────────────────────────────

    def _slice_antennas(self, iq_samples: np.ndarray,
                        markers: list[int]) -> dict[str, np.ndarray]:
        """
        Нарізає IQ-дані на сегменти по антенах, починаючи від першого маркера.

        Структура одного циклу після BLANK:
          [ANT_A: step_samples][ANT_B: step_samples][ANT_C: step_samples][BLANK: step_samples]

        Для надійності береться середина кожного кроку (+/- 20 % від країв),
        щоб уникнути перехідних процесів перемикання GPIO.
        """
        s = self._step_samples
        guard = int(s * 0.20)   # 20 % — перехідний процес на початку та кінці кроку
        useful = slice(guard, s - guard)

        ant_names = [f'ANT_{chr(65 + i)}' for i in range(self._n_antennas)]  # ANT_A, ANT_B, ANT_C
        segments: dict[str, list[np.ndarray]] = {name: [] for name in ant_names}
        offsets: list[int] = []

        for marker in markers:
            start = marker + s   # одразу після BLANK-кроку → початок ANT_A
            for i, name in enumerate(ant_names):
                seg_start = start + i * s
                seg_end = seg_start + s
                if seg_end > len(iq_samples):
                    break
                segments[name].append(iq_samples[seg_start:seg_end][useful])
            offsets.append(start)

        self._last_offsets = offsets

        # Конкатенуємо всі сегменти кожної антени в один масив
        result: dict[str, np.ndarray] = {}
        for name in ant_names:
            if segments[name]:
                result[name] = np.concatenate(segments[name])
            else:
                result[name] = np.zeros(0, dtype=np.complex64)

        return result

    # ── Private: Quality metric ───────────────────────────────────────────────

    def _update_sync_quality(self, markers: list[int]) -> None:
        """
        Оцінює рівномірність маркерів:
        якість = 1 − (std відхилень між маркерами) / (очікуваний крок)
        """
        if len(markers) < 2:
            self._sync_quality = 0.0
            return

        gaps = np.diff(markers)
        expected = self._cycle_samples
        deviation = float(np.std(gaps)) / expected
        self._sync_quality = float(np.clip(1.0 - deviation * 5, 0.0, 1.0))
