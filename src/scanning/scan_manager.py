"""
ScanManager — управління списком частот для сканування.
"""
import numpy as np
from core.config import SCAN_RANGES, SAMPLE_RATE


class ScanManager:
    """
    Будує та обходить список центральних частот для скануючого режиму.
    Підтримує стандартний діапазон (з SCAN_RANGES) та довільний.
    """

    def __init__(self):
        self._freq_list: list[float] = []
        self._index: int = 0
        self._build_default_list()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def freq_list(self) -> list[float]:
        """Поточний список частот сканування (Гц)."""
        return list(self._freq_list)

    @property
    def index(self) -> int:
        """Поточний індекс у списку частот."""
        return self._index

    @property
    def step_count(self) -> int:
        """Кількість кроків у поточному діапазоні."""
        return len(self._freq_list)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_next_frequency(self) -> float:
        """Повертає наступну частоту та просуває індекс по колу."""
        if not self._freq_list:
            return 433e6
        f = self._freq_list[self._index]
        self._index = (self._index + 1) % len(self._freq_list)
        return float(f)

    def set_custom_range(self, start_mhz: float, stop_mhz: float) -> list[float]:
        """
        Встановлює довільний діапазон сканування.

        Args:
            start_mhz: Початкова частота (МГц).
            stop_mhz:  Кінцева частота (МГц).

        Returns:
            Список центральних частот (Гц).
        """
        start_hz = start_mhz * 1e6
        stop_hz = stop_mhz * 1e6

        if stop_hz <= start_hz:
            self._freq_list = [start_hz]
        else:
            freq_arr = np.arange(start_hz, stop_hz, SAMPLE_RATE)
            self._freq_list = freq_arr.tolist() if len(freq_arr) > 0 else [start_hz]

        self._index = 0
        print(f"[Scanner] New range: {start_mhz}-{stop_mhz} MHz. Steps: {self.step_count}")
        print(f"[Scanner] Grid: {[f / 1e6 for f in self._freq_list]}")
        return list(self._freq_list)

    def get_priority_freqs(self) -> list[int]:
        """Пріоритетні частоти для швидкої перевірки (МГц)."""
        return [433, 868, 915, 1200, 2400, 5800]

    def reset(self) -> None:
        """Скидає індекс до початку."""
        self._index = 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_default_list(self) -> None:
        """Будує список частот зі стандартних діапазонів SCAN_RANGES."""
        result = []
        for start, stop, _step in SCAN_RANGES:
            chunk = np.arange(start * 1e6, stop * 1e6, SAMPLE_RATE)
            result.extend(chunk.tolist())
        self._freq_list = sorted(set(result))
