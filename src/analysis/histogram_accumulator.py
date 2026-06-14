"""
HistogramAccumulator — накопичення та відображення рівнів активності спектра.
"""
import numpy as np


class HistogramAccumulator:
    """
    Накопичує максимальні рівні сигналу по смугах частот або WiFi-каналах.
    Підтримує режим затухання: значення зберігаються hold_frames кадрів.
    """

    DEFAULT_BAND_WIDTH_MHZ: float = 20.0
    DEFAULT_HOLD_FRAMES: int = 50
    CHANNEL_BAR_WIDTH: float = 0.8
    DECAY_FACTOR: float = 0.95

    def __init__(self,
                 band_width_mhz: float = DEFAULT_BAND_WIDTH_MHZ,
                 hold_frames: int = DEFAULT_HOLD_FRAMES):
        self._bins: dict = {}
        self._channel_labels: dict = {}
        self._channel_mode: bool = False
        self._band_width_mhz = band_width_mhz
        self._hold_frames = hold_frames

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def channel_mode(self) -> bool:
        """True — активний режим WiFi-каналів."""
        return self._channel_mode

    @property
    def band_width_mhz(self) -> float:
        return self._band_width_mhz

    @band_width_mhz.setter
    def band_width_mhz(self, value: float) -> None:
        self._band_width_mhz = float(value)

    @property
    def hold_frames(self) -> int:
        return self._hold_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Скидає накопичені дані."""
        self._bins = {}
        self._channel_labels = {}

    def update(self, freqs: np.ndarray, psd: np.ndarray,
               is_mask_active: bool, channel_activity: dict | None = None) -> None:
        """
        Оновлює гістограму новими даними.

        Args:
            freqs:            Масив частот (Гц).
            psd:              Масив потужностей (дБ).
            is_mask_active:   True — маска увімкнена.
            channel_activity: Словник {канал: потужність} від ChannelAnalyzer.
        """
        # Затухання лічильників
        for key in list(self._bins.keys()):
            self._bins[key]['counter'] -= 1
            if self._bins[key]['counter'] <= 0:
                del self._bins[key]
                self._channel_labels.pop(key, None)

        if channel_activity and len(channel_activity) > 0:
            self._channel_mode = True
            self._update_channel_mode(channel_activity)
        else:
            self._channel_mode = False
            self._update_band_mode(freqs, psd, is_mask_active)

    def get_data(self) -> tuple[list, list, bool]:
        """
        Повертає (keys, values, channel_mode) для відображення.
        """
        if not self._bins:
            return [], [], self._channel_mode
        sorted_keys = sorted(self._bins.keys())
        values = [self._bins[k]['value'] for k in sorted_keys]
        return sorted_keys, values, self._channel_mode

    def get_band_width(self) -> float:
        """Ширина стовпчика для BarGraphItem."""
        if self._channel_mode:
            return self.CHANNEL_BAR_WIDTH
        return self._band_width_mhz * 0.8

    # ── Private ───────────────────────────────────────────────────────────────

    def _update_channel_mode(self, channel_activity: dict) -> None:
        for ch_num, power in channel_activity.items():
            new_value = max(0.0, power + 10.0)
            if ch_num not in self._bins:
                self._bins[ch_num] = {'value': new_value, 'counter': self._hold_frames}
                self._channel_labels[ch_num] = ch_num
            else:
                self._bins[ch_num]['value'] = max(
                    self._bins[ch_num]['value'] * self.DECAY_FACTOR, new_value
                )
                self._bins[ch_num]['counter'] = self._hold_frames

    def _update_band_mode(self, freqs: np.ndarray, psd: np.ndarray,
                           is_mask_active: bool) -> None:
        threshold = 5.0 if is_mask_active else -50.0
        freqs_mhz = freqs / 1e6
        band_starts = np.floor(freqs_mhz / self._band_width_mhz) * self._band_width_mhz

        for band_start in np.unique(band_starts):
            mask = band_starts == band_start
            valid_psd = psd[mask]
            valid_psd = valid_psd[valid_psd > threshold]

            if len(valid_psd) == 0:
                continue

            sorted_vals = np.sort(valid_psd)[::-1]
            top_count = max(1, len(sorted_vals) // 10)
            avg_max = float(np.mean(sorted_vals[:top_count]))
            new_value = max(0.0, avg_max + 60.0)
            band_center = float(band_start + self._band_width_mhz / 2)

            if band_center not in self._bins:
                self._bins[band_center] = {'value': new_value, 'counter': self._hold_frames}
            else:
                self._bins[band_center]['value'] = max(
                    self._bins[band_center]['value'] * self.DECAY_FACTOR, new_value
                )
                self._bins[band_center]['counter'] = self._hold_frames
