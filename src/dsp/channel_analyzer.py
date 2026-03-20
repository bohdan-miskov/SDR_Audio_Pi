"""
ChannelAnalyzer — аналіз активності WiFi-каналів.
"""
import numpy as np
from core.config import get_band_profile, WIFI_CHANNEL_WIDTH


class ChannelAnalyzer:
    """
    Визначає активні WiFi-канали в поточному вікні спектра.
    Результат зберігається у channel_activity.
    """

    def __init__(self):
        self._channel_activity: dict[int, float] = {}
        self._current_profile: dict | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property 
    def channel_activity(self) -> dict[int, float]:
        """Словник {номер_каналу: потужність_дБ} для активних каналів."""
        return self._channel_activity

    @property
    def current_profile(self) -> dict | None:
        """Поточний профіль діапазону (wifi_2_4 / wifi_5 / None)."""
        return self._current_profile

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, freqs: np.ndarray, psd: np.ndarray,
                center_freq: float, mask_enabled: bool) -> None:
        """
        Знаходить активні канали у видимому діапазоні та оновлює channel_activity.
        """
        profile = get_band_profile(center_freq)
        self._current_profile = profile
        self._channel_activity = {}

        if not profile or not profile.get('channels'):
            return

        channel_width = profile.get('channel_width', WIFI_CHANNEL_WIDTH)

        if mask_enabled:
            threshold = 3.0
        else:
            threshold = float(np.percentile(psd, 20)) + 5.0

        for ch_num, ch_freq in profile['channels'].items():
            if freqs[0] <= ch_freq <= freqs[-1]:
                mask = np.abs(freqs - ch_freq) < channel_width / 2
                if np.any(mask):
                    power = float(np.max(psd[mask]))
                    if power > threshold:
                        self._channel_activity[ch_num] = power
