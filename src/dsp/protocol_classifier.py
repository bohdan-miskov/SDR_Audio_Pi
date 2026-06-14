"""
ProtocolClassifier — визначення типу протоколу за шириною та потужністю сигналу.
"""
import numpy as np
from core.config import get_frequency_band, PROTOCOL_SIGNATURES


class ProtocolClassifier:
    """
    Класифікує протокол за шириною смуги та частотою.
    Результати зберігаються у detected_protocol, detected_bandwidth, detected_power.
    """

    def __init__(self):
        self._detected_protocol: str | None = None
        self._detected_bandwidth: float = 0.0
        self._detected_power: float = 0.0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def detected_protocol(self) -> str | None:
        """Ключ з PROTOCOL_SIGNATURES або 'NARROWBAND'/'WIDEBAND'/'ULTRA_WIDE'."""
        return self._detected_protocol

    @property
    def detected_bandwidth(self) -> float:
        """Виміряна ширина смуги сигналу (Гц)."""
        return self._detected_bandwidth

    @property
    def detected_power(self) -> float:
        """Пікова потужність сигналу (дБ у масштабі PSD)."""
        return self._detected_power

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, freqs: np.ndarray, psd: np.ndarray,
                 center_freq: float, mask_enabled: bool) -> None:
        """
        Визначає протокол для поточного вікна спектра.
        Оновлює detected_protocol, detected_bandwidth, detected_power.
        """
        self._detected_protocol = None
        self._detected_bandwidth = 0.0
        self._detected_power = 0.0

        band = get_frequency_band(center_freq)
        if not band:
            return

        bandwidth, peak_power = self._measure_bandwidth(freqs, psd)

        min_power = 5.0 if mask_enabled else float(np.percentile(psd, 30)) + 10.0
        if peak_power < min_power:
            return

        self._detected_bandwidth = bandwidth
        self._detected_power = peak_power

        # Пошук по сигнатурах
        for proto_key, sig in PROTOCOL_SIGNATURES.items():
            if band in sig['bands'] and sig['bw_min'] <= bandwidth <= sig['bw_max']:
                self._detected_protocol = proto_key
                return

        # Загальна класифікація за шириною
        if bandwidth > 0:
            if bandwidth < 2e6:
                self._detected_protocol = 'NARROWBAND'
            elif bandwidth < 8e6:
                self._detected_protocol = 'WIDEBAND'
            else:
                self._detected_protocol = 'ULTRA_WIDE'

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _measure_bandwidth(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
        """
        Вимірює ширину сигналу за критерієм -6 дБ від піку.

        Returns:
            (bandwidth_hz, peak_power_db).
        """
        if len(psd) == 0:
            return 0.0, 0.0

        peak_power = float(np.max(psd))
        threshold = peak_power - 6.0

        above = psd > threshold
        if not np.any(above):
            return 0.0, peak_power

        indices = np.where(above)[0]
        if len(indices) < 2:
            return 0.0, peak_power

        bandwidth = float(freqs[indices[-1]] - freqs[indices[0]])
        return bandwidth, peak_power
