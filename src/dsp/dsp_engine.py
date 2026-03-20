"""
DSPEngine — оркестратор цифрової обробки сигналів.
Реалізує IProcessor та координує FFTProcessor, ChannelAnalyzer,
ProtocolClassifier, ThreatDetector.
"""
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from dsp.fft_processor import FFTProcessor
from dsp.channel_analyzer import ChannelAnalyzer
from dsp.protocol_classifier import ProtocolClassifier
from dsp.threat_detector import ThreatDetector


class DSPEngine(QObject):
    """
    Головний DSP-клас. Пов'язує всі підмодулі обробки сигналів
    та публікує Qt-сигнал threat_detected.
    """

    # Qt-сигнал для GUI: (повідомлення, колір)
    threat_detected = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self._fft = FFTProcessor()
        self._channels = ChannelAnalyzer()
        self._classifier = ProtocolClassifier()
        self._detector = ThreatDetector(
            alert_callback=lambda msg, color: self.threat_detected.emit(msg, color)
        )

    # ── IProcessor properties ─────────────────────────────────────────────────

    @property
    def mask_enabled(self) -> bool:
        return self._fft.mask_enabled

    # ── Публічні проксі-властивості (для зворотної сумісності з GUI) ──────────

    @property
    def channel_activity(self) -> dict[int, float]:
        return self._channels.channel_activity

    @property
    def detected_protocol(self) -> str | None:
        return self._classifier.detected_protocol

    @property
    def detected_bandwidth(self) -> float:
        return self._classifier.detected_bandwidth

    @property
    def detected_power(self) -> float:
        return self._classifier.detected_power

    # ── IProcessor interface ──────────────────────────────────────────────────

    def set_mask_flag(self) -> None:
        self._fft.enable_mask()

    def clear_mask(self) -> None:
        self._fft.clear_mask()

    def process(self, iq_samples: np.ndarray, center_freq: float) -> tuple:
        """
        Повний конвеєр обробки одного буфера IQ.

        Returns:
            (freqs, psd) або (None, None) при порожньому вході.
        """
        if len(iq_samples) == 0:
            return None, None

        # 1. FFT → спектр
        freqs, psd = self._fft.process(iq_samples, center_freq)

        # 2. Аналіз WiFi-каналів
        self._channels.analyze(freqs, psd, center_freq, self._fft.mask_enabled)

        # 3. Класифікація протоколу
        self._classifier.classify(freqs, psd, center_freq, self._fft.mask_enabled)

        # 4. Детекція загроз
        noise_floor = 0.0 if self._fft.mask_enabled else float(np.percentile(psd, 30))
        self._detector.analyze(
            freqs=freqs,
            psd=psd,
            noise_floor=noise_floor,
            center_freq=center_freq,
            mask_enabled=self._fft.mask_enabled,
            current_profile=self._channels.current_profile,
            channel_activity=self._channels.channel_activity,
            detected_protocol=self._classifier.detected_protocol,
            detected_bandwidth=self._classifier.detected_bandwidth,
            detected_power=self._classifier.detected_power,
        )

        return freqs, psd
