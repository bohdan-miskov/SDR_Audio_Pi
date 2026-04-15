"""
DSPEngine — оркестратор цифрової обробки сигналів.
Координує FFTProcessor, ChannelAnalyzer, ProtocolClassifier, ThreatDetector
та опційний AmplitudeSyncDetector (Amplitude Blanking).
"""
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from dsp.fft_processor import FFTProcessor
from dsp.channel_analyzer import ChannelAnalyzer
from dsp.protocol_classifier import ProtocolClassifier
from dsp.threat_detector import ThreatDetector
from dsp.amplitude_sync import AmplitudeSyncDetector
from dsp.direction_finder import DirectionFinder


class DSPEngine(QObject):
    """
    Головний DSP-клас. Пов'язує всі підмодулі обробки сигналів
    та публікує Qt-сигнал threat_detected.

    Amplitude Blanking:
        При sync_enabled=True перед основним конвеєром запускається
        AmplitudeSyncDetector, який вирівнює IQ-буфер по антенному циклу.
        Синхронізовані дані кожної антени зберігаються в antenna_iq.
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
        self._sync = AmplitudeSyncDetector()
        self._finder = DirectionFinder()

        # Результати останньої синхронізації
        self._antenna_iq: dict[str, np.ndarray] = {}
        self._sync_enabled: bool = False
        self._direction_enabled: bool = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def mask_enabled(self) -> bool:
        return self._fft.mask_enabled

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

    @property
    def sync_enabled(self) -> bool:
        """True — Amplitude Blanking синхронізація активна."""
        return self._sync_enabled

    @sync_enabled.setter
    def sync_enabled(self, value: bool) -> None:
        self._sync_enabled = bool(value)
        print(f"[DSP] AmplitudeSyncDetector: {'ON' if value else 'OFF'}")

    @property
    def direction_enabled(self) -> bool:
        """True — пеленгація активна (sync_enabled повинен бути True)."""
        return self._direction_enabled

    @direction_enabled.setter
    def direction_enabled(self, value: bool) -> None:
        self._direction_enabled = bool(value)
        if value and not self._sync_enabled:
            self._sync_enabled = True
            print("[DSP] sync_enabled auto-ON (needed for direction_enabled)")
        print(f"[DSP] DirectionFinder: {'ON' if value else 'OFF'}")

    @property
    def bearing(self) -> float:
        """Останній розрахований азимут (°), 0..360."""
        return self._finder.bearing

    @property
    def bearing_confidence(self) -> float:
        """Впевненість пеленгу 0..1."""
        return self._finder.confidence

    @property
    def bearing_pattern(self) -> np.ndarray:
        """360-точковий просторовий спектр потужності."""
        return self._finder.power_pattern

    @property
    def sync_markers(self) -> list[int]:
        """Позиції BLANK-маркерів у останньому буфері."""
        return self._sync.last_markers

    @property
    def antenna_iq(self) -> dict[str, np.ndarray]:
        """
        Синхронізовані IQ-дані по антенах після останнього process().
        {'ANT_A': ndarray, 'ANT_B': ndarray, 'ANT_C': ndarray}
        Порожній dict, якщо sync_enabled=False або синхронізація не вдалась.
        """
        return self._antenna_iq

    # ── IProcessor interface ──────────────────────────────────────────────────

    def set_mask_flag(self) -> None:
        self._fft.enable_mask()

    def clear_mask(self) -> None:
        self._fft.clear_mask()

    def process(self, iq_samples: np.ndarray, center_freq: float) -> tuple:
        """
        Повний конвеєр обробки одного буфера IQ.

        Якщо sync_enabled=True:
            1. AmplitudeSyncDetector → antenna_iq (нарізка по антенах)
            2. Основний конвеєр (FFT, аналіз) використовує весь iq_samples

        Returns:
            (freqs, psd) або (None, None) при порожньому вході.
        """
        if len(iq_samples) == 0:
            return None, None

        # 0. Amplitude Blanking синхронізація (опційно)
        if self._sync_enabled:
            result = self._sync.process(iq_samples)
            self._antenna_iq = result if result is not None else {}
            if result is None:
                print(f"[DSP] Sync: no markers (quality={self._sync.sync_quality:.2f})")

            # 0b. Пеленгація кута (опційно)
            if self._direction_enabled and self._antenna_iq:
                bearing = self._finder.estimate(self._antenna_iq)
                if bearing is not None:
                    conf = self._finder.confidence
                    if conf > 0.3:  # повідомляємо тільки при достатньо впевненому результаті
                        color = "#00ff88" if conf > 0.6 else "#ffcc00"
                        self.threat_detected.emit(
                            f"BEARING: {bearing:.1f}deg  conf={conf:.2f}", color
                        )
        else:
            self._antenna_iq = {}

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
