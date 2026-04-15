"""
RadioSensor — SDR-приймач (PlutoSDR або симуляція).
Реалізує IRadio через @property та явне управління станом.
"""
import numpy as np
from core.config import SAMPLE_RATE, BUFFER_SIZE, GAIN_DEFAULT
from hardware.simulation_generator import SimulationGenerator

try:
    import adi
    HAS_ADI = True
except ImportError:
    HAS_ADI = False


class RadioSensor:
    """
    Обгортка над PlutoSDR (libiio/adi).
    При відсутності бібліотеки або апаратного забезпечення
    автоматично переходить в режим симуляції.
    """

    def __init__(self, uri: str = "ip:192.168.2.1"):
        self._uri = uri
        self._sdr = None
        self._sim = SimulationGenerator()
        self._mode = "INIT"
        self._current_freq: int = 433_000_000
        self._current_gain: int = GAIN_DEFAULT

        # Спроба підключення при старті
        self.connect()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def current_freq(self) -> int:
        return self._current_freq

    @current_freq.setter
    def current_freq(self, freq_hz: int) -> None:
        self._current_freq = int(freq_hz)

    @property
    def current_gain(self) -> int:
        return self._current_gain

    @current_gain.setter
    def current_gain(self, gain_db: int) -> None:
        self._current_gain = int(gain_db)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def uri(self) -> str:
        return self._uri

    # ── IRadio interface ─────────────────────────────────────────────────────

    def connect(self) -> str:
        """Спроба підключення до залізного SDR. Повертає рядок статусу."""
        if not HAS_ADI:
            print("[Hardware] Library 'adi' not found. Using Simulation.")
            self._mode = "SIMULATION"
            return "No Lib"

        print(f"[Hardware] Connecting to PlutoSDR at {self._uri}...")
        try:
            if self._sdr:
                del self._sdr

            sdr = adi.Pluto(self._uri)
            sdr.sample_rate = int(SAMPLE_RATE)
            sdr.rx_buffer_size = int(BUFFER_SIZE)
            sdr.rx_lo = int(self._current_freq)
            sdr.rx_enabled_channels = [0]

            try:
                sdr.rx_hardwaregain_chan0 = int(self._current_gain)
            except Exception:
                sdr.gain_control_mode_chan0 = "manual"
                sdr.rx_hardwaregain_chan0 = int(self._current_gain)

            self._sdr = sdr
            self._mode = "HARDWARE"
            msg = f"Connected! SR: {sdr.sample_rate / 1e6} MSPS"
            print(f"[Hardware] {msg}")
            return msg

        except Exception as exc:
            print(f"[Hardware Error] Connection failed: {exc}")
            self._sdr = None
            self._mode = "SIMULATION"
            return str(exc)

    def tune(self, freq_hz: int) -> None:
        """Налаштування на нову частоту (Гц)."""
        self._current_freq = int(freq_hz)
        if self._mode == "HARDWARE" and self._sdr:
            try:
                self._sdr.rx_lo = int(freq_hz)
            except Exception:
                pass

    def set_gain(self, gain_db: int) -> None:
        """Встановлення посилення (дБ)."""
        self._current_gain = int(gain_db)
        if self._mode == "HARDWARE" and self._sdr:
            try:
                self._sdr.rx_hardwaregain_chan0 = int(gain_db)
            except Exception:
                pass

    def get_samples(self) -> np.ndarray:
        """Повертає масив IQ-семплів (complex64)."""
        if self._mode == "HARDWARE" and self._sdr:
            try:
                iq = self._sdr.rx()
                if iq.dtype != np.complex64:
                    iq = iq.astype(np.complex64)
                if iq.ndim == 2 and iq.shape[0] == 2:
                    iq = iq[0] + 1j * iq[1]
                iq /= 2048.0
                return iq
            except Exception as exc:
                print(f"[RX Error] {exc}")
                return np.zeros(BUFFER_SIZE, dtype=np.complex64)
        else:
            return self._sim.get_iq_samples(self._current_freq, self._current_gain)
