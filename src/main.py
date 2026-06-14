"""
main.py — точка входу SDR-монітора.
MainDispatcher координує всі підсистеми.
"""

import sys
import time
from PyQt6.QtWidgets import QApplication
from services.pi_server_service import PiServerService

from hardware.radio_sensor import RadioSensor
from hardware.antenna_controller import AntennaController
from scanning.scan_manager import ScanManager
from dsp.dsp_engine import DSPEngine
from gui.main_window import SDRInterface
from analysis.histogram_accumulator import HistogramAccumulator
from core.config import GAIN_DEFAULT


class MainDispatcher:
    """
    Головний оркестратор системи.
    Ініціалізує всі підсистеми та реалізує ігровий цикл через update_tick().
    """

    HOP_INTERVAL: float = 0.2  # Швидкість сканування (с між стрибками)

    def __init__(self):
        print("[System] Initializing modules...")

        # — Підсистеми —
        self.radio = RadioSensor(sim_antenna_mux=True)  # BLANK-провали активні в симуляції
        self.antennas = AntennaController()
        self.scanner = ScanManager()
        self.dsp = DSPEngine()
        self.histo = HistogramAccumulator()

        # — Початкове налаштування —
        self.radio.set_gain(GAIN_DEFAULT)
        self.radio.tune(433_000_000)

        # — Стан сканування —
        self._is_scanning: bool = False
        self._last_hop_time: float = 0.0

        # — GUI —
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")
        self.pi_server = PiServerService()
        # Встановлюємо вузький початковий діапазон ДО старту воркера —
        # так він ніколи не бачить широкий SCAN_RANGES (868/1200/5800 МГц)
        self.pi_server._pending_rf_range = (430.0, 440.0)
        self.pi_server.start()
        self.ui = SDRInterface(ctrl=self)
        self.ui.show()

        # Підключення DSP → GUI
        self.dsp.threat_detected.connect(self.ui.log)
        # Підключення DSP → PiServerService (передача колезі)
        self.dsp.threat_detected.connect(
            lambda msg, color: self.pi_server.rf_threat_received.emit(msg, color)
        )
        # Підключення реального азимуту DSP → PiServerService
        self.dsp.bearing_updated.connect(
            lambda bearing, conf, unc, method:
                self.pi_server.bearing_updated.emit(bearing, conf, unc, method)
        )
        # Вмикаємо пеленгацію за замовчуванням
        self.dsp.direction_enabled = True
        self.ui.log("System Ready.", "lime")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_scanning(self) -> bool:
        return self._is_scanning

    # ── Mask ──────────────────────────────────────────────────────────────

    def set_mask(self) -> None:
        self.dsp.set_mask_flag()
        self.ui.log("Mask SET", "cyan")

    def clear_mask(self) -> None:
        self.dsp.clear_mask()
        self.ui.log("Mask CLEARED", "yellow")

    # ── Histogram ─────────────────────────────────────────────────────────

    def reset_histogram(self) -> None:
        self.histo.clear()
        self.ui.log("Histogram reset.", "gray")

    def open_histogram(self) -> None:
        self.ui.open_histogram()

    # ── Frequency / Gain ──────────────────────────────────────────────────

    def set_freq(self, freq_hz: float) -> None:
        self.stop_scan()
        self.radio.tune(int(freq_hz))
        if hasattr(self, "ui"):
            self.ui.block_freq_update = True
            self.ui.f_spin.setValue(freq_hz / 1e6)
            self.ui.block_freq_update = False
        # Синхронізуємо PiProxy воркер — вузьке вікно навколо поточної частоти
        if (self.pi_server.rf_worker
                and self.pi_server.rf_worker.is_alive()):
            freq_mhz = freq_hz / 1e6
            self.pi_server.rf_worker.set_range(freq_mhz - 14.0, freq_mhz + 14.0)

    def set_gain(self, gain: int) -> None:
        self.radio.set_gain(gain)

    # ── Scan range ────────────────────────────────────────────────────────

    def set_scan_range(self, start: float, stop: float) -> None:
        self.scanner.set_custom_range(start, stop)
        self.ui.log(f"Range set: {start}-{stop} MHz", "cyan")
        self.reset_histogram()
        # Синхронізуємо PiProxy worker (і pending range якщо ще не запущений)
        if (self.pi_server.rf_worker
                and self.pi_server.rf_worker.is_alive()):
            self.pi_server.rf_worker.set_range(start, stop)
        else:
            self.pi_server._pending_rf_range = (start, stop)

    # ── Scan toggle ───────────────────────────────────────────────────────

    def toggle_scan(self) -> None:
        self._is_scanning = not self._is_scanning
        if self._is_scanning:
            self.ui.btn_scan.setText("STOP SCAN")
            self.ui.btn_scan.setStyleSheet(
                "background: #bd2c00; color: white; font-weight: bold;"
            )
            self.ui.log("Scanning started...", "yellow")

            # Синхронізуємо PiProxy воркер з ПОТОЧНИМ діапазоном GUI
            # (на випадок якщо user натиснув START SCAN без зміни полів)
            freqs = self.scanner.freq_list
            if freqs and self.pi_server.rf_worker and self.pi_server.rf_worker.is_alive():
                start_mhz = float(min(freqs)) / 1e6
                stop_mhz  = float(max(freqs)) / 1e6 + 0.001  # невеликий відступ
                self.pi_server.rf_worker.set_range(start_mhz, stop_mhz)
        else:
            self.stop_scan()

    def stop_scan(self) -> None:
        self._is_scanning = False
        self.ui.btn_scan.setText(" START SCAN")
        self.ui.btn_scan.setStyleSheet(
            "background: #2ea043; color: white; font-weight: bold;"
        )

    # ── Hardware ──────────────────────────────────────────────────────────

    def connect_pluto(self) -> None:
        self.ui.log("Connecting to Pluto...", "yellow")
        status = self.radio.connect()
        if "Connected" in status:
            self.ui.log(status, "lime")
            self.ui.setWindowTitle("SDR Monitor [Mode: HARDWARE]")
        else:
            self.ui.log(f"Error: {status}", "red")

    # ── Main loop tick ────────────────────────────────────────────────────

    def update_tick(self) -> None:
        # 1. Сканування — стрибок на нову частоту
        if self._is_scanning:
            if time.time() - self._last_hop_time > self.HOP_INTERVAL:
                next_freq = self.scanner.get_next_frequency()
                self.radio.tune(int(next_freq))
                self._last_hop_time = time.time()
                self.ui.block_freq_update = True
                self.ui.f_spin.setValue(next_freq / 1e6)
                self.ui.block_freq_update = False

        # 2. Отримання IQ-даних
        iq_data = self.radio.get_samples()
        center_f = self.radio.current_freq

        # 3. DSP-обробка
        freqs, psd = self.dsp.process(iq_data, center_f)
        if freqs is None:
            return

        # 4. Гістограма
        self.histo.update(freqs, psd, self.dsp.mask_enabled, self.dsp.channel_activity)

        # 5. Графіки
        self.ui.update_graphs(freqs, psd)

        # 6. Вікно гістограми
        self.ui.update_histogram_window()

    def run(self) -> None:
        sys.exit(self.app.exec())


if __name__ == "__main__":
    system = MainDispatcher()
    system.run()
