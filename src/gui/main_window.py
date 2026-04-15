"""
SDRInterface — головне вікно SDR-монітора.
Координує ControlPanel та ChartPanel, підписується на таймер оновлення.
"""
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer

from gui.panels.control_panel import ControlPanel
from gui.panels.chart_panel import ChartPanel
from gui.windows.histogram_window import HistogramWindow

import numpy as np


class SDRInterface(QMainWindow):
    """
    Головне вікно програми.
    Делегує всю логіку контролеру (ctrl), залишаючи собі лише View.
    """

    TIMER_MS: int = 50  # Інтервал оновлення графіків

    def __init__(self, ctrl):
        super().__init__()
        self._ctrl = ctrl
        self._hist_window: HistogramWindow | None = None

        self.setWindowTitle(f"SDR Monitor [Mode: {self._ctrl.radio.mode}]")
        self.resize(1200, 800)

        # Центральний віджет
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Панелі
        self._ctrl_panel = ControlPanel(ctrl=self._ctrl, parent=self)
        self._chart_panel = ChartPanel(ctrl=self._ctrl, parent=self)

        layout.addWidget(self._ctrl_panel)
        layout.addWidget(self._chart_panel)

        # Таймер
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._ctrl.update_tick)
        self._timer.start(self.TIMER_MS)

    # ── Proxy properties (для зворотної сумісності з MainDispatcher) ──────

    @property
    def block_freq_update(self) -> bool:
        return self._ctrl_panel.block_freq_update

    @block_freq_update.setter
    def block_freq_update(self, value: bool) -> None:
        self._ctrl_panel.block_freq_update = value

    @property
    def f_spin(self):
        """Прямий доступ до спінбоксу частоти (використовується в MainDispatcher)."""
        return self._ctrl_panel.f_spin

    @property
    def btn_scan(self):
        """Прямий доступ до кнопки SCAN (використовується в MainDispatcher)."""
        return self._ctrl_panel.btn_scan

    # ── Public API ────────────────────────────────────────────────────────

    def log(self, text: str, color: str = "white") -> None:
        """Додає рядок у консольний лог."""
        self._ctrl_panel.log(text, color)

    def open_histogram(self) -> None:
        """Відкриває вікно гістограми (якщо ще не відкрито)."""
        if self._hist_window is None:
            self._hist_window = HistogramWindow(self)
        self._hist_window.show()

    def update_graphs(self, freqs: np.ndarray, psd: np.ndarray) -> None:
        """Оновлює спектр та водоспад, синхронізує спінбокс."""
        self._chart_panel.update(freqs, psd, self._ctrl.dsp.mask_enabled)

        # Синхронізуємо відображення поточної частоти
        if not self.f_spin.hasFocus() and not self.block_freq_update:
            current_mhz = self._ctrl.radio.current_freq / 1e6
            if abs(self.f_spin.value() - current_mhz) > 0.001:
                self.f_spin.blockSignals(True)
                self.f_spin.setValue(current_mhz)
                self.f_spin.blockSignals(False)

    def update_histogram_window(self) -> None:
        """Оновлює гістограму (якщо вікно відкрите)."""
        if self._hist_window is None or not self._hist_window.isVisible():
            return
        keys, values, channel_mode = self._ctrl.histo.get_data()
        bar_width = self._ctrl.histo.get_band_width()
        self._hist_window.update_data(keys, values, bar_width, channel_mode)
