"""
HistogramWindow — окреме вікно для відображення гістограми активності.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout
import pyqtgraph as pg


class HistogramWindow(QDialog):
    """Плаваюче вікно з Bar-графіком активності частот або WiFi-каналів."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signal Histogram & Bandwidth")
        self.resize(800, 400)

        self._channel_mode: bool = False

        layout = QVBoxLayout(self)

        self._plot = pg.PlotWidget(title="Frequency Activity")
        self._plot.setLabel('bottom', 'Frequency', units='MHz')
        self._plot.setLabel('left', 'Signal Level')
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot)

        self._bar_item = pg.BarGraphItem(x=[], height=[], width=16, brush='c')
        self._plot.addItem(self._bar_item)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def channel_mode(self) -> bool:
        return self._channel_mode

    # ── Public API ──────────────────────────────────────────────────────────

    def update_data(self, x: list, y: list,
                    bar_width: float = 16.0, channel_mode: bool = False) -> None:
        """Оновлює дані гістограми."""
        if not x:
            return

        if channel_mode != self._channel_mode:
            self._channel_mode = channel_mode
            if channel_mode:
                self._plot.setTitle("WiFi Channel Activity")
                self._plot.setLabel('bottom', 'Channel Number')
            else:
                self._plot.setTitle("Frequency Activity")
                self._plot.setLabel('bottom', 'Frequency', units='MHz')

        self._bar_item.setOpts(x=x, height=y, width=bar_width)
