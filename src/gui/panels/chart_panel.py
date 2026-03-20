"""
ChartPanel — правий блок з графіками: Spectrum Analyzer + Waterfall.
"""
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.QtCore import QRectF, Qt
import pyqtgraph as pg

from core.config import get_band_profile

DISPLAY_WIDTH = 1024


class ChartPanel(QWidget):
    """
    Містить два графіки:
    - plot_spectrum: лінійний спектр (FFT з каналами)
    - plot_waterfall: водоспад (кольорове зображення)
    """

    WF_HEIGHT: int = 150

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl
        self._wf_data = np.full((self.WF_HEIGHT, DISPLAY_WIDTH), -100.0)
        self._channel_markers: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # — Spectrum plot —
        self._plot_spectrum = pg.PlotWidget(title="Spectrum Analyzer")
        self._plot_spectrum.showGrid(True, True, 0.2)
        self._plot_spectrum.setMouseEnabled(x=True, y=True)
        self._plot_spectrum.setYRange(-130, 50)
        self._curve = self._plot_spectrum.plot(pen=pg.mkPen('#FFD700', width=1))
        self._plot_spectrum.scene().sigMouseClicked.connect(self._on_click)
        layout.addWidget(self._plot_spectrum, 1)

        # — Waterfall plot —
        self._plot_waterfall = pg.PlotWidget(title="Waterfall")
        self._img = pg.ImageItem()
        self._img.setLookupTable(pg.colormap.get('inferno').getLookupTable())
        self._plot_waterfall.addItem(self._img)
        layout.addWidget(self._plot_waterfall, 1)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def plot_spectrum(self) -> pg.PlotWidget:
        return self._plot_spectrum

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, freqs: np.ndarray, psd: np.ndarray, mask_enabled: bool) -> None:
        """
        Оновлює обидва графіки.
        """
        # Downsample якщо даних більше ніж DISPLAY_WIDTH
        if len(psd) > DISPLAY_WIDTH:
            indices = np.linspace(0, len(psd) - 1, DISPLAY_WIDTH)
            psd_small = np.interp(indices, np.arange(len(psd)), psd)
            freqs_small = np.interp(indices, np.arange(len(freqs)), freqs)
        else:
            psd_small = psd
            freqs_small = freqs

        freqs_m = freqs_small / 1e6
        self._curve.setData(freqs_m, psd_small)

        # Waterfall
        if self._wf_data.shape[1] != len(psd_small):
            self._wf_data = np.full((self.WF_HEIGHT, len(psd_small)), -100.0)

        self._wf_data = np.roll(self._wf_data, 1, axis=0)
        self._wf_data[0] = psd_small
        self._img.setImage(self._wf_data.T, autoLevels=False)
        self._img.setLevels([0, 20] if mask_enabled else [-100, -50])
        self._img.setRect(QRectF(freqs_m[0], 0, freqs_m[-1] - freqs_m[0], self.WF_HEIGHT))

        self._update_channel_markers(freqs)

    # ── Private ───────────────────────────────────────────────────────────────

    def _update_channel_markers(self, freqs: np.ndarray) -> None:
        """Малює маркери WiFi-каналів на спектрограмі."""
        for marker in self._channel_markers:
            self._plot_spectrum.removeItem(marker)
        self._channel_markers.clear()

        if len(freqs) == 0:
            return

        center_freq = (freqs[0] + freqs[-1]) / 2
        profile = get_band_profile(center_freq)

        if not profile or not profile.get('channels'):
            return

        for ch_num, ch_freq in profile['channels'].items():
            if freqs[0] <= ch_freq <= freqs[-1]:
                freq_mhz = ch_freq / 1e6

                line = pg.InfiniteLine(
                    pos=freq_mhz, angle=90,
                    pen=pg.mkPen('#444', width=1, style=Qt.DashLine)
                )
                text = pg.TextItem(f"{ch_num}", color='#888', anchor=(0.5, 0))
                text.setPos(freq_mhz, 35)

                self._plot_spectrum.addItem(line)
                self._plot_spectrum.addItem(text)
                self._channel_markers.extend([line, text])

    def _on_click(self, e) -> None:
        """Клік на спектрограмі → налаштування частоти."""
        if self._plot_spectrum.plotItem in self._plot_spectrum.scene().items(e.scenePos()):
            mp = self._plot_spectrum.plotItem.vb.mapSceneToView(e.scenePos())
            self._ctrl.set_freq(mp.x() * 1e6)
