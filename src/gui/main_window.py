import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtCore import Qt


class RadarWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Drone Detection Radar")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #050505; color: #00FF00;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.status_label = QLabel("СТАТУС: МОНІТОРИНГ")
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.status_label)

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('#050505')
        self.graph_widget.setXRange(0, 2000)
        self.graph_widget.setYRange(0, 1.0)
        self.layout.addWidget(self.graph_widget)

        self.curve = self.graph_widget.plot(
            pen=pg.mkPen('#00FF00', width=1.5),
            skipFiniteCheck=True,
            autoDownsample=True
        )

        self.goertzel_lines = {}
        for freq in self.config['detection']['goertzel_frequencies']:
            v_line = pg.InfiniteLine(pos=freq, angle=90, pen=pg.mkPen('#333333', width=1))
            self.graph_widget.addItem(v_line)
            self.goertzel_lines[freq] = v_line

        self.rec_button = QPushButton("ЗАПИС (REC)")
        self.rec_button.setStyleSheet(
            "background-color: #111; color: #FF3333; font-weight: bold; padding: 15px; border: 1px solid #333;")
        self.layout.addWidget(self.rec_button)

        self.chunk_size = self.config['device']['chunk_size']
        self.rate = self.config['device']['rate']
        self.window_func = np.hanning(self.chunk_size)
        self.freq_axis = np.fft.rfftfreq(self.chunk_size, 1 / self.rate)

    def update_plot(self, audio_data):
        if len(audio_data) != self.chunk_size:
            return

        windowed = audio_data * self.window_func
        spectrum = np.abs(np.fft.rfft(windowed))

        spectrum += 1e-6
        np.log10(spectrum, out=spectrum)
        spectrum *= 20.0

        spectrum += 60.0
        spectrum /= 60.0

        self.curve.setData(self.freq_axis, spectrum)

    def update_goertzel_status(self, freq_results):
        for freq, snr in freq_results.items():
            line = self.goertzel_lines.get(freq)
            if line:
                if snr > 8.0:
                    line.setPen(pg.mkPen('#FF0000', width=3))
                elif snr > 3.0:
                    line.setPen(pg.mkPen('#FFFF00', width=2))
                else:
                    line.setPen(pg.mkPen('#333333', width=1))

    def update_detection(self, is_drone, confidence):
        if is_drone:
            self.status_label.setText(f"ВИЯВЛЕНО ДРОН ({confidence:.1f}%)")
            self.status_label.setStyleSheet("color: #FF3333; font-size: 22px; font-weight: bold; background: #200;")
        else:
            self.status_label.setText("СТАТУС: МОНІТОРИНГ")
            self.status_label.setStyleSheet(
                "color: #00FF00; font-size: 20px; font-weight: bold; background: transparent;")