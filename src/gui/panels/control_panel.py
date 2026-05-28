"""
ControlPanel — ліва панель управління SDR-інтерфейсу.
Містить: SYSTEM, SCANNER RANGE, HISTOGRAM, NOISE MASK, FREQ, GAIN, LOG.
"""
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QDoubleSpinBox,
    QSlider, QPushButton, QTextEdit, QGridLayout, QGroupBox,
)
from PyQt5.QtCore import Qt
import time


class ControlPanel(QFrame):
    """
    Ліва панель інтерфейсу SDR-монітора.
    Зберігає посилання на головний контролер (ctrl) для делегування дій.
    """

    PANEL_WIDTH: int = 280

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setStyleSheet("background: #1a1a1a; border-right: 1px solid #333;")
        self._build_ui()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def scan_start(self) -> float:
        return self._spin_start.value()

    @property
    def scan_stop(self) -> float:
        return self._spin_stop.value()

    @property
    def block_freq_update(self) -> bool:
        return self._block_freq_update

    @block_freq_update.setter
    def block_freq_update(self, value: bool) -> None:
        self._block_freq_update = value

    @property
    def f_spin(self) -> QDoubleSpinBox:
        """Спінбокс поточної частоти (МГц)."""
        return self._f_spin

    @property
    def btn_scan(self) -> QPushButton:
        return self._btn_scan

    # ── Public API ────────────────────────────────────────────────────────

    def log(self, text: str, color: str = "white") -> None:
        """Додає рядок у лог з часовою міткою."""
        t = time.strftime("%H:%M:%S")
        self._log_w.append(f"<span style='color:{color}'>[{t}] {text}</span>")
        self._log_w.moveCursor(self._log_w.textCursor().End)

    def set_scan_btn_scanning(self) -> None:
        """Переводить кнопку сканування в стан СТОП."""
        self._btn_scan.setText("STOP SCAN")
        self._btn_scan.setStyleSheet(
            "background: #bd2c00; color: white; font-weight: bold;"
        )

    def set_scan_btn_idle(self) -> None:
        """Переводить кнопку сканування в стан СТАРТ."""
        self._btn_scan.setText(" START SCAN")
        self._btn_scan.setStyleSheet(
            "background: #2ea043; color: white; font-weight: bold;"
        )

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._block_freq_update: bool = False

        layout = QVBoxLayout(self)

        # — SYSTEM —
        layout.addWidget(QLabel("SYSTEM", styleSheet="color: white; font-weight: bold;"))
        btn_connect = QPushButton("🔌 RECONNECT")
        btn_connect.setStyleSheet("background: #444; color: white;")
        btn_connect.clicked.connect(self._ctrl.connect_pluto)
        layout.addWidget(btn_connect)
        layout.addSpacing(15)

        # — SCANNER RANGE —
        scan_group = QGroupBox("SCANNER RANGE")
        scan_group.setStyleSheet(
            "QGroupBox { color: #afa; border: 1px solid #444; margin-top: 10px; }"
        )
        gl_scan = QGridLayout(scan_group)

        self._spin_start = QDoubleSpinBox()
        self._spin_start.setRange(70, 6000)
        self._spin_start.setValue(430)
        self._spin_start.setDecimals(0)

        self._spin_stop = QDoubleSpinBox()
        self._spin_stop.setRange(70, 6000)
        self._spin_stop.setValue(440)
        self._spin_stop.setDecimals(0)

        spin_style = "background: #222; color: white; border: 1px solid #555;"
        self._spin_start.setStyleSheet(spin_style)
        self._spin_stop.setStyleSheet(spin_style)

        gl_scan.addWidget(QLabel("Start:"), 0, 0)
        gl_scan.addWidget(self._spin_start, 0, 1)
        gl_scan.addWidget(QLabel("Stop:"), 1, 0)
        gl_scan.addWidget(self._spin_stop, 1, 1)

        self._btn_scan = QPushButton("▶ START SCAN")
        self._btn_scan.setStyleSheet(
            "background: #2ea043; color: white; font-weight: bold;"
        )
        self._btn_scan.clicked.connect(self._on_scan_toggle)
        gl_scan.addWidget(self._btn_scan, 2, 0, 1, 2)
        layout.addWidget(scan_group)

        # — HISTOGRAM —
        btn_hist = QPushButton("📊 SHOW HISTOGRAM")
        btn_hist.setStyleSheet(
            "background: #6a1b9a; color: white; font-weight: bold; margin-top: 5px;"
        )
        btn_hist.clicked.connect(self._ctrl.open_histogram)
        layout.addWidget(btn_hist)

        btn_reset_hist = QPushButton("RESET HISTOGRAM")
        btn_reset_hist.setStyleSheet("background: #444; color: #ddd; margin-top: 0px;")
        btn_reset_hist.clicked.connect(self._ctrl.reset_histogram)
        layout.addWidget(btn_reset_hist)
        layout.addSpacing(15)

        # — NOISE MASK —
        mask_group = QGroupBox("NOISE MASK")
        mask_group.setStyleSheet(
            "QGroupBox { color: #afa; border: 1px solid #444; margin-top: 10px; }"
        )
        gl_mask = QGridLayout(mask_group)

        btn_set_mask = QPushButton("SET MASK")
        btn_set_mask.setStyleSheet(
            "background: #007acc; color: white; font-weight: bold;"
        )
        btn_set_mask.clicked.connect(self._ctrl.set_mask)

        btn_clear_mask = QPushButton("CLEAR")
        btn_clear_mask.setStyleSheet(
            "background: #bd2c00; color: white; font-weight: bold;"
        )
        btn_clear_mask.clicked.connect(self._ctrl.clear_mask)

        gl_mask.addWidget(btn_set_mask, 0, 0)
        gl_mask.addWidget(btn_clear_mask, 0, 1)
        layout.addWidget(mask_group)
        layout.addSpacing(15)

        # — FREQUENCY —
        layout.addWidget(
            QLabel("CURRENT FREQ (MHz)", styleSheet="color: cyan; font-weight: bold;")
        )
        self._f_spin = QDoubleSpinBox()
        self._f_spin.setRange(70, 6000)
        self._f_spin.setValue(433.0)
        self._f_spin.setDecimals(3)
        self._f_spin.setStyleSheet(
            "background: #333; color: white; padding: 5px; font-size: 16px;"
        )
        self._f_spin.editingFinished.connect(self._on_freq_changed)
        layout.addWidget(self._f_spin)

        # — GAIN —
        layout.addWidget(QLabel("GAIN", styleSheet="color: #aaa;"))
        self._gain = QSlider(Qt.Horizontal)
        self._gain.setRange(0, 80)
        self._gain.setValue(50)
        self._gain.valueChanged.connect(lambda v: self._ctrl.set_gain(v))
        layout.addWidget(self._gain)
        layout.addSpacing(10)

        # — LOG —
        self._log_w = QTextEdit()
        self._log_w.setReadOnly(True)
        self._log_w.setStyleSheet(
            "background: #111; color: lime; border: none; "
            "font-size: 11px; font-family: Consolas;"
        )
        layout.addWidget(self._log_w)

    # ── Handlers ──────────────────────────────────────────────────────────

    def _on_scan_toggle(self) -> None:
        self._ctrl.set_scan_range(self._spin_start.value(), self._spin_stop.value())
        self._ctrl.toggle_scan()

    def _on_freq_changed(self) -> None:
        if self._block_freq_update:
            return
        val_mhz = self._f_spin.value()
        self._ctrl.set_freq(val_mhz * 1e6)
        self._f_spin.clearFocus()
