import json
import uuid
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from core.config import SAMPLE_RATE, DB_OFFSET
from PyQt6.QtCore import QObject, pyqtSlot, QByteArray, pyqtSignal
from PyQt6.QtNetwork import QTcpServer, QTcpSocket, QHostAddress

from services.database_service import DatabaseService
from models.detection_event import DetectionEvent
from models.detection_object import DetectionObject
from models.object_class import ObjectClass
from models.gps_data import GPSData
from models.detection_background import DetectionBackground
from models.service_response import ServiceResponse, StatusCode
from models.stream_data import StreamDataChunk


class RFStreamWorker(threading.Thread):
    """
    Фоновий потік для стрімінгу спектральних даних.
    Використовує threading.Thread замість QThread, щоб уникнути конфлікту PyQt6/PyQt6.
    """

    def __init__(self, data_signal: pyqtSignal, threat_signal: pyqtSignal) -> None:
        super().__init__()
        self.data_signal = data_signal
        self.threat_signal = threat_signal
        self._running = False
        self._lock = threading.Lock()
        self._pending_range = None
        self._hop_interval = 0.2
        self._last_hop_time = 0.0

        # Імпортуємо RF/DSP модулі ліниво, щоб уникнути будь-яких PyQt6/PyQt6 конфліктів при старті
        from hardware.radio_sensor import RadioSensor
        from scanning.scan_manager import ScanManager
        from dsp.fft_processor import FFTProcessor
        from dsp.channel_analyzer import ChannelAnalyzer
        from dsp.protocol_classifier import ProtocolClassifier
        from dsp.threat_detector import ThreatDetector

        self.radio = RadioSensor(sim_antenna_mux=True)
        self.scanner = ScanManager()
        self.fft_proc = FFTProcessor()
        self.channels = ChannelAnalyzer()
        self.classifier = ProtocolClassifier()
        self.detector = ThreatDetector(alert_callback=self._on_threat_detected)

    def _on_threat_detected(self, msg: str, color: str) -> None:
        self.threat_signal.emit(msg, color)

    def set_range(self, start_mhz: float, stop_mhz: float) -> None:
        with self._lock:
            self._pending_range = (start_mhz, stop_mhz)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        self.radio.connect()

        while self._running:
            # 1. Оновлення діапазону сканування
            with self._lock:
                if self._pending_range is not None:
                    start_mhz, stop_mhz = self._pending_range
                    print(f"[RFWorker] Applying new range: {start_mhz}-{stop_mhz} MHz")
                    # Якщо діапазон менший за SAMPLE_RATE — туним на центр
                    span_hz = (stop_mhz - start_mhz) * 1e6
                    if span_hz < SAMPLE_RATE:
                        center_mhz = (start_mhz + stop_mhz) / 2.0
                        self.scanner.set_custom_range(
                            center_mhz - 0.001, center_mhz + 0.001
                        )
                    else:
                        self.scanner.set_custom_range(start_mhz, stop_mhz)
                    self._pending_range = None

            # 2. Сканування — перехід на наступну частоту
            t_now = time.time()
            if t_now - self._last_hop_time > self._hop_interval:
                next_freq = self.scanner.get_next_frequency()
                self.radio.tune(int(next_freq))
                self._last_hop_time = t_now

            # 3. Отримання IQ-семплів
            iq_data = self.radio.get_samples()
            center_f = self.radio.current_freq

            # 4. DSP обробка
            freqs, psd = self.fft_proc.process(iq_data, center_f)
            if freqs is None or len(freqs) == 0:
                time.sleep(0.01)
                continue

            self.channels.analyze(freqs, psd, center_f, self.fft_proc.mask_enabled)
            self.classifier.classify(freqs, psd, center_f, self.fft_proc.mask_enabled)

            noise_floor = (
                0.0 if self.fft_proc.mask_enabled else float(np.percentile(psd, 30))
            )

            # Фільтр хибних срацьовань: якщо максимальний SNR нижче порогу — нічого не робити
            # (RSSI_THRESHOLD з конфігу = 15 dB; запобігає детекцію шуму симуляції)
            from core.config import RSSI_THRESHOLD
            max_snr = float(np.max(psd)) - noise_floor
            if max_snr < RSSI_THRESHOLD and not self.fft_proc.mask_enabled:
                # Тільки шум — не запускаємо детектор щоб не накопичувати persistence
                time.sleep(0.01)
                continue

            self.detector.analyze(
                freqs=freqs,
                psd=psd,
                noise_floor=noise_floor,
                center_freq=center_f,
                mask_enabled=self.fft_proc.mask_enabled,
                current_profile=self.channels.current_profile,
                channel_activity=self.channels.channel_activity,
                detected_protocol=self.classifier.detected_protocol,
                detected_bandwidth=self.classifier.detected_bandwidth,
                detected_power=self.classifier.detected_power,
            )

            # --- Формування пакету у форматі StreamDataChunk ---

            # Subsampling до 1024 точок для оптимізації мережі
            n_points = len(psd)
            target_points = 1024
            if n_points > target_points:
                indices = np.linspace(0, n_points - 1, target_points, dtype=int)
                psd_sub = psd[indices]
            else:
                psd_sub = psd

            # Відносне кодування PSD (SNR) у uint8:
            #   1. Відраховуємо шум як 10-й перцентиль — шум → 0 dB
            #   2. Кодуємо: value = clip(SNR + DB_OFFSET, 0, 255)
            # Клієнт декодує: SNR_dB = value - DB_OFFSET
            noise_floor_ref = float(np.percentile(psd_sub, 10))
            psd_snr = psd_sub - noise_floor_ref
            data_magnitude = np.clip(
                np.round(psd_snr + DB_OFFSET), 0, 255
            ).astype(np.uint8)

            sample_rate = float(getattr(self.radio, 'sample_rate', SAMPLE_RATE))

            chunk = StreamDataChunk(
                stream_type="RF",
                data_magnitude=data_magnitude,
                center_freq_hz=float(center_f),
                sample_rate_hz=sample_rate,
                timestamp=time.time(),
            )

            self.data_signal.emit(chunk.to_dict())
            time.sleep(0.1)


import random


def _random_angle_distance() -> dict:  # DEPRECATED — replaced by real bearing
    """Генерує випадковий кут (0‑360°) та відстань (1‑7 км).
    Повертає словник: {'angle': <float>, 'distance': <float>}"""
    angle = random.uniform(0, 360)
    distance = random.uniform(1, 7)
    return {"angle": round(angle, 1), "distance": round(distance, 3)}

class PiServerService(QObject):
    """
    Сервіс-сервер для Raspberry Pi.
    Приймає підключення від Desktop-клієнта, обробляє команди та керує периферією.
    """

    rf_data_received = pyqtSignal(dict)
    rf_threat_received = pyqtSignal(str, str)
    # Сигнал з реальним азимутом від DSPEngine (підключається з main.py)
    bearing_updated = pyqtSignal(float, float, float, str)

    def __init__(self, port: int = 6000, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.port = port
        self.server: Optional[QTcpServer] = None
        self.client_socket: Optional[QTcpSocket] = None

        # --- RF WORKER ---
        self.rf_worker: Optional[RFStreamWorker] = None
        self._pending_rf_range: Optional[tuple[float, float]] = None

        # --- Кеш останнього реального азимуту від DirectionFinder ---
        self._last_bearing: float = 0.0
        self._last_bearing_conf: float = 0.0
        self._last_bearing_unc: float = 90.0
        self._last_bearing_valid: bool = False

        # --- ПІДКЛЮЧЕННЯ БД ---
        self.db = DatabaseService()

        # Підключаємо єдиний сигнал результату
        self.db.request_finished.connect(self.send_db_response)

        # Підключення сигналів від RF-воркера
        self.rf_data_received.connect(self.send_rf_stream_data)
        self.rf_threat_received.connect(self._handle_rf_threat)

        # Підключення реального азимуту
        self.bearing_updated.connect(self._on_bearing_updated)

    @pyqtSlot(float, float, float, str)
    def _on_bearing_updated(self, bearing: float, conf: float,
                             unc: float, method: str) -> None:
        """Отримує реальний азимут від DSPEngine і кешує його."""
        self._last_bearing = bearing
        self._last_bearing_conf = conf
        self._last_bearing_unc = unc
        self._last_bearing_valid = True

    def start(self) -> None:
        self.server = QTcpServer(self)
        self.server.newConnection.connect(self._handle_new_connection)

        if self.server.listen(QHostAddress.SpecialAddress.Any, self.port):
            print(f"[PiProxy] Server listening on port {self.port}")
        else:
            print(f"[PiProxy] Error starting server: {self.server.errorString()}")

        # Автоматичний запуск RF-воркера при старті сервера
        self._start_rf_worker()

    def _start_rf_worker(self) -> None:
        """Автоматичний запуск RF-воркера при старті сервера."""
        if self.rf_worker and self.rf_worker.is_alive():
            print("[PiProxy] RF stream already running")
        else:
            print("[PiProxy] Auto-starting RF stream worker...")
            self.rf_worker = RFStreamWorker(
                self.rf_data_received, self.rf_threat_received
            )
            if self._pending_rf_range:
                start_mhz, stop_mhz = self._pending_rf_range
                self.rf_worker.set_range(start_mhz, stop_mhz)
            self.rf_worker.start()

    def stop(self) -> None:
        if self.rf_worker and self.rf_worker.is_alive():
            print("[PiProxy] Stopping RF stream worker...")
            self.rf_worker.stop()
            self.rf_worker.join(timeout=2.0)
            self.rf_worker = None

        if self.client_socket:
            self.client_socket.disconnectFromHost()
            if self.client_socket.state() != QTcpSocket.SocketState.UnconnectedState:
                self.client_socket.waitForDisconnected(1000)

        if self.server:
            self.server.close()

        print("[PiProxy] Server stopped.")

    @pyqtSlot(str, str)
    def _handle_rf_threat(self, message: str, color: str) -> None:
        """Обробка виявленої загрози від RF-воркера."""
        print(f"[PiProxy] RF Threat: {message} ({color})")

        # Пропускаємо окремі частоти (Ціль: 900 МГц)
        if message.startswith(" Ціль:"):
            return

        obj_class = "drone"
        if "WiFi" in message or "WIFI" in message:
            obj_class = "wifi"
        elif "LORA" in message or "LoRa" in message:
            obj_class = "lora"
        elif "Narrowband" in message:
            obj_class = "narrowband"
        elif "Wideband" in message:
            obj_class = "wideband"
        elif "ДЕТЕКЦІЯ" in message:
            obj_class = "target"
        elif "Analog" in message or "Video" in message or "CH" in message:
            obj_class = "analog_video"

        # Використовуємо реальний азимут з DirectionFinder (якщо є)
        if self._last_bearing_valid:
            real_angle = self._last_bearing
        else:
            real_angle = random.uniform(0, 360)  # fallback поки немає пеленгу
        distance_km = random.uniform(1, 7)  # TODO: замінити на RSSI-оцінку

        import re
        # Шукаємо частоту в MHz
        freqs_found = re.findall(r"(\d+(?:\.\d+)?)\s*MHz", message, re.IGNORECASE)
        freq_mhz = float(freqs_found[0]) if freqs_found else 0.0

        # Якщо частоти в MHz немає, але є номер каналу CH - конвертуємо
        if freq_mhz == 0.0:
            ch_match = re.search(r"CH(\d+)", message, re.IGNORECASE)
            if ch_match:
                ch_num = int(ch_match.group(1))
                # WiFi 2.4 ГГц канали
                wifi_24 = {1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432,
                           6: 2437, 7: 2442, 8: 2447, 9: 2452, 10: 2457,
                           11: 2462, 12: 2467, 13: 2472, 14: 2484}
                # WiFi 5 ГГц канали
                wifi_5 = {36: 5180, 40: 5200, 44: 5220, 48: 5240,
                          52: 5260, 56: 5280, 60: 5300, 64: 5320,
                          100: 5500, 104: 5520, 108: 5540, 112: 5560,
                          116: 5580, 120: 5600, 124: 5620, 128: 5640,
                          132: 5660, 136: 5680, 140: 5700, 144: 5720,
                          149: 5745, 153: 5765, 157: 5785, 161: 5805, 165: 5825}
                if ch_num in wifi_24:
                    freq_mhz = wifi_24[ch_num]
                elif ch_num in wifi_5:
                    freq_mhz = wifi_5[ch_num]

        event = DetectionEvent(
            id=str(uuid.uuid4()),
            type="RF",
            name=message,
            object_class=obj_class,
            confidence=1.0,
            timestamp=datetime.now().isoformat(),
            distance_km=distance_km,
            angle=real_angle,
            frequency_hz=freq_mhz * 1e6,
        )
        self.send_detection_event(event)

    @pyqtSlot()
    def _handle_new_connection(self) -> None:
        """
        Обробка нового підключення.
        Стратегія: Одночасно лише один клієнт. Новий клієнт 'вибиває' старого.
        """
        if self.server is None:
            return

        if self.client_socket:
            peer_address = self.client_socket.peerAddress()
            addr_str = peer_address.toString() if peer_address else "Unknown"
            print(f"[PiProxy] Closing old connection from {addr_str}")
            self.client_socket.close()
            self.client_socket.deleteLater()

        self.client_socket = self.server.nextPendingConnection()
        if self.client_socket is None:
            return

        peer_address = self.client_socket.peerAddress()
        addr_str = peer_address.toString() if peer_address else "Unknown"
        print(f"[PiProxy] Client connected: {addr_str}")

        self.client_socket.readyRead.connect(self._read_data)
        self.client_socket.disconnected.connect(self._handle_disconnected)

    @pyqtSlot()
    def _handle_disconnected(self) -> None:
        print("[PiProxy] Client disconnected.")
        self.client_socket = None

    @pyqtSlot()
    def _read_data(self) -> None:
        if not self.client_socket:
            return

        while self.client_socket.canReadLine():
            line = self.client_socket.readLine().trimmed()
            try:
                json_str = line.data().decode("utf-8")
                if not json_str:
                    continue

                packet = json.loads(json_str)
                action = packet.get("action")
                data = packet.get("data", {})

                print(f"[PiProxy] Received action: {action}")
                self._process_command(action, data)

            except json.JSONDecodeError:
                print(f"[PiProxy] JSON Error: {line}")
                # Тут можна відправити клієнту 400 Bad Request, якщо треба
            except Exception as e:
                print(f"[PiProxy] Processing Error: {e}")

    def _process_command(self, action: str, data: Dict[str, Any]) -> None:
        """Головний маршрутизатор команд."""

        if action.startswith("db_"):
            self._handle_db_command(action, data)
        else:
            self._handle_hardware_command(action, data)

    def _handle_hardware_command(self, action: str, data: Dict[str, Any]) -> None:
        """Обробка команд, пов'язаних з сенсорами та залізом."""

        if action == "get_gps":
            # TODO: Get real GPS data
            # gps_data = self.hardware.get_gps()
            # self.send_gps_data(gps_data)
            pass

        elif action == "start_rf_stream":
            # Воркер вже запущений автоматично, але можна перезапустити
            self._start_rf_worker()

        elif action == "stop_rf_stream":
            if self.rf_worker and self.rf_worker.is_alive():
                print("[PiProxy] Stopping RF stream...")
                self.rf_worker.stop()
                self.rf_worker.join(timeout=2.0)
                self.rf_worker = None
                print("[PiProxy] RF stream stopped.")
            else:
                print("[PiProxy] RF stream is not running")

        elif action == "start_sound_stream":
            # TODO: Start Audio process
            # self.send_sound_stream_data({...})
            pass

        elif action == "stop_sound_stream":
            pass

        elif action == "start_alarm":
            relays = data.get("relays", [])
            print(f"[PiProxy] Activating relays: {relays}")
            # TODO: GPIO logic
            pass

        elif action == "stop_alarm":
            print("[PiProxy] Deactivating relays")
            pass

        elif action == "false_alarm":
            event_id = data.get("event_id")
            print(f"[PiProxy] Marking event {event_id} as false alarm")
            # TODO: Log false alarm to DB / Retrain model
            pass

        elif action == "set_rf_range":
            r_range = data.get("range", [])
            print(f"[PiProxy] Set follow rf range {r_range}")
            if len(r_range) == 2:
                try:
                    start_mhz = float(r_range[0])
                    stop_mhz = float(r_range[1])
                    self._pending_rf_range = (start_mhz, stop_mhz)
                    if self.rf_worker and self.rf_worker.is_alive():
                        self.rf_worker.set_range(start_mhz, stop_mhz)
                    else:
                        print("[PiProxy] RF worker not running, restarting...")
                        self._start_rf_worker()
                except ValueError as e:
                    print(f"[PiProxy] Invalid range parameters: {r_range} -> {e}")

        else:
            print(f"[PiProxy] Unknown hardware command: {action}")

    def _handle_db_command(self, action: str, data: Dict[str, Any]) -> None:
        """Обробка CRUD операцій та запитів до бази даних."""

        try:
            if action == "db_request_page":
                page = data.get("page", 1)
                size = data.get("size", 15)
                self.db.request_objects_page(page, size)

            elif action == "db_request_add":
                raw_obj = data.get("object")
                if raw_obj:
                    new_object_model = DetectionObject.from_dict(raw_obj)
                    print(f"[PiProxy] Adding object: {new_object_model.name}")
                    self.db.add_object(new_object_model)
                else:
                    raise ValueError("Missing 'object' data")

            elif action == "db_request_update":
                raw_obj = data.get("object")
                if raw_obj:
                    updated_object_model = DetectionObject.from_dict(raw_obj)
                    print(f"[PiProxy] Updating object ID: {updated_object_model.id}")
                    self.db.update_object(updated_object_model)
                else:
                    raise ValueError("Missing 'object' data")

            elif action == "db_request_delete":
                obj_id = data.get("id")
                print(f"[PiProxy] Deleting object ID: {obj_id}")
                if obj_id:
                    self.db.delete_object(obj_id)
                else:
                    raise ValueError("Missing 'id'")

            elif action == "db_request_classes":
                self.db.request_classes()

            elif action == "db_request_add_class":
                class_dict = data.get("class")
                if class_dict:
                    new_class = ObjectClass.from_dict(class_dict)
                    self.db.add_class(new_class)
                else:
                    raise ValueError("Missing 'class' data")

            elif action == "db_request_rename_class":
                old_cls_dict = data.get("old_class")
                new_cls_dict = data.get("new_class")

                if old_cls_dict and new_cls_dict:
                    cls_id = old_cls_dict.get("id")
                    new_name = new_cls_dict.get("name")
                    if cls_id and new_name:
                        # Створюємо DTO з ID і новим ім'ям для апдейту
                        cls_model = ObjectClass(id=cls_id, name=new_name)
                        self.db.update_class(cls_model)
                    else:
                        print("[PiProxy] Rename Class Error: Invalid Data")
                        raise ValueError("Invalid ID or Name")
                else:
                    print("[PiProxy] Rename Class Error: Missing old/new class data")
                    raise ValueError("Missing old/new class data")

            elif action == "db_request_delete_class":
                class_id = data.get("id")
                if class_id:
                    self.db.delete_class(class_id)
                else:
                    raise ValueError("Missing 'id1'")

            else:
                print(f"[PiProxy] Unknown DB command: {action}")
                self._send_protocol_error(action, "Unknown command")

        except Exception as e:
            print(f"[PiProxy] DB Logic Error: {e}")
            self._send_protocol_error(action, str(e))

    # --- SENDER METHODS ---

    def _send_protocol_error(self, operation: str, error_msg: str) -> None:
        """
        Відправляє помилку валідації або протоколу через стандартний ServiceResponse.
        Замінює стару логіку send_db_error.
        """
        response = ServiceResponse(
            operation=operation,
            status=StatusCode.BAD_REQUEST,
            message=f"Protocol/Validation Error: {error_msg}",
        )
        self.send_db_response(response)

    def send_packet(self, action: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Відправка відповіді клієнту."""
        if (
            self.client_socket
            and self.client_socket.state() == QTcpSocket.SocketState.ConnectedState
        ):
            payload = {
                "action": action,
                "data": data if data else {},
                "timestamp": datetime.now().isoformat(),
            }
            try:
                msg = (json.dumps(payload) + "\n").encode("utf-8")
                self.client_socket.write(msg)
                self.client_socket.flush()
            except Exception as e:
                print(f"[PiProxy] Send Error: {e}")

    def send_detection_event(self, event: DetectionEvent) -> None:
        print(f"[PiProxy] Sending Detection: {event.name}")
        self.send_packet("detection", event.to_dict())

    def send_detection_background(self, back: DetectionBackground) -> None:
        print("[PiProxy] Sending Detection background")
        self.send_packet("detection_background", back.to_dict())

    def send_gps_data(self, gps_data: GPSData) -> None:
        """Відправляє координати {lat, lon, alt}."""
        self.send_packet("gps_position", gps_data.to_dict())

    def send_rf_stream_data(self, spectrum_data: StreamDataChunk) -> None:
        """Відправляє пакет даних спектру."""
        self.send_packet("rf_stream", spectrum_data)

    def send_sound_stream_data(self, audio_analysis: StreamDataChunk) -> None:
        """Відправляє дані аналізу звуку."""
        self.send_packet("sound_stream", audio_analysis.to_dict())

    # --- DB RESPONSE SENDERS (СЛОТИ) ---

    @pyqtSlot(object)
    def send_db_response(self, response: ServiceResponse) -> None:
        """
        Відправляє результат виконання будь-якої DB операції (успіх або помилка).
        """
        print(f"[PiProxy] DB Response: {response.operation} -> {response.status}")

        packet_data = response.to_dict()

        self.send_packet("db_operation_result", packet_data)
