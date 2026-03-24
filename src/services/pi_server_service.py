import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from PyQt6.QtCore import QObject, pyqtSlot, QByteArray
from PyQt6.QtNetwork import QTcpServer, QTcpSocket, QHostAddress

from src.services.database_service import DatabaseService
from src.models.detection_event import DetectionEvent
from src.models.detection_object import DetectionObject
from src.models.object_class import ObjectClass
from src.models.gps_data import GPSData
from src.models.detection_background import DetectionBackground
from src.models.service_response import ServiceResponse, StatusCode


class PiServerService(QObject):
    """
    Сервіс-сервер для Raspberry Pi.
    Приймає підключення від Desktop-клієнта, обробляє команди та керує периферією.
    """

    def __init__(self, port: int = 6000, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.port = port
        self.server: Optional[QTcpServer] = None
        self.client_socket: Optional[QTcpSocket] = None

        # --- ПІДКЛЮЧЕННЯ БД ---
        self.db = DatabaseService()

        # 2. Підключаємо єдиний сигнал результату
        self.db.request_finished.connect(self.send_db_response)

        # self.hardware_manager = ...

    def start(self) -> None:
        self.server = QTcpServer(self)
        self.server.newConnection.connect(self._handle_new_connection)

        if self.server.listen(QHostAddress.SpecialAddress.Any, self.port):
            print(f"[PiProxy] Server listening on port {self.port}")
        else:
            print(f"[PiProxy] Error starting server: {self.server.errorString()}")

    def stop(self) -> None:
        if self.client_socket:
            self.client_socket.disconnectFromHost()
            if self.client_socket.state() != QTcpSocket.SocketState.UnconnectedState:
                self.client_socket.waitForDisconnected(1000)

        if self.server:
            self.server.close()

        print("[PiProxy] Server stopped.")

    @pyqtSlot()
    def _handle_new_connection(self) -> None:
        """
        Обробка нового підключення.
        Стратегія: Одночасно лише один клієнт. Новий клієнт 'вибиває' старого.
        """
        if self.client_socket:
            print(
                f"[PiProxy] Closing old connection from {self.client_socket.peerAddress().toString()}"
            )
            self.client_socket.close()
            self.client_socket.deleteLater()

        self.client_socket = self.server.nextPendingConnection()
        print(
            f"[PiProxy] Client connected: {self.client_socket.peerAddress().toString()}"
        )

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
                json_str = bytes(line).decode("utf-8")
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
            # TODO: Start SDR process
            # self.send_rf_stream_data({...})
            pass

        elif action == "stop_rf_stream":
            pass

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
            # TODO: Set range for detecting, range is in mhz
            pass

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
                self.client_socket.write(QByteArray(msg))
                self.client_socket.flush()
            except Exception as e:
                print(f"[PiProxy] Send Error: {e}")

    def send_detection_event(self, event: DetectionEvent) -> None:
        print(f"[PiProxy] Sending Detection: {event.name}")
        self.send_packet("detection", event.to_dict())

    def send_detection_background(self, back: DetectionBackground) -> None:
        print(f"[PiProxy] Sending Detection background")
        self.send_packet("detection_background", back.to_dict())

    def send_gps_data(self, gps_data: GPSData) -> None:
        """Відправляє координати {lat, lon, alt}."""
        self.send_packet("gps_position", gps_data.to_dict())

    def send_rf_stream_data(self, spectrum_data: Dict[str, Any]) -> None:
        """Відправляє пакет даних спектру."""
        self.send_packet("rf_stream", spectrum_data)

    def send_sound_stream_data(self, audio_analysis: Dict[str, Any]) -> None:
        """Відправляє дані аналізу звуку."""
        self.send_packet("sound_stream", audio_analysis)

    # --- DB RESPONSE SENDERS (СЛОТИ) ---

    @pyqtSlot(object)
    def send_db_response(self, response: ServiceResponse) -> None:
        """
        Відправляє результат виконання будь-якої DB операції (успіх або помилка).
        """
        print(f"[PiProxy] DB Response: {response.operation} -> {response.status}")

        packet_data = response.to_dict()

        self.send_packet("db_operation_result", packet_data)
