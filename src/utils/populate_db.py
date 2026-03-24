import sys
import os
import time
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

from src.services.database_service import DatabaseService
from src.models.detection_object import DetectionObject
from src.models.object_class import ObjectClass


def run_seeding():
    app = QCoreApplication(sys.argv)
    db = DatabaseService()
    loop = QEventLoop()

    # Стан завантаження
    classes_to_add = [
        "Recon Drone",
        "Loitering Munition",
        "FPV / Kamikaze",
        "Fixed Wing Recon",
        "False Alarm",
        "Interference",
    ]

    # Словник сигнатур: (Назва, Class_ID, is_dangerous, rf_params, sound_params)
    # Зверни увагу: для одного імені робимо два окремих записи
    signatures_data = [
        # DJI Mavic 3
        (
            "DJI Mavic 3",
            1,
            True,
            ["2400000000-2483500000", "5725000000-5850000000"],
            [],
        ),
        ("DJI Mavic 3", 1, True, [], [450.0, 600.0]),
        # Shahed-136
        ("Shahed-136", 2, True, ["1575420000-1575420000", "1227600000-1227600000"], []),
        ("Shahed-136", 2, True, [], [60.0, 95.0]),
        # FPV Drone
        ('FPV Drone 7"', 3, True, ["915000000-928000000", "5650000000-5900000000"], []),
        ('FPV Drone 7"', 3, True, [], [850.0, 1100.0]),
        # Orlan-10
        ("Orlan-10", 4, True, ["433000000-440000000", "900000000-920000000"], []),
        ("Orlan-10", 4, True, [], [130.0, 170.0]),
        # False Alarms (Тільки звук або тільки радіо)
        ("Crow (Ворона)", 5, False, [], [1200.0, 1600.0]),
        ("Gas Mower (Косарка)", 5, False, [], [80.0, 120.0]),
        ("Public WiFi Hotspot", 6, False, ["2400000000-2483000000"], []),
        ("GSM 900 Link", 6, False, ["935000000-960000000"], []),
    ]

    print("--- START SEEDING ---")

    # 1. Додаємо класи
    for class_name in classes_to_add:
        print(f"Adding class: {class_name}")
        db.add_class(ObjectClass(id=None, name=class_name))
        time.sleep(0.1)  # Даємо час потокам відпрацювати

    time.sleep(1)  # Чекаємо завершення транзакцій класів

    # 2. Додаємо сигнатури
    for name, c_id, dangerous, rf, sound in signatures_data:
        print(f"Adding signature: {name} (Dangerous: {dangerous})")

        # Створюємо DTO
        obj = DetectionObject(
            id=None,
            name=name,
            class_id=c_id,
            object_class="",  # Сервіс сам підтягне ім'я по ID
            is_dangerous=dangerous,
            rf_params_hz=rf,
            sound_params_hz=sound,
        )

        db.add_object(obj)
        time.sleep(0.1)

    print("--- SEEDING FINISHED ---")
    print("Wait 2 seconds and close...")
    QTimer.singleShot(2000, loop.quit)
    loop.exec()


if __name__ == "__main__":

    run_seeding()
