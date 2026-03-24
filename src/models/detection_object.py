from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectionObject:
    """
    Модель об'єкта для бази даних.
    Оновлено для підтримки списків частот та перейменовано audio -> sound.
    """

    id: Optional[int]  # Може бути None, якщо об'єкт новий
    name: str

    class_id: int
    object_class: str  # Популюється назвою класу

    is_dangerous: bool = False
    rf_params_hz: List[str] = field(default_factory=list)  # Список рядків "min-max"
    sound_params_hz: List[int] = field(default_factory=list)  # Список чисел

    @staticmethod
    def from_dict(data: dict) -> "DetectionObject":
        raw_id = data.get("id")
        obj_id = int(raw_id) if raw_id is not None else None

        return DetectionObject(
            id=obj_id,
            name=data.get("name", "Unnamed"),
            class_id=int(data.get("class_id", 0)),
            object_class=data.get("object_class", "Unknown"),
            is_dangerous=bool(data.get("is_dangerous", False)),
            rf_params_hz=data.get("rf_params_hz", []),
            sound_params_hz=data.get("sound_params_hz", []),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "class_id": self.class_id,
            "object_class": self.object_class,
            "is_dangerous": self.is_dangerous,
            "rf_params_hz": self.rf_params_hz,
            "sound_params_hz": self.sound_params_hz,
        }
