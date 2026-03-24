"""
Модель події детекції.
type: Джерело детекції -> ТІЛЬКИ "RF" або "Sound".
object_class: Клас об'єкта -> "drone", "bird", "mavic_3" тощо.
"""

from dataclasses import dataclass
import uuid
from datetime import datetime
from typing import Optional
import numpy as np

from src.models.source_type import SourceType


@dataclass
class DetectionEvent:

    id: str
    type: str  # "RF" або "Sound"
    name: str
    object_class: str
    confidence: float
    timestamp: str
    distance_km: float
    angle: float
    frequency_hz: float

    @staticmethod
    def from_dict(data: dict) -> "DetectionEvent":
        """Парсинг вхідного словника JSON у об'єкт."""

        raw_type = data.get("type", SourceType.RF)
        if raw_type not in [SourceType.RF, SourceType.SOUND]:
            raw_type = SourceType.RF

        obj_class = data.get("object_class", data.get("class", "unknown"))

        return DetectionEvent(
            id=data.get("id", str(uuid.uuid4())),
            type=raw_type,
            name=data.get("name", "unknown"),
            object_class=obj_class,
            confidence=float(data.get("confidence", 0.0)),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            distance_km=float(data.get("distance_km", 0)),
            angle=float(data.get("angle", 0)),
            frequency_hz=float(data.get("frequency_hz", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "object_class": self.object_class,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "distance_km": self.distance_km,
            "angle": self.angle,
            "frequency_hz": self.frequency_hz,
        }
