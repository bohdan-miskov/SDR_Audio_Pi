from dataclasses import dataclass, asdict
import uuid
from datetime import datetime
import numpy as np
from typing import List, Union


@dataclass
class SpectralData:
    """Важкі дані для спектрального аналізу"""

    # Центральна частота прийому в Герцах.
    center_freq_hz: float

    # Смуга пропускання (ширина огляду) в Герцах.
    sample_rate_hz: float

    # Тривалість запису у секундах.
    duration_sec: float

    # Матриця амплітуд (спектрограма).
    # Shape: (Rows, Cols) -> (Time, Frequency).
    #   - Rows (висота): кількість пакетів у часі.
    #   - Cols (ширина): кількість бінів FFT (наприклад, 1024).
    #
    # Тип: uint8 (0...255).
    # Переводиться у db за формулою dB=value_uint8−DB_OFFSET(у constants)
    data_magnitude: np.ndarray

    @staticmethod
    def from_dict(data: dict) -> "SpectralData":

        mag_data = data.get("data_magnitude", [])
        if isinstance(mag_data, list):
            mag_data = np.array(mag_data, dtype=np.uint8)

        return SpectralData(
            center_freq_hz=float(data.get("center_freq_hz", 0)),
            sample_rate_hz=float(data.get("sample_rate_hz", 0)),
            duration_sec=float(data.get("duration_sec", 0)),
            data_magnitude=mag_data,
        )

    def to_dict(self) -> dict:
        return {
            "center_freq_hz": self.center_freq_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_sec": self.duration_sec,
            "data_magnitude": (
                self.data_magnitude.tolist()
                if isinstance(self.data_magnitude, np.ndarray)
                else self.data_magnitude
            ),
        }


@dataclass
class DetectionBackground:
    """
    Фоновий спектр.
    """

    id: str
    timestamp: str
    spectral_data: SpectralData

    @staticmethod
    def from_dict(data: dict) -> "DetectionBackground":
        spec_raw = data.get("spectral_data", {})
        spectral_obj = SpectralData.from_dict(spec_raw)

        return DetectionBackground(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            spectral_data=spectral_obj,
        )

    def to_dict(self) -> dict:
        spec_dict = self.spectral_data.to_dict()

        return {"id": self.id, "timestamp": self.timestamp, "spectral_data": spec_dict}
