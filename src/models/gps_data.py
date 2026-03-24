from dataclasses import dataclass


@dataclass
class GPSData:
    """
    Модель об'єкта для роботи з gps.
    """

    lat: float
    lon: float
    strength: int  # Значення 0-100

    @staticmethod
    def from_dict(data: dict) -> "GPSData":
        return GPSData(
            lat=data.get("lat", 0),
            lon=int(data.get("lon", 0)),
            strength=data.get("strength", 0),
        )

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "strength": self.strength,
        }
