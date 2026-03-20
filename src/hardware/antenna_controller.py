"""
AntennaController — керування антенами через GPIO (Raspberry Pi).
При відсутності RPi.GPIO використовує заглушку.
"""
from core.config import GPIO_PINS

try:
    import RPi.GPIO as GPIO
    IS_RPI = True
except ImportError:
    IS_RPI = False

    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0

        def setmode(self, m): pass
        def setup(self, p, m): pass
        def output(self, p, s): pass
        def cleanup(self): pass

    GPIO = MockGPIO()


class AntennaController:
    """Управляє вибором антени через GPIO-піни."""

    def __init__(self):
        self._pins: dict[str, int] = GPIO_PINS
        self._current_ant: str | None = None

        GPIO.setmode(GPIO.BCM)
        for pin in self._pins.values():
            GPIO.setup(pin, GPIO.OUT)

        print(f"[Антена] Ініціалізація (RPI Mode: {IS_RPI})")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_antenna(self) -> str | None:
        """Назва поточно активної антени або None."""
        return self._current_ant

    @property
    def available_antennas(self) -> list[str]:
        """Список доступних антен."""
        return list(self._pins.keys())

    # ── Public methods ────────────────────────────────────────────────────────

    def switch_to(self, antenna_name: str) -> bool:
        """
        Перемикає активну антену.

        Returns:
            True — успішно, False — невідома антена.
        """
        if antenna_name not in self._pins:
            return False
        target_pin = self._pins[antenna_name]
        for name, pin in self._pins.items():
            state = GPIO.HIGH if pin == target_pin else GPIO.LOW
            GPIO.output(pin, state)
        self._current_ant = antenna_name
        return True

    def cleanup(self) -> None:
        """Звільняє GPIO-ресурси."""
        GPIO.cleanup()
