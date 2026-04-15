"""
AntennaController — керування антенами через GPIO (Raspberry Pi).
Підтримує режим Amplitude Blanking: цикл ANT_A → ANT_B → ANT_C → BLANK.
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
    """
    Управляє вибором антени через GPIO-піни.

    Підтримує Amplitude Blanking:
        switch_to('ANT_A') → GPIO 17 HIGH, решта LOW
        switch_to('ANT_B') → GPIO 27 HIGH, решта LOW
        switch_to('ANT_C') → GPIO 22 HIGH, решта LOW
        blank()            → ВСІ LOW (мертвий крок, маркер циклу)
    """

    # Порядок перемикання антен у пеленгаторному циклі (без BLANK — він додається автоматично)
    # 5 антен рівномірно по колу (0°, 72°, 144°, 216°, 288°)
    ANTENNA_CYCLE: list[str] = ['ANT_A', 'ANT_B', 'ANT_C', 'ANT_D', 'ANT_E']

    def __init__(self):
        self._pins: dict[str, int] = GPIO_PINS
        self._current_ant: str | None = None
        self._cycle_index: int = 0            # Поточна позиція в ANTENNA_CYCLE
        self._blanking_enabled: bool = False  # Чи використовується Amplitude Blanking

        GPIO.setmode(GPIO.BCM)
        for pin in self._pins.values():
            GPIO.setup(pin, GPIO.OUT)

        # При старті занулити всі виходи
        self._all_low()
        print(f"[Антена] Ініціалізація (RPI Mode: {IS_RPI})")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_antenna(self) -> str | None:
        """Назва поточно активної антени або None (під час BLANK)."""
        return self._current_ant

    @property
    def available_antennas(self) -> list[str]:
        """Список доступних антен."""
        return list(self._pins.keys())

    @property
    def blanking_enabled(self) -> bool:
        """True — активовано режим Amplitude Blanking."""
        return self._blanking_enabled

    @blanking_enabled.setter
    def blanking_enabled(self, value: bool) -> None:
        self._blanking_enabled = bool(value)
        print(f"[Антена] Amplitude Blanking: {'ON' if value else 'OFF'}")

    @property
    def cycle_index(self) -> int:
        """Поточний індекс у ANTENNA_CYCLE."""
        return self._cycle_index

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
        for pin in self._pins.values():
            GPIO.output(pin, GPIO.HIGH if pin == target_pin else GPIO.LOW)
        self._current_ant = antenna_name
        return True

    def blank(self) -> None:
        """
        BLANK-крок Amplitude Blanking: відключає всі антени.
        Для приймача це виглядає як різкий провал амплітуди —
        маркер початку нового антенного циклу.
        """
        self._all_low()
        self._current_ant = None

    def next_in_cycle(self) -> str | None:
        """
        Робить наступний крок у пеленгаторному циклі.
        Якщо blanking_enabled=True, після останньої антени викликає blank().

        Returns:
            Назва антени, що тільки-но активована, або None (крок BLANK).
        """
        n = len(self.ANTENNA_CYCLE)

        if self._cycle_index < n:
            # Звичайний крок — активуємо антену
            ant_name = self.ANTENNA_CYCLE[self._cycle_index]
            self.switch_to(ant_name)
            self._cycle_index += 1
            return ant_name
        else:
            # Крок BLANK (тільки якщо blanking увімкнено)
            if self._blanking_enabled:
                self.blank()
            self._cycle_index = 0
            return None

    def reset_cycle(self) -> None:
        """Скидає цикл до ANT_A."""
        self._cycle_index = 0

    def cleanup(self) -> None:
        """Звільняє GPIO-ресурси."""
        self._all_low()
        GPIO.cleanup()

    # ── Private ───────────────────────────────────────────────────────────────

    def _all_low(self) -> None:
        """Знімає напругу з усіх GPIO-пінів."""
        for pin in self._pins.values():
            GPIO.output(pin, GPIO.LOW)
        self._current_ant = None
