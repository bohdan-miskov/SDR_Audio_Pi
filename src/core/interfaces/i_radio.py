"""
core/interfaces/i_radio.py
Інтерфейс SDR-приймача через typing.Protocol.
Protocol не конфліктує з іншими метакласами (зокрема QObject).
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class IRadio(Protocol):
    """Контракт для будь-якого SDR-приймача."""

    @property
    def current_freq(self) -> int: ...

    @property
    def current_gain(self) -> int: ...

    @property
    def mode(self) -> str: ...

    def connect(self) -> str: ...
    def tune(self, freq_hz: int) -> None: ...
    def set_gain(self, gain_db: int) -> None: ...
    def get_samples(self) -> np.ndarray: ...
