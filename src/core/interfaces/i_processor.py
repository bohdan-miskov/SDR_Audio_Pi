"""
core/interfaces/i_processor.py
Інтерфейс DSP-процесора через typing.Protocol.
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class IProcessor(Protocol):
    """Контракт для будь-якого DSP-процесора."""

    @property
    def mask_enabled(self) -> bool: ...

    def process(self, iq_samples: np.ndarray, center_freq: float) -> tuple: ...
    def set_mask_flag(self) -> None: ...
    def clear_mask(self) -> None: ...
