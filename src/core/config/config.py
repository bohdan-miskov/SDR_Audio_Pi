"""
config.py — зворотна сумісність.
Всі константи тепер живуть у core/config/.
Цей файл реекспортує їх, щоб старі імпорти не ламались.
"""
from core.config import (
    SAMPLE_RATE, BUFFER_SIZE, GAIN_DEFAULT, RSSI_THRESHOLD,
    SCAN_RANGES, GPIO_PINS,
    WIFI_CHANNELS_2_4, WIFI_CHANNELS_5, WIFI_CHANNEL_WIDTH,
    BAND_PROFILES, get_band_profile,
    PROTOCOL_BANDS, PROTOCOL_SIGNATURES, get_frequency_band,
    TARGET_OBJECTS,
)

__all__ = [
    "SAMPLE_RATE", "BUFFER_SIZE", "GAIN_DEFAULT", "RSSI_THRESHOLD",
    "SCAN_RANGES", "GPIO_PINS",
    "WIFI_CHANNELS_2_4", "WIFI_CHANNELS_5", "WIFI_CHANNEL_WIDTH",
    "BAND_PROFILES", "get_band_profile",
    "PROTOCOL_BANDS", "PROTOCOL_SIGNATURES", "get_frequency_band",
    "TARGET_OBJECTS",
]