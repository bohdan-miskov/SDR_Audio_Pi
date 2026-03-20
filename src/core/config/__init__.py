"""
core/config — публічний API конфігураційного модуля.
Реекспортує всі константи та хелпери для зворотної сумісності.
"""
from .sdr_settings import SAMPLE_RATE, BUFFER_SIZE, GAIN_DEFAULT, RSSI_THRESHOLD
from .scan_settings import SCAN_RANGES, GPIO_PINS
from .wifi_profiles import (
    WIFI_CHANNELS_2_4, WIFI_CHANNELS_5, WIFI_CHANNEL_WIDTH,
    BAND_PROFILES, get_band_profile,
)
from .protocol_signatures import PROTOCOL_BANDS, PROTOCOL_SIGNATURES, get_frequency_band
from .target_objects import TARGET_OBJECTS

__all__ = [
    # sdr_settings
    "SAMPLE_RATE", "BUFFER_SIZE", "GAIN_DEFAULT", "RSSI_THRESHOLD",
    # scan_settings
    "SCAN_RANGES", "GPIO_PINS",
    # wifi_profiles
    "WIFI_CHANNELS_2_4", "WIFI_CHANNELS_5", "WIFI_CHANNEL_WIDTH",
    "BAND_PROFILES", "get_band_profile",
    # protocol_signatures
    "PROTOCOL_BANDS", "PROTOCOL_SIGNATURES", "get_frequency_band",
    # target_objects
    "TARGET_OBJECTS",
]
