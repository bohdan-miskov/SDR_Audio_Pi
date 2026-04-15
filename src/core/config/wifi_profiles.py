# --- WiFi канали та профілі діапазонів ---

# WiFi 2.4 ГГц канали (канал: центральна частота в Гц)
WIFI_CHANNELS_2_4 = {
    1: 2412e6, 2: 2417e6, 3: 2422e6, 4: 2427e6, 5: 2432e6,
    6: 2437e6, 7: 2442e6, 8: 2447e6, 9: 2452e6, 10: 2457e6,
    11: 2462e6, 12: 2467e6, 13: 2472e6, 14: 2484e6,
}

# WiFi 5 ГГц канали (основні UNII діапазони)
WIFI_CHANNELS_5 = {
    # UNII-1 (5150–5250 MHz)
    36: 5180e6, 40: 5200e6, 44: 5220e6, 48: 5240e6,
    # UNII-2A (5250–5350 MHz)
    52: 5260e6, 56: 5280e6, 60: 5300e6, 64: 5320e6,
    # UNII-2C (5470–5725 MHz)
    100: 5500e6, 104: 5520e6, 108: 5540e6, 112: 5560e6,
    116: 5580e6, 120: 5600e6, 124: 5620e6, 128: 5640e6,
    132: 5660e6, 136: 5680e6, 140: 5700e6, 144: 5720e6,
    # UNII-3 (5725–5850 MHz)
    149: 5745e6, 153: 5765e6, 157: 5785e6, 161: 5805e6, 165: 5825e6,
}

# Стандартна ширина WiFi-каналу
WIFI_CHANNEL_WIDTH = 22e6

# Профілі діапазонів (для автоматичного визначення режиму каналів)
BAND_PROFILES = {
    'wifi_2_4': {
        'name': 'WiFi 2.4 GHz',
        'start': 2400e6,
        'stop': 2500e6,
        'channels': WIFI_CHANNELS_2_4,
        'channel_width': 22e6,
    },
    'wifi_5': {
        'name': 'WiFi 5 GHz',
        'start': 5150e6,
        'stop': 5850e6,
        'channels': WIFI_CHANNELS_5,
        'channel_width': 20e6,
    },
}


def get_band_profile(center_freq: float) -> dict | None:
    """Повертає профіль діапазону за центральною частотою (Гц)."""
    for _name, prof in BAND_PROFILES.items():
        if prof['start'] <= center_freq <= prof['stop']:
            return prof
    return None
