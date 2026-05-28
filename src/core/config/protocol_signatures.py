# --- Сигнатури протоколів та частотні діапазони ---

# Частотні діапазони для класифікації протоколів
PROTOCOL_BANDS = {
    'lora_433':   (430e6,  440e6),
    'elrs_868':   (860e6,  875e6),
    'elrs_915':   (900e6,  930e6),
    'wifi_2_4':   (2400e6, 2500e6),
    'wifi_5':     (5150e6, 5350e6),
    'analog_5_8': (5650e6, 5950e6),
    'analog_1_2': (1200e6, 1400e6),
}

# Сигнатури протоколів дронів
# Поля: name, icon, color, bw_min/bw_max (Гц), bands (список ключів PROTOCOL_BANDS)
PROTOCOL_SIGNATURES = {
    'DJI_WIFI': {
        'name': 'DJI WiFi',
        'icon': '🚁',
        'color': '#00ff00',
        'bw_min': 8e6,
        'bw_max': 25e6,
        'bands': ['wifi_2_4', 'wifi_5'],
        'description': 'DJI OcuSync / WiFi control',
    },
    'ELRS': {
        'name': 'ELRS',
        'icon': '📡',
        'color': '#ff8800',
        'bw_min': 0.3e6,
        'bw_max': 2.5e6,
        'bands': ['elrs_868', 'elrs_915', 'wifi_2_4'],
        'description': 'ExpressLRS control link',
    },
    'CROSSFIRE': {
        'name': 'Crossfire',
        'icon': '📡',
        'color': '#ff4400',
        'bw_min': 0.3e6,
        'bw_max': 1.5e6,
        'bands': ['elrs_868', 'elrs_915'],
        'description': 'TBS Crossfire control link',
    },
    'ANALOG_VIDEO': {
        'name': 'Analog Video',
        'icon': '📺',
        'color': '#ff00ff',
        'bw_min': 6e6,
        'bw_max': 20e6,
        'bands': ['analog_5_8', 'analog_1_2'],
        'description': 'Analog FPV video transmitter',
    },
    'LORA': {
        'name': 'LoRa',
        'icon': '📶',
        'color': '#00ffff',
        'bw_min': 0.1e6,
        'bw_max': 0.5e6,
        'bands': ['lora_433'],
        'description': 'LoRa telemetry or control',
    },
}


def get_frequency_band(center_freq: float) -> str | None:
    """Визначає назву діапазону (ключ PROTOCOL_BANDS) за частотою (Гц)."""
    for band_name, (start, stop) in PROTOCOL_BANDS.items():
        if start <= center_freq <= stop:
            return band_name
    return None
