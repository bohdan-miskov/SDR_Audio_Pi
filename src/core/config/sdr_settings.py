# --- Базові параметри SDR-приймача ---

SAMPLE_RATE = 30000000   # 30 MSPS
BUFFER_SIZE = 262144     # Розмір буфера IQ: 6.2 циклів при step=0.2ms, 6 антен
GAIN_DEFAULT = 50        # Посилення за замовчуванням, dB

# Поріг виявлення сигналу (дБ над рівнем шуму)
RSSI_THRESHOLD = 15

# Зміщення для кодування PSD (dB) у uint8:
# value_uint8 = clip(round(dB + DB_OFFSET), 0, 255)
# Декодування: dB = value_uint8 - DB_OFFSET
# Діапазон покриття: [-100, +155] dBm → [0, 255]
DB_OFFSET = 100
