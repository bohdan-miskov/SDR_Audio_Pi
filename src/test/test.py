#!/usr/bin/env python3
"""
ШВИДКИЙ ТЕСТ PlutoSDR
Запуск: python quick_test.py
"""

import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal.windows import blackman
import matplotlib.pyplot as plt

print("=" * 60)
print("ШВИДКИЙ ТЕСТ PlutoSDR")
print("=" * 60)

# Імпорт бібліотеки
try:
    import adi
    print("✓ Бібліотека pyadi-iio доступна")
except ImportError:
    print("✗ Помилка: pyadi-iio не встановлена")
    print("Встановіть: pip install pyadi-iio")
    exit(1)

# Підключення
print("\n1. Підключення до PlutoSDR...")
try:
    sdr = adi.Pluto("ip:192.168.2.1")
    print("   ✓ З'єднання встановлено")
except Exception as e:
    print(f"   ✗ Помилка підключення: {e}")
    print("\n   Перевірте:")
    print("   - PlutoSDR підключений до USB")
    print("   - Виконайте: ping 192.168.2.1")
    print("   - Перезавантажте PlutoSDR")
    exit(1)

# Налаштування
print("\n2. Налаштування параметрів...")
sdr.sample_rate = 10000000
sdr.rx_buffer_size = 16384
sdr.rx_enabled_channels = [0]
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 60

# Тест на різних частотах
test_freqs = [
    (433e6, "433 MHz (LoRa)"),
    (2450e6, "2450 MHz (Wi-Fi)"),
    (868e6, "868 MHz (ELRS)")
]

results = []

for freq, name in test_freqs:
    print(f"\n3. Тест частоти: {name}")
    sdr.rx_lo = int(freq)
    
    # Очищення буфера
    for _ in range(3):
        _ = sdr.rx()
    
    # Реальне читання
    iq_raw = sdr.rx()
    
    print(f"   Dtype: {iq_raw.dtype}")
    print(f"   Shape: {iq_raw.shape}")
    
    # Нормалізація
    if np.issubdtype(iq_raw.dtype, np.integer):
        iq = iq_raw.astype(np.complex64) / 2048.0
    else:
        iq = iq_raw.astype(np.complex64)
    
    # Статистика
    power = np.mean(np.abs(iq)**2)
    print(f"   Потужність: {power:.6e}")
    
    # FFT (scipy.fft — швидше та точніше)
    window = blackman(len(iq))
    fft_data = fftshift(fft(iq * window))
    psd = 10 * np.log10(np.abs(fft_data)**2 + 1e-15)
    
    noise = np.median(psd)
    peak = np.max(psd)
    dynamic_range = peak - noise
    
    print(f"   Шум: {noise:.1f} dB")
    print(f"   Пік: {peak:.1f} dB")
    print(f"   Динамічний діапазон: {dynamic_range:.1f} dB")
    
    # Оцінка
    if dynamic_range < 10:
        status = "⚠️ ПРОБЛЕМА (плоский спектр)"
    elif dynamic_range < 30:
        status = "⚠️ СЛАБКИЙ СИГНАЛ"
    else:
        status = "✓ НОРМАЛЬНО"
    
    print(f"   Оцінка: {status}")
    
    results.append({
        'freq': freq,
        'name': name,
        'psd': psd,
        'noise': noise,
        'peak': peak,
        'dynamic': dynamic_range,
        'status': status
    })

# Підсумок
print("\n" + "=" * 60)
print("ПІДСУМОК")
print("=" * 60)

all_ok = True
for r in results:
    print(f"{r['name']:20s} -> {r['status']}")
    if "ПРОБЛЕМА" in r['status']:
        all_ok = False

if all_ok:
    print("\n✓ ВСЕ ПРАЦЮЄ! PlutoSDR готовий до роботи.")
else:
    print("\n⚠️ ВИЯВЛЕНІ ПРОБЛЕМИ!")
    print("\nМожливі рішення:")
    print("1. Збільшіть gain: sdr.rx_hardwaregain_chan0 = 70")
    print("2. Перевірте антену (має бути підключена до RX порту)")
    print("3. Оновіть прошивку PlutoSDR")
    print("4. Спробуйте інший USB кабель")

# Графік (якщо matplotlib встановлено)
try:
    print("\n4. Малюємо графіки...")
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
    if len(results) == 1:
        axes = [axes]
    
    for ax, r in zip(axes, results):
        freqs_mhz = np.linspace(-5, 5, len(r['psd'])) + r['freq']/1e6
        ax.plot(freqs_mhz, r['psd'], 'cyan', linewidth=0.5)
        ax.axhline(r['noise'], color='red', linestyle='--', label=f'Noise: {r["noise"]:.1f} dB')
        ax.axhline(r['noise'] + 18, color='yellow', linestyle='--', label='Threshold: +18 dB')
        ax.set_title(f'{r["name"]} - {r["status"]}')
        ax.set_xlabel('Частота (MHz)')
        ax.set_ylabel('Потужність (dB)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('pluto_test_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Графік збережено: pluto_test_results.png")
    plt.show()
    
except ImportError:
    print("   (matplotlib не встановлено, пропускаємо графіки)")

print("\nГотово!")