import numpy as np
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path

TARGET_RATE = 16000
THRESHOLD = 0.005

def envelope(y, rate, threshold):
    y_abs = np.abs(y)
    window_len = int(rate / 10)
    window = np.ones(window_len) / window_len
    y_mean = np.convolve(y_abs, window, mode='same')
    mask = y_mean > threshold
    return mask

# Вказуємо шлях прямо до твоєї папки 'Dataset'
# Вказуємо шлях прямо до твоєї папки 'Dataset'
# Піднімаємось на 2 рівні вгору від папки src/ml до кореня проєкту
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
dataset_path = project_root / 'Dataset'

print(f"Починаю перевірку файлів у папці: {dataset_path}")

# Шукаємо ВСІ wav файли у підпапках (drone та background noise)
wav_files = list(dataset_path.rglob('*.wav'))

if not wav_files:
    print("❌ Помилка: Не знайдено жодного .wav файлу. Перевір, чи лежать файли саме в папках drone та background noise.")
else:
    for file_path in tqdm(wav_files, desc="Обробка аудіо"):
        try:
            # Завантажуємо файл, примусово робимо 16000 Гц і Моно
            signal, rate = librosa.load(str(file_path), sr=TARGET_RATE, mono=True)

            # ВАЖЛИВО: Застосовуємо видалення тиші ТІЛЬКИ для дронів!
            if file_path.parent.name == 'drone':
                mask = envelope(signal, rate, THRESHOLD)
                clean_signal = signal[mask]
            else:
                # Для фонового шуму (background noise) нічого не відрізаємо
                clean_signal = signal 

            # Якщо після очищення хоч щось залишилося
            if len(clean_signal) > 0:
                # Переводимо у правильний формат int16
                clean_signal_int16 = np.int16(np.clip(clean_signal * 32767, -32768, 32767))
                
                # Перезаписуємо той самий файл ідеальним стандартизованим звуком
                wavfile.write(str(file_path), rate, clean_signal_int16)
            else:
                print(f"\nПопередження: Файл {file_path.name} виявився абсолютно порожнім після обрізки тиші.")

        except Exception as e:
            print(f"\nПомилка з файлом {file_path.name}: {e}")

    print(f"\n🎉 Готово! Всі {len(wav_files)} файлів приведено до стандарту: 16000 Гц, int16.")