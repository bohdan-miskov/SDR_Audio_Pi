import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path
import numpy as np
import time

# Налаштування
FS = 16000  # Наша стандартна частота
SECONDS = 8  # Тривалість запису
COPIES = 50   # Кількість клонів для датасету

print("🎙️ УВАГА! Приготуйся ввімкнути відео дрона на YouTube.")
print("Почнемо запис через 3 секунди...")
sd.sleep(3000)

print("🔴 ЗАПИС ПІШОВ! Вмикай звук (8 секунд)...")
recording = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1, dtype=np.int16)
sd.wait()
print("✅ Запис завершено!")

# Зберігаємо
dataset_dir = Path("dataset/Clean_Audio/background noise") # Змінили папку
dataset_dir.mkdir(parents=True, exist_ok=True)

# Генеруємо унікальний ідентифікатор для цього запуску (на основі часу)
run_id = int(time.time())

print(f"Клонуємо файл {COPIES} разів у датасет (Run ID: {run_id})...")
for i in range(COPIES):
    # Тепер назви будуть унікальними, наприклад: youtube_hack_1684321_0.wav
    filename = dataset_dir / f"youtube_hack_{run_id}_{i}.wav"
    write(filename, FS, recording)

print("🎉 Готово! Тепер запускай тренування: python -m src.ml.train")