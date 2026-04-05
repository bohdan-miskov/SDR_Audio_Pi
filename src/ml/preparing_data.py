import numpy as np
import librosa
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd
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


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

base_data_path = project_root / 'Dataset' / 'Audio'
clean_data_path = project_root / 'Dataset' / 'Clean_Audio'
csv_path = current_dir / 'drone_dataset.csv'

df = pd.read_csv(csv_path)

classes = df['label'].unique()

for c in classes:
    (clean_data_path / c).mkdir(parents=True, exist_ok=True)

print("Починаю очищення та збереження файлів...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    label = row['label']
    f = row['fname']

    src_path = base_data_path / label / f
    dst_path = clean_data_path / label / f

    if not src_path.exists():
        continue

    try:
        signal, rate = librosa.load(str(src_path), sr=TARGET_RATE)

        mask = envelope(signal, rate, THRESHOLD)
        clean_signal = signal[mask]

        if len(clean_signal) > 0:
            clean_signal_int16 = np.int16(clean_signal * 32767)
            wavfile.write(str(dst_path), rate, clean_signal_int16)

    except Exception as e:
        print(f"Помилка з файлом {f}: {e}")

print(f"Готово! Очищені файли збережено в {clean_data_path}")