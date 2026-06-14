import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy.io import wavfile
from tqdm import tqdm
from python_speech_features import mfcc
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, Dense, TimeDistributed, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AudioTrainer")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

models_dir = current_dir.parent / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

clean_data_path = project_root / 'dataset' 

class Config:
    def __init__(self, mode='convn', nfilt=26, nfeat=13, nfft=8192, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate / 10)
        self.model_path = models_dir / f'{mode}.keras'
        self.p_path = models_dir / f'{mode}.p'
        self.min = float('inf')
        self.max = -float('inf')

config = Config(mode='convn')

def get_conv_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        MaxPool2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_rand_feat(df, classes, class_dist):
    if config.p_path.exists():
        logger.info("Знайдено старий кеш ознак. Видаляю його для оновлення даних...")
        try:
            config.p_path.unlink()
        except Exception as e:
            logger.warning(f"Не вдалося автоматично видалити кеш: {e}. Спробуй видалити {config.p_path.name} вручну.")

    X, y = [], []
    target_shape = None
    prob_dist = class_dist / class_dist.sum()
    n_samples = int((df['length'].sum() / 0.1) * 2)

    logger.info(f"Генеруємо {n_samples} семплів (витягуємо MFCC спектрограми)...")

    for _ in tqdm(range(n_samples), desc="Обробка аудіо"):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)

        available_files = df[df.label == rand_class]['fname'].values
        if len(available_files) == 0:
            continue

        file = np.random.choice(available_files)
        file_path = clean_data_path / rand_class / file

        try:
            rate, wav = wavfile.read(file_path)

            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)

            if wav.shape[0] <= config.step:
                continue

            rand_index = np.random.randint(0, wav.shape[0] - config.step)
            sample = wav[rand_index:rand_index + config.step].astype(np.float32)

            if np.random.rand() > 0.5:
                noise_amp = 0.05 * np.random.uniform(0, 1) * np.amax(np.abs(sample) + 1e-6)
                sample = sample + noise_amp * np.random.normal(size=sample.shape[0])

            X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)

            if target_shape is None:
                target_shape = X_sample.shape

            if X_sample.shape != target_shape:
                if X_sample.shape[0] > target_shape[0]:
                    X_sample = X_sample[:target_shape[0], :]
                else:
                    pad_width = target_shape[0] - X_sample.shape[0]
                    X_sample = np.pad(X_sample, ((0, pad_width), (0, 0)), mode='constant')

            X.append(X_sample)
            y.append(classes.index(rand_class))

        except Exception as e:
            logger.debug(f"Помилка обробки файлу {file}: {e}")
            continue

    if len(X) == 0:
        logger.error("Не вдалося згенерувати жодного семпла! Перевірте наявність аудіофайлів.")
        raise ValueError("Порожній датасет.")

    X, y = np.array(X), np.array(y)
    config.min = np.min(X)
    config.max = np.max(X)
    X = (X - config.min) / (config.max - config.min)

    if config.mode.startswith('conv'):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    y = to_categorical(y, num_classes=len(classes))

    logger.info(f"Зберігаємо свіжий кеш ознак у {config.p_path.name}...")
    with open(config.p_path, 'wb') as f:
        pickle.dump((X, y, config), f, protocol=2)

    return (X, y, config)

def main():
    logger.info("Скануємо папки з аудіофайлами...")
    
    data = []
    classes = []
    
    if not clean_data_path.exists():
        logger.error(f"Папку {clean_data_path} не знайдено! Створіть 'dataset/Clean_Audio' в корені проєкту.")
        return

    for class_dir in clean_data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            classes.append(class_name)
            
            for wav_file in class_dir.glob('*.wav'):
                try:
                    rate, signal = wavfile.read(wav_file)
                    length = signal.shape[0] / rate
                    data.append({'fname': wav_file.name, 'label': class_name, 'length': length})
                except Exception as e:
                    logger.warning(f"Не вдалося прочитати пошкоджений файл {wav_file.name}: {e}")
    
    if not data:
        logger.error("Не знайдено жодного .wav файлу у папках. Завантажте дані для тренування.")
        return

    df = pd.DataFrame(data)
    df.set_index('fname', inplace=True)
    
    logger.info(f"Знайдено {len(df)} валідних файлів. Виявлені класи: {classes}")

    class_dist = df.groupby(['label'])['length'].mean()
    df.reset_index(inplace=True)

    X, y, current_config = build_rand_feat(df, classes, class_dist)

    logger.info("Балансуємо ваги класів та компілюємо архітектуру моделі...")
    y_flat = np.argmax(y, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
    class_weights_dict = dict(enumerate(class_weights))

    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model(input_shape, len(classes))

    checkpoint = ModelCheckpoint(
        filepath=str(current_config.model_path),
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True
    )

    logger.info("Запуск процесу тренування (10 епох)...")
    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        shuffle=True,
        class_weight=class_weights_dict,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    logger.info(f"Тренування успішно завершено! Найкращу модель збережено як {current_config.model_path.name}")

if __name__ == '__main__':
    main()