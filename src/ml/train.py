import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
from tqdm import tqdm
from python_speech_features import mfcc
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, Dense, TimeDistributed, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

models_dir = current_dir.parent / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

clean_data_path = project_root / 'Dataset' / 'Clean_Audio'
csv_path = current_dir / 'drone_dataset.csv'


class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
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


config = Config(mode='conv')


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
        print("Знайдено старий кеш. Видаляю його автоматично для оновлення даних...")
        try:
            config.p_path.unlink()
        except Exception as e:
            print(f"Не вдалося видалити кеш: {e}. Спробуй видалити {config.p_path.name} вручну!")

    X, y = [], []
    prob_dist = class_dist / class_dist.sum()
    n_samples = int((df['length'].sum() / 0.1) * 2)

    print(f"Генеруємо {n_samples} семплів (витягуємо MFCC)...")

    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)

        available_files = df[df.label == rand_class]['fname'].values
        if len(available_files) == 0:
            continue

        file = np.random.choice(available_files)
        file_path = clean_data_path / rand_class / file

        try:
            rate, wav = wavfile.read(file_path)

            if wav.shape[0] <= config.step:
                continue

            rand_index = np.random.randint(0, wav.shape[0] - config.step)
            sample = wav[rand_index:rand_index + config.step]

            X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)

            config.min = min(np.amin(X_sample), config.min)
            config.max = max(np.amax(X_sample), config.max)

            X.append(X_sample)
            y.append(classes.index(rand_class))

        except Exception as e:
            continue

    if len(X) == 0:
        raise ValueError("Не вдалося згенерувати жодного семпла! Перевірте аудіофайли.")

    X, y = np.array(X), np.array(y)
    X = (X - config.min) / (config.max - config.min)

    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    y = to_categorical(y, num_classes=len(classes))

    print(f"Зберігаємо свіжий кеш ознак у {config.p_path.name}...")
    with open(config.p_path, 'wb') as f:
        pickle.dump((X, y, config), f, protocol=2)

    return (X, y, config)


def main():
    print("Читаємо дані...")
    df = pd.read_csv(csv_path)
    df.set_index('fname', inplace=True)

    if 'length' not in df.columns:
        df['length'] = 0.0

    print("Рахуємо реальну довжину очищених файлів...")
    files_to_drop = []

    for f in tqdm(list(df.index)):
        label = df.at[f, 'label']
        file_path = clean_data_path / label / f

        try:
            rate, signal = wavfile.read(file_path)
            df.at[f, 'length'] = signal.shape[0] / rate
        except (FileNotFoundError, ValueError):
            files_to_drop.append(f)

    if files_to_drop:
        print(f"Видалено {len(files_to_drop)} недоступних або пошкоджених файлів з таблиці.")
        df.drop(files_to_drop, inplace=True)

    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()
    df.reset_index(inplace=True)

    X, y, current_config = build_rand_feat(df, classes, class_dist)

    print("Балансуємо класи та збираємо модель...")
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

    print("Починаємо тренування (10 епох)...")
    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        shuffle=True,
        class_weight=class_weights_dict,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    print(f"Тренування завершено! Найкращу модель збережено як {current_config.model_path.name}")


if __name__ == '__main__':
    main()