import pickle
import numpy as np
from python_speech_features import mfcc
from keras.models import load_model
from scipy.signal import butter, lfilter  # <--- Додай цей імпорт


class Config:
    pass


class DroneDetector:
    def __init__(self, model_path='models/conv.keras', pickle_path='models/conv.p'):
        with open(pickle_path, 'rb') as handle:
            self.config = pickle.load(handle)[2]

        self.model = load_model(model_path, compile=False)
        self.classes = ['background noise', 'drone']

    # --- ДОДАЄМО ФУНКЦІЮ ФІЛЬТРАЦІЇ ---
    def highpass_filter(self, data, cutoff=100.0, fs=16000, order=4):
        """Відрізає низькочастотний гул (вітер, кроки), залишаючи високі частоти (дрон)."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    # ----------------------------------

    def predict(self, audio_chunk, rate=16000):
        # 1. Очищаємо аудіо від вітру перед аналізом
        clean_audio = self.highpass_filter(audio_chunk, cutoff=100.0, fs=rate)

        batch_x = []

        # 2. Нарізаємо вже ОЧИЩЕНИЙ звук
        for i in range(0, len(clean_audio) - self.config.step, self.config.step):
            sample = clean_audio[i:i + self.config.step]

            x = mfcc(sample, rate, numcep=self.config.nfeat, nfilt=self.config.nfilt, nfft=self.config.nfft)
            x = (x - self.config.min) / (self.config.max - self.config.min)
            batch_x.append(x)

        if not batch_x:
            return {"class": "unknown", "confidence": 0.0}

        X_batch = np.array(batch_x)
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)

        y_prob = self.model.predict(X_batch, verbose=0)
        avg_prob = np.mean(y_prob, axis=0).flatten()
        pred_idx = np.argmax(avg_prob)

        confidence = float(avg_prob[pred_idx] * 100)

        return {
            "class": self.classes[pred_idx],
            "confidence": round(confidence, 2)
        }