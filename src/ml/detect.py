import pickle
import numpy as np
from python_speech_features import mfcc
from keras.models import load_model


class Config:
    pass


class DroneDetector:
    def __init__(self, model_path='models/conv.keras', pickle_path='models/conv.p'):
        with open(pickle_path, 'rb') as handle:
            self.config = pickle.load(handle)[2]

        self.model = load_model(model_path, compile=False)
        self.classes = ['background noise', 'drone']

    def predict(self, audio_chunk, rate=16000):
        batch_x = []

        for i in range(0, len(audio_chunk) - self.config.step, self.config.step):
            sample = audio_chunk[i:i + self.config.step]

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