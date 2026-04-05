import numpy as np


class GoertzelFilter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def get_power(self, audio_data, target_freq):
        n = len(audio_data)
        k = int(0.5 + (n * target_freq) / self.sample_rate)
        w = (2.0 * np.pi / n) * k
        coeff = 2.0 * np.cos(w)

        q1, q2 = 0.0, 0.0

        for x in audio_data.tolist():
            q0 = coeff * q1 - q2 + x
            q2 = q1
            q1 = q0

        return q1 ** 2 + q2 ** 2 - coeff * q1 * q2

    def analyze_drone_harmonics(self, audio_data, base_freqs):
        results = {}

        avg_power = np.mean(np.square(audio_data)) + 1e-6

        baseline = avg_power * len(audio_data)

        for freq in base_freqs:
            pwr = self.get_power(audio_data, freq)
            results[freq] = pwr / baseline

        return results