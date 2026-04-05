import numpy as np

def get_fft_spectrum(audio_data, sample_rate=16000):
    n = len(audio_data)

    window = np.hanning(n)
    windowed_data = audio_data * window

    fft_result = np.fft.rfft(windowed_data)
    magnitude = np.abs(fft_result)

    magnitude += 1e-6
    np.log10(magnitude, out=magnitude)
    magnitude *= 20.0

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return {
        "frequencies": freqs,
        "magnitudes_db": magnitude
    }