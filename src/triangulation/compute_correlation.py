import numpy as np
from scipy import signal

def get_delays(audio_channels, sample_rate=16000):
    if len(audio_channels.shape) < 2 or audio_channels.shape[1] < 2:
        raise ValueError("Для тріангуляції потрібно мінімум 2 канали звуку (стерео)")

    num_channels = audio_channels.shape[1]
    ref_signal = audio_channels[:, 0] 
    delays = []

    for i in range(1, num_channels):
        target_signal = audio_channels[:, i]
        
        corr = signal.correlate(target_signal, ref_signal, mode='full')

        lag = np.argmax(corr) - (len(ref_signal) - 1)
        
        time_delay = lag / sample_rate
        delays.append(time_delay)
        
        print(f"[TDOA] Затримка Мік {i} відносно Мік 0: {time_delay:.6f} сек")

    return delays