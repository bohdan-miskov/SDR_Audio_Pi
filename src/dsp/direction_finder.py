"""
DirectionFinder — гібридний пеленгатор для кільцевого масиву антен (UCA).

Геометрія:
    N антен рівномірно по колу радіусом R = ARRAY_RADIUS_M:
        φ_k = 2π·k/N  для k = 0..N-1

Алгоритм (три рівні, результат об'єднується Kalman filter):

    Рівень 1 — RSS Beamscanning (завжди):
        Порівнює потужність сигналу по антенах.
        Точність: ±30–45°. Не потребує когерентності.

    Рівень 2 — GCC-PHAT TDOA (головний):
        Generalized Cross-Correlation Phase Transform.
        Вимірює затримку між записами одного пакету на різних антенах.
        Точність: ±5–15° при SNR > 10 dB.

    Рівень 3 — Sequential MUSIC (уточнення):
        Активується тільки при confidence TDOA > 0.3.
        Точність: ±10–20°.

    Kalman Filter:
        Об'єднує результати трьох рівнів з урахуванням
        довіри кожного методу. Стабілізує трек між кадрами.
"""
import numpy as np
from scipy.linalg import eigh

from core.config import ARRAY_RADIUS_M, N_ANTENNAS, SAMPLE_RATE

_C = 3e8


# ══════════════════════════════════════════════════════════════════════════════
#  Kalman Filter для азимуту
# ══════════════════════════════════════════════════════════════════════════════

class _AzimuthKalman:
    """
    Простий Kalman filter для азимуту.
    Стан: [азимут_рад, кутова_швидкість_рад/с]
    """

    def __init__(self):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1000.0
        self.Q = np.diag([0.01, 0.1])          # шум процесу
        self.H = np.array([[1.0, 0.0]])

        # Шум вимірювань для кожного методу (дисперсія в рад²)
        self._R = {
            'rss':   np.radians(40.0) ** 2,
            'tdoa':  np.radians(12.0) ** 2,
            'music': np.radians(18.0) ** 2,
        }

        self.initialized = False
        self._last_t: float | None = None

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def predict(self, dt: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, azimuth_deg: float, method: str,
               confidence: float) -> float:
        import time

        az_rad = np.radians(azimuth_deg)

        if not self.initialized:
            self.x[0] = az_rad
            self.initialized = True
            self._last_t = time.perf_counter()
            return azimuth_deg

        now = time.perf_counter()
        dt = min(now - self._last_t, 1.0)
        self._last_t = now

        self.predict(dt)

        # Масштабуємо шум по впевненості: низька впевненість → більший шум
        r = self._R[method] / max(confidence, 0.05)
        R = np.array([[r]])

        y = np.array([self._wrap(az_rad - self.x[0])])
        S = self.H @ self.P @ self.H.T + R
        K = (self.P @ self.H.T / S[0, 0]).flatten()  # гарантуємо shape (2,)

        self.x = self.x + K * float(y[0])
        self.x[0] = self._wrap(self.x[0])
        self.P = (np.eye(2) - np.outer(K, self.H)) @ self.P
        return float(np.degrees(float(self.x[0]))) % 360.0

    def uncertainty_deg(self) -> float:
        return float(np.degrees(np.sqrt(max(self.P[0, 0], 0.0))))


# ══════════════════════════════════════════════════════════════════════════════
#  DirectionFinder
# ══════════════════════════════════════════════════════════════════════════════

class DirectionFinder:
    """
    Гібридний пеленгатор: RSS + GCC-PHAT TDOA + Sequential MUSIC + Kalman.

    Використання:
        finder = DirectionFinder(center_freq=433e6)
        bearing = finder.estimate(antenna_iq)   # {'ANT_A': iq, ...}
        print(f"Азимут: {bearing:.1f}°  ±{finder.uncertainty:.1f}°")
    """

    SCAN_STEPS: int = 360

    def __init__(self, center_freq: float = 433e6):
        self._freq: float = center_freq
        self._radius: float = ARRAY_RADIUS_M
        self._n: int = N_ANTENNAS
        self._sr: float = float(SAMPLE_RATE)

        # Кути антен: φ_k = 2π·k/N
        self._phi = np.array([2 * np.pi * k / self._n
                               for k in range(self._n)])

        # Позиції антен у 2D (для TDOA)
        self._positions = np.column_stack([
            self._radius * np.cos(self._phi),
            self._radius * np.sin(self._phi),
        ])  # (N, 2)

        self._lambda: float = _C / center_freq
        self._phase_const: float = 2 * np.pi * self._radius / self._lambda
        self._scan_angles = np.linspace(0, 2 * np.pi,
                                        self.SCAN_STEPS, endpoint=False)
        self._steering_matrix = self._build_steering_matrix()

        # Kalman filter
        self._kf = _AzimuthKalman()

        # Результати
        self._bearing_deg: float = 0.0
        self._confidence: float = 0.0
        self._uncertainty_deg: float = 90.0
        self._power_pattern: np.ndarray = np.zeros(self.SCAN_STEPS)
        self._last_method: str = 'rss'

        print(
            f"[DF] UCA: N={self._n}, R={self._radius*100:.0f}cm, "
            f"f={center_freq/1e6:.0f}MHz, λ={self._lambda*100:.1f}cm"
        )

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def bearing(self) -> float:
        return self._bearing_deg

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def uncertainty(self) -> float:
        """Оцінка похибки кута в градусах (з Kalman)."""
        return self._uncertainty_deg

    @property
    def power_pattern(self) -> np.ndarray:
        return self._power_pattern

    @property
    def scan_angles_deg(self) -> np.ndarray:
        return np.degrees(self._scan_angles)

    @property
    def last_method(self) -> str:
        """Який метод використовувався в останньому estimate()."""
        return self._last_method

    def set_frequency(self, freq_hz: float) -> None:
        self._freq = freq_hz
        self._lambda = _C / freq_hz
        self._phase_const = 2 * np.pi * self._radius / self._lambda
        self._steering_matrix = self._build_steering_matrix()

    # ── Public API ──────────────────────────────────────────────────────────

    def estimate(self, antenna_iq: dict[str, np.ndarray]) -> float | None:
        """
        Головний метод пеленгації.

        Args:
            antenna_iq: {'ANT_A': iq_array, ..., 'ANT_E': iq_array}
                        — виходить з AmplitudeSyncDetector.process()

        Returns:
            Азимут (°), 0..360, або None якщо даних недостатньо.
        """
        ant_names = [f'ANT_{chr(65 + k)}' for k in range(self._n)]
        iq_list = [antenna_iq.get(name) for name in ant_names]

        if any(arr is None or len(arr) == 0 for arr in iq_list):
            return None

        min_len = min(len(arr) for arr in iq_list)
        if min_len < 32:
            return None

        iq_matrix = np.array([arr[:min_len] for arr in iq_list],
                              dtype=np.complex128)  # (N, T)

        # ── Рівень 1: RSS (завжди) ──────────────────────────────────────
        rss_angle, rss_conf = self._rss_beamscanning(iq_matrix)

        # ── Рівень 2: GCC-PHAT TDOA (головний) ─────────────────────────
        tdoa_angle, tdoa_conf = self._gcc_phat_tdoa(iq_matrix)

        # ── Рівень 3: Sequential MUSIC (якщо TDOA впевнений) ───────────
        if tdoa_conf > 0.25:
            music_angle, music_conf = self._sequential_music(iq_matrix)
        else:
            music_angle, music_conf = rss_angle, 0.1

        # ── Kalman Fusion ───────────────────────────────────────────────
        self._kf.update(rss_angle,   'rss',   rss_conf)
        self._kf.update(tdoa_angle,  'tdoa',  tdoa_conf)
        fused = self._kf.update(music_angle, 'music', music_conf)

        self._bearing_deg = fused
        self._uncertainty_deg = self._kf.uncertainty_deg()

        # Впевненість — максимум з трьох методів
        self._confidence = float(np.clip(
            max(rss_conf, tdoa_conf, music_conf), 0.0, 1.0
        ))

        # Який метод був найнадійнішим
        best = max(
            [('rss', rss_conf), ('tdoa', tdoa_conf), ('music', music_conf)],
            key=lambda x: x[1]
        )
        self._last_method = best[0]

        return self._bearing_deg

    def estimate_music(self, antenna_iq: dict[str, np.ndarray],
                       n_snapshots: int = 16) -> float | None:
        """
        Сумісність з попереднім API — делегує до estimate().
        """
        return self.estimate(antenna_iq)

    # ── Рівень 1: RSS Beamscanning ──────────────────────────────────────────

    def _rss_beamscanning(
            self, iq_matrix: np.ndarray) -> tuple[float, float]:
        """
        Порівнює потужність по антенах.
        Повертає: (азимут_deg, confidence 0..1)
        """
        powers = np.mean(np.abs(iq_matrix) ** 2, axis=1)
        best_ant = int(np.argmax(powers))
        angle_deg = float(np.degrees(self._phi[best_ant])) % 360.0

        mean_p = float(np.mean(powers))
        max_p = float(np.max(powers))
        conf = float(np.clip((max_p / (mean_p + 1e-12) - 1) / (self._n - 1),
                             0.0, 1.0))

        # Оновлюємо power_pattern для GUI
        self._power_pattern = np.abs(
            self._steering_matrix.conj().T @
            (iq_matrix.mean(axis=1) /
             (np.abs(iq_matrix.mean(axis=1)) + 1e-12))
        ) ** 2

        return angle_deg, conf

    # ── Рівень 2: GCC-PHAT TDOA ─────────────────────────────────────────────

    def _gcc_phat(self, sig_i: np.ndarray,
                  sig_j: np.ndarray) -> float:
        """
        Generalized Cross-Correlation Phase Transform.
        Повертає затримку між sig_i та sig_j у секундах.
        """
        N = len(sig_i)
        n_fft = 2 * N

        Si = np.fft.fft(sig_i, n=n_fft)
        Sj = np.fft.fft(sig_j, n=n_fft)

        G = Si * np.conj(Sj)
        # PHAT: нормалізуємо по амплітуді — залишається тільки фаза
        G_phat = G / (np.abs(G) + 1e-10)

        cc = np.real(np.fft.ifft(G_phat))

        # Максимальна можлива затримка = час пробігу між найдальшими антенами
        max_delay_s = 2 * self._radius / _C
        max_delay_samp = max(4, int(np.ceil(max_delay_s * self._sr)) + 2)
        max_delay_samp = min(max_delay_samp, N // 4)

        # Шукаємо пік у вікні [-max_delay, +max_delay]
        cc_pos = cc[:max_delay_samp + 1]
        cc_neg = cc[n_fft - max_delay_samp:]

        if np.max(np.abs(cc_pos)) >= np.max(np.abs(cc_neg)):
            peak_idx = int(np.argmax(np.abs(cc_pos)))
            delay_samp = peak_idx
        else:
            peak_idx = int(np.argmax(np.abs(cc_neg)))
            delay_samp = peak_idx - len(cc_neg)

        return float(delay_samp) / self._sr

    def _gcc_phat_tdoa(
            self, iq_matrix: np.ndarray) -> tuple[float, float]:
        """
        Обчислює затримки антена_0 vs всі інші через GCC-PHAT,
        потім методом найменших квадратів знаходить азимут.
        Повертає: (азимут_deg, confidence 0..1)
        """
        delays = {}
        for j in range(1, self._n):
            delays[j] = self._gcc_phat(iq_matrix[0], iq_matrix[j])

        # Grid search: для кожного кута рахуємо суму ймовірностей
        scores = np.zeros(self.SCAN_STEPS)
        for idx, theta in enumerate(self._scan_angles):
            direction = np.array([np.cos(theta), np.sin(theta)])
            score = 0.0
            for j, delay_meas in delays.items():
                # Геометрична затримка між антеною 0 та антеною j
                delta_pos = self._positions[j] - self._positions[0]
                delay_geom = float(np.dot(delta_pos, direction)) / _C

                # Gaussian likelihood
                sigma = 1.5 / self._sr   # ~1.5 семпли
                score += np.exp(
                    -0.5 * ((delay_meas - delay_geom) / sigma) ** 2
                )
            scores[idx] = score

        peak_idx = int(np.argmax(scores))
        azimuth_deg = float(np.degrees(self._scan_angles[peak_idx])) % 360.0

        # Параболічна інтерполяція
        refined = self._parabolic_peak(scores, peak_idx)
        azimuth_deg = float(np.degrees(refined)) % 360.0

        # Впевненість
        mean_s = float(np.mean(scores))
        peak_s = float(scores[peak_idx])
        conf = float(np.clip((peak_s / (mean_s + 1e-12) - 1) /
                             (self._n * 3), 0.0, 1.0))

        return azimuth_deg, conf

    # ── Рівень 3: Sequential MUSIC ──────────────────────────────────────────

    def _sequential_music(
            self, iq_matrix: np.ndarray,
            n_sources: int = 1) -> tuple[float, float]:
        """
        Sequential MUSIC з просторовою кореляційною матрицею.
        Повертає: (азимут_deg, confidence 0..1)
        """
        M, T = iq_matrix.shape

        # Псевдокогерентна кореляційна матриця
        R = np.zeros((M, M), dtype=np.complex128)
        for i in range(M):
            for j in range(M):
                R[i, j] = np.mean(iq_matrix[i] * np.conj(iq_matrix[j]))
        R = (R + R.conj().T) / 2

        # Власний розклад
        eigvals, eigvecs = eigh(R)  # відсортовані за зростанням

        # Шумовий підпростір
        noise_subspace = eigvecs[:, :M - n_sources]
        En = noise_subspace @ noise_subspace.conj().T

        # MUSIC pseudo-spectrum
        power_music = np.zeros(self.SCAN_STEPS)
        for i in range(self.SCAN_STEPS):
            a = self._steering_matrix[:, i]
            denom = float(np.real(a.conj() @ En @ a))
            power_music[i] = 1.0 / (abs(denom) + 1e-12)

        peak_idx = int(np.argmax(power_music))
        refined = self._parabolic_peak(power_music, peak_idx)
        azimuth_deg = float(np.degrees(refined)) % 360.0

        mean_p = float(np.mean(power_music))
        peak_p = float(power_music[peak_idx])
        conf = float(np.clip((peak_p / (mean_p + 1e-12) - 1) /
                             (M - 1) / 10, 0.0, 1.0))

        return azimuth_deg, conf

    # ── Допоміжні ───────────────────────────────────────────────────────────

    def _build_steering_matrix(self) -> np.ndarray:
        phase = self._phase_const * np.cos(
            self._phi[:, np.newaxis] - self._scan_angles[np.newaxis, :]
        )
        return np.exp(1j * phase)

    def _parabolic_peak(self, power: np.ndarray, idx: int) -> float:
        n = len(power)
        il = (idx - 1) % n
        ir = (idx + 1) % n
        yl, yc, yr = power[il], power[idx], power[ir]
        denom = 2 * (2 * yc - yl - yr)
        if abs(denom) < 1e-12:
            return self._scan_angles[idx]
        delta = (yl - yr) / denom
        step = 2 * np.pi / self.SCAN_STEPS
        return self._scan_angles[idx] + delta * step
