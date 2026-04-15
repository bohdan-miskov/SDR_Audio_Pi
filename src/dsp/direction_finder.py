"""
DirectionFinder — пеленгатор для кільцевого масиву 5 антен (UCA).

Геометрія:
    5 антен рівномірно по колу радіусом R = ARRAY_RADIUS_M:
        φ_k = 2π·k/5  для k = 0, 1, 2, 3, 4
        (0°, 72°, 144°, 216°, 288°)

Фізика:
    Сигнал з напрямку θ (азимут від 0° до 360°) досягає антени k з
    фазовим зсувом:
        ψ_k = (2π·R / λ) · cos(φ_k − θ)
    де λ = c / f_center — довжина хвилі.

Алгоритм (Steering Vector Scan — перебір напрямків):
    1. Для кожної антени k вимірюємо середню фазу: ψ̂_k = angle(mean(IQ_k))
    2. Будуємо steering-вектор a(θ) = exp(j·ψ_k(θ)) для кожного θ
    3. Знаходимо θ що максимізує |<ψ̂, a(θ)>|² (скалярний добуток)
       → це еквівалент метрики "просторового степенювання"
    4. Уточнюємо через MUSIC-like пошук навколо максимуму

Точність (теоретична при SNR > 10 dB):
    На частоті 433 МГц (λ ≈ 69 см) з R = 10 см:
        2πR/λ ≈ 0.91 рад (помірна роздільна здатність)
    Очікувана точність: ±5..10°

Для покращення точності:
    - Збільшити R або кількість антен
    - Застосувати MUSIC/ESPRIT замість Steering Vector Scan
"""
import numpy as np
from scipy.signal import correlate

from core.config import ARRAY_RADIUS_M, N_ANTENNAS

# Швидкість світла (м/с)
_C = 3e8


class DirectionFinder:
    """
    Розраховує азимут (кут приходу сигналу) з синхронізованих IQ-даних
    кільцевого масиву 5 антен.

    Використання:
        finder = DirectionFinder(center_freq=433e6)
        bearing = finder.estimate(antenna_iq)   # {'ANT_A': iq, ...}
        print(f"Азимут: {bearing:.1f}°")
    """

    # Роздільна здатність пошуку (кроків на оберт)
    SCAN_STEPS: int = 360

    def __init__(self, center_freq: float = 433e6):
        """
        Args:
            center_freq: Центральна частота прийому (Гц).
        """
        self._freq: float = center_freq
        self._radius: float = ARRAY_RADIUS_M
        self._n: int = N_ANTENNAS

        # Кути антен у радіанах: φ_k = 2π·k/N
        self._phi: np.ndarray = np.array(
            [2 * np.pi * k / self._n for k in range(self._n)]
        )

        # Довжина хвилі та просторова фазова константа
        self._lambda: float = _C / center_freq
        self._phase_const: float = 2 * np.pi * self._radius / self._lambda

        # Попередньо обчислені steering-вектори для всіх кутів
        self._scan_angles: np.ndarray = np.linspace(0, 2 * np.pi, self.SCAN_STEPS, endpoint=False)
        self._steering_matrix: np.ndarray = self._build_steering_matrix()

        # Результати останнього estimate()
        self._bearing_deg: float = 0.0
        self._confidence: float = 0.0
        self._power_pattern: np.ndarray = np.zeros(self.SCAN_STEPS)

        print(
            f"[DF] UCA: N={self._n}, R={self._radius*100:.0f}cm, "
            f"f={center_freq/1e6:.0f}MHz, lam={self._lambda*100:.1f}cm, "
            f"2piR/lam={self._phase_const:.3f}rad"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def bearing(self) -> float:
        """Останній обчислений азимут (°), 0..360."""
        return self._bearing_deg

    @property
    def confidence(self) -> float:
        """Впевненість 0..1 (відношення піку до середнього у просторовому спектрі)."""
        return self._confidence

    @property
    def power_pattern(self) -> np.ndarray:
        """Просторовий спектр потужності (масив довжиною SCAN_STEPS)."""
        return self._power_pattern

    @property
    def scan_angles_deg(self) -> np.ndarray:
        """Кути сканування (°), відповідає індексам power_pattern."""
        return np.degrees(self._scan_angles)

    def set_frequency(self, freq_hz: float) -> None:
        """Оновлює центральну частоту та перебудовує steering-матрицю."""
        self._freq = freq_hz
        self._lambda = _C / freq_hz
        self._phase_const = 2 * np.pi * self._radius / self._lambda
        self._steering_matrix = self._build_steering_matrix()

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(self, antenna_iq: dict[str, np.ndarray]) -> float | None:
        """
        Обчислює азимут за синхронізованими IQ-даними антен.

        Args:
            antenna_iq: {'ANT_A': iq_array, 'ANT_B': ..., ..., 'ANT_E': ...}
                        — виходить з AmplitudeSyncDetector.process()

        Returns:
            Азимут у градусах (0..360) або None якщо даних недостатньо.
        """
        # Збираємо IQ по порядку антен
        ant_names = [f'ANT_{chr(65 + k)}' for k in range(self._n)]
        iq_list = [antenna_iq.get(name) for name in ant_names]

        if any(arr is None or len(arr) == 0 for arr in iq_list):
            return None

        # Вирівнюємо довжини (беремо мінімум)
        min_len = min(len(arr) for arr in iq_list)
        iq_matrix = np.array([arr[:min_len] for arr in iq_list], dtype=np.complex128)

        # Метод: Steering Vector Scan (з двома уточненнями)
        bearing = self._steering_scan(iq_matrix)
        self._bearing_deg = bearing
        return bearing

    # ── Private: Algorithm ────────────────────────────────────────────────────

    def _build_steering_matrix(self) -> np.ndarray:
        """
        Будує steering-матрицю [N_antennas × SCAN_STEPS].
        steering[k, i] = exp(j · (2πR/λ) · cos(φ_k − scan_angles[i]))
        """
        # broadcast: phi (N,1) - angles (1,S) → (N, S)
        phase = self._phase_const * np.cos(
            self._phi[:, np.newaxis] - self._scan_angles[np.newaxis, :]
        )
        return np.exp(1j * phase)

    def _steering_scan(self, iq_matrix: np.ndarray) -> float:
        """
        Scanning Correlator:
          1. Усереднений вектор кожної антени → snapshot вектор x̄
          2. Просторовий спектр: P(θ) = |a†(θ) · x̄|²
          3. Пік P(θ) → кут
          4. Параболічна інтерполяція навколо піку для субрезолюційної точності

        Для надійності берем декілька snapshot-ів і усереднюємо спектр.
        """
        # Snapshot-вектор: середнє IQ по часу для кожної антени
        # Нормуємо по модулю щоб вилучити амплітудні несиметрії антен
        snapshot = np.mean(iq_matrix, axis=1)
        norms = np.abs(snapshot)
        norms[norms < 1e-12] = 1.0
        snapshot_norm = snapshot / norms   # тільки фазова інформація

        # Просторовий спектр P(θ) = |steering† · snapshot|²
        # steering_matrix: (N, S), snapshot_norm: (N,)
        power = np.abs(self._steering_matrix.conj().T @ snapshot_norm) ** 2
        self._power_pattern = power

        # Метрика впевненості: пік / середнє (чим вище — тим чіткіший пік)
        mean_p = float(np.mean(power))
        peak_p = float(np.max(power))
        self._confidence = float(np.clip((peak_p / mean_p - 1) / (self._n - 1), 0, 1)) \
            if mean_p > 1e-12 else 0.0

        # Грубий пік
        peak_idx = int(np.argmax(power))

        # Параболічна інтерполяція для точності між кутовими кроками
        refined_angle = self._parabolic_peak(power, peak_idx)
        bearing_deg = float(np.degrees(refined_angle)) % 360.0
        return bearing_deg

    def _parabolic_peak(self, power: np.ndarray, idx: int) -> float:
        """
        Уточнює положення піку між дискретними кутами через параболу через 3 точки.
        Повертає уточнений кут у радіанах.
        """
        n = len(power)
        il = (idx - 1) % n
        ir = (idx + 1) % n
        yl, yc, yr = power[il], power[idx], power[ir]

        denom = 2 * (2 * yc - yl - yr)
        if abs(denom) < 1e-12:
            return self._scan_angles[idx]

        delta = (yl - yr) / denom           # зсув від піку у частках кроку
        step = 2 * np.pi / self.SCAN_STEPS
        return self._scan_angles[idx] + delta * step

    # ── Advanced: MUSIC (опційно) ──────────────────────────────────────────────

    def estimate_music(self, antenna_iq: dict[str, np.ndarray],
                       n_snapshots: int = 16) -> float | None:
        """
        MUSIC (MUltiple SIgnal Classification) — вища точність за рахунок
        декомпозиції коваріаційної матриці на сигнальний та шумовий підпростори.

        Рекомендується при SNR > 5 dB та достатній кількості семплів.
        """
        ant_names = [f'ANT_{chr(65 + k)}' for k in range(self._n)]
        iq_list = [antenna_iq.get(name) for name in ant_names]

        if any(arr is None or len(arr) < n_snapshots for arr in iq_list):
            return None

        min_len = min(len(a) for a in iq_list)
        iq_matrix = np.array([a[:min_len] for a in iq_list], dtype=np.complex128)

        # Розбиваємо на snapshot-блоки
        snap_size = min_len // n_snapshots
        X = np.column_stack([
            iq_matrix[:, i * snap_size:(i + 1) * snap_size].mean(axis=1)
            for i in range(n_snapshots)
        ])  # (N, n_snapshots)

        # Коваріаційна матриця R = X·X† / n_snapshots
        R = (X @ X.conj().T) / n_snapshots

        # Власний розклад (eigenvalues у зростаючому порядку)
        eigvals, eigvecs = np.linalg.eigh(R)

        # Шумовий підпростір: всі вектори крім 1 (припускаємо 1 джерело)
        noise_subspace = eigvecs[:, :-1]   # (N, N-1)

        # MUSIC псевдо-спектр: 1 / |a†(θ) · En · En† · a(θ)|
        En = noise_subspace @ noise_subspace.conj().T   # (N, N)
        power_music = np.zeros(self.SCAN_STEPS)
        for i, angle in enumerate(self._scan_angles):
            a = self._steering_matrix[:, i]
            denom = np.abs(a.conj() @ En @ a)
            power_music[i] = 1.0 / (denom + 1e-12)

        self._power_pattern = power_music
        peak_idx = int(np.argmax(power_music))
        refined = self._parabolic_peak(power_music, peak_idx)
        bearing = float(np.degrees(refined)) % 360.0
        self._bearing_deg = bearing

        mean_p = float(np.mean(power_music))
        self._confidence = float(np.clip(
            (float(np.max(power_music)) / mean_p - 1) / (self._n - 1), 0, 1
        ))
        return bearing
