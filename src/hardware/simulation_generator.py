"""
SimulationGenerator — генератор тестових IQ-сигналів.

Режими:
  antenna_mux=False: суцільний IQ-шум із вбудованими сигналами (для FFT-відображення).
  antenna_mux=True : імітує GPIO-перемикання антен.
      Буфер = [ANT_0][ANT_1]...[ANT_N][BLANK][ANT_0]...
      BLANK = повні нулі — ідеально чіткий провал для AmplitudeSyncDetector.
      Кожна антена має фазовий зсув CW-сигналу відповідно до напрямку джерела.
"""
import numpy as np
import random
import time
from scipy.fft import fft, ifft, fftfreq, next_fast_len

from core.config import (
    SAMPLE_RATE, BUFFER_SIZE,
    N_ANTENNAS, ANTENNA_STEP_MS, ARRAY_RADIUS_M,
)

_C = 3e8  # швидкість світла (м/с)


class SimulationGenerator:
    """
    Генерує IQ-семпли для симуляції PlutoSDR.

    Args:
        antenna_mux:     True → BLANK-маркери + фазові зсуви між антенами.
        sim_azimuth_deg: Азимут симульованого джерела сигналу (0–360°).
        signal_freq_offset_hz: Зсув несучої у базовій смузі (Гц), за замовчуванням 50 кГц.
        snr_db:          Відношення сигнал/шум у дБ для CW-сигналу.
    """

    def __init__(
        self,
        antenna_mux: bool = True,
        sim_azimuth_deg: float = 45.0,
        signal_freq_offset_hz: float = 50_000.0,
        snr_db: float = 20.0,
    ):
        self._antenna_mux = antenna_mux
        self._sim_azimuth_deg = float(sim_azimuth_deg)
        self._freq_offset = float(signal_freq_offset_hz)
        self._snr_db = float(snr_db)

        # Параметри антенного циклу
        self._n_antennas: int = N_ANTENNAS
        self._step_samples: int = int(SAMPLE_RATE * ANTENNA_STEP_MS / 1000)
        self._cycle_len: int = self._step_samples * (self._n_antennas + 1)

        # Кількість повних кроків що вміщаються в буфер
        self._n_steps: int = BUFFER_SIZE // self._step_samples

        # Кути антен рівномірно по колу
        self._ant_angles_rad: list[float] = [
            2 * np.pi * k / self._n_antennas for k in range(self._n_antennas)
        ]

        # Для backward-compatible FFT-режиму (antenna_mux=False)
        self._targets = [
            {"freq": 433e6,  "type": "LORA 433",  "bw": 0.2e6,  "power": 25, "style": "hop"},
            {"freq": 868e6,  "type": "ELRS 868",  "bw": 0.5e6,  "power": 30, "style": "hop"},
            {"freq": 915e6,  "type": "CROSS 915", "bw": 0.5e6,  "power": 30, "style": "hop"},
            {"freq": 1200e6, "type": "VIDEO 1.2", "bw": 8.0e6,  "power": 20, "style": "plateau"},
            {"freq": 2400e6, "type": "WIFI 2.4",  "bw": 20.0e6, "power": 15, "style": "arch"},
            {"freq": 5800e6, "type": "VIDEO 5.8", "bw": 14.0e6, "power": 25, "style": "plateau"},
        ]
        self._hop_states: dict[int, dict] = {
            i: {"last_hop": 0.0, "curr_freq": t["freq"]}
            for i, t in enumerate(self._targets)
        }
        self._N_fft_buf = next_fast_len(BUFFER_SIZE)

        print(
            f"[SimGen] antenna_mux={antenna_mux}, "
            f"step={self._step_samples}, cycle={self._cycle_len}, "
            f"n_steps={self._n_steps} ({self._n_steps / (self._n_antennas + 1):.1f} cycles/buf)"
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def targets(self) -> list[dict]:
        return list(self._targets)

    @property
    def antenna_mux(self) -> bool:
        return self._antenna_mux

    @antenna_mux.setter
    def antenna_mux(self, value: bool) -> None:
        self._antenna_mux = bool(value)

    @property
    def sim_azimuth_deg(self) -> float:
        return self._sim_azimuth_deg

    @sim_azimuth_deg.setter
    def sim_azimuth_deg(self, value: float) -> None:
        self._sim_azimuth_deg = float(value) % 360.0

    # ── Public API ───────────────────────────────────────────────────────────

    def get_iq_samples(self, center_freq: float, gain_level: float) -> np.ndarray:
        if self._antenna_mux:
            return self._get_mux_buffer(center_freq, gain_level)
        return self._get_plain_buffer(center_freq, gain_level, self._N_fft_buf)

    # ── Private: antenna-mux mode ────────────────────────────────────────────

    def _get_mux_buffer(self, center_freq: float, gain_level: float) -> np.ndarray:
        """
        Генерує буфер з чергуванням антенних кроків та BLANK.

        CW-сигнал генерується ТІЛЬКИ якщо center_freq близько до одноїз предвизначених цілей.
        Інакше — лише шум (немає детекції на порожніх діапазонах).
        """
        s = self._step_samples
        buf = np.zeros(BUFFER_SIZE, dtype=np.complex64)

        # --- Чи є реальна ціль на цій частоті? ---
        half_bw = SAMPLE_RATE / 2.0   # половина вікна SDR
        near_targets = [
            t for t in self._targets
            if abs(center_freq - t["freq"]) < half_bw
        ]

        # --- Параметри сигналу ---
        noise_amp = 0.001 * (10 ** (gain_level / 200.0))
        signal_amp = noise_amp * (10 ** (self._snr_db / 20.0)) if near_targets else 0.0

        # Фазові зсуви між антенами
        azimuth_rad = np.deg2rad(self._sim_azimuth_deg)
        phase_shifts = [
            2 * np.pi * center_freq * ARRAY_RADIUS_M
            * np.cos(phi - azimuth_rad) / _C
            for phi in self._ant_angles_rad
        ]

        t = np.arange(s, dtype=np.float32) / SAMPLE_RATE
        n_per_cycle = self._n_antennas + 1

        for step_idx in range(self._n_steps):
            pos = step_idx * s
            role = step_idx % n_per_cycle

            if role == self._n_antennas:
                # BLANK: нулі (buf вже np.zeros)
                pass
            else:
                ant_idx = role
                phi = phase_shifts[ant_idx]
                noise = noise_amp * (
                    np.random.normal(0, 1, s).astype(np.float32)
                    + 1j * np.random.normal(0, 1, s).astype(np.float32)
                )
                if signal_amp > 0:
                    cw = signal_amp * np.exp(
                        1j * (2 * np.pi * self._freq_offset * t + phi)
                    ).astype(np.complex64)
                    buf[pos:pos + s] = (cw + noise).astype(np.complex64)
                else:
                    buf[pos:pos + s] = noise.astype(np.complex64)

        # Заповнюємо хвіст (якщо є) першим антенним кроком (щоб не було зони нулів)
        tail_start = self._n_steps * s
        if tail_start < BUFFER_SIZE:
            tail_len = BUFFER_SIZE - tail_start
            phi = phase_shifts[0]
            t_tail = np.arange(tail_len, dtype=np.float32) / SAMPLE_RATE
            cw_tail = signal_amp * np.exp(
                1j * (2 * np.pi * self._freq_offset * t_tail + phi)
            ).astype(np.complex64)
            noise_tail = noise_amp * (
                np.random.normal(0, 1, tail_len).astype(np.float32)
                + 1j * np.random.normal(0, 1, tail_len).astype(np.float32)
            )
            buf[tail_start:] = (cw_tail + noise_tail).astype(np.complex64)

        return buf

    # ── Private: plain IQ (для FFT-відображення без мукса) ───────────────────

    def _get_plain_buffer(
        self, center_freq: float, gain_level: float, n_fft: int
    ) -> np.ndarray:
        """Класичний генератор спектру через IFFT (для режиму antenna_mux=False)."""
        freq_bins = fftfreq(n_fft, d=1.0 / SAMPLE_RATE) + center_freq
        base_noise_amp = 0.0001 * (10 ** (gain_level / 100.0))
        spectrum = (
            np.random.normal(0, 1, n_fft)
            + 1j * np.random.normal(0, 1, n_fft)
        ) * base_noise_amp

        for i, t in enumerate(self._targets):
            if abs(center_freq - t["freq"]) > 60e6:
                continue
            sig_amp = base_noise_amp * (10 ** (t["power"] / 20.0))
            if t["style"] == "hop":
                state = self._hop_states[i]
                if time.time() - state["last_hop"] > 0.1:
                    state["last_hop"] = time.time()
                    state["curr_freq"] = t["freq"] + random.randint(-5, 5) * 1e6
                mask = np.abs(freq_bins - state["curr_freq"]) < t["bw"] / 2
            elif t["style"] in ("plateau", "arch"):
                mask = np.abs(freq_bins - t["freq"]) < t["bw"] / 2
            else:
                continue
            if not np.any(mask):
                continue
            n = int(np.sum(mask))
            noise_chunk = (
                np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
            ) * sig_amp
            if t["style"] == "arch":
                indices = np.where(mask)[0]
                x = (freq_bins[indices] - t["freq"]) / (t["bw"] / 2)
                noise_chunk *= np.sqrt(np.maximum(0, 1 - x ** 2))
            spectrum[mask] += noise_chunk

        result = ifft(spectrum)
        return result[:BUFFER_SIZE].astype(np.complex64)
