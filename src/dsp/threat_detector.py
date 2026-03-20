"""
ThreatDetector — детекція цільових об'єктів та протоколів.
Generates PyQt5 signals via callback to avoid Qt dependency in pure DSP logic.
"""
import time
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from core.config import (
    RSSI_THRESHOLD, TARGET_OBJECTS, PROTOCOL_SIGNATURES,
)


class ThreatDetector:
    """
    Аналізує спектр на наявність відомих загроз:
    - Мульти-частотні об'єкти (гребінка частот TARGET_OBJECTS)
    - Підтверджені сигнали протоколів (WiFi, ELRS тощо)

    Не залежить від Qt напряму: результати передає через callback.
    Callback підпис: callback(message: str, color: str)
    """

    # Час у секундах між повторними алертами для одного об'єкта / протоколу
    OBJ_REPEAT_INTERVAL: float = 10.0
    ALERT_COOLDOWN: float = 2.0
    WIFI_COOLDOWN: float = 5.0
    FREQ_HISTORY_WINDOW: float = 30.0

    def __init__(self, alert_callback):
        """
        alert_callback(msg: str, color: str) — функція для відправки алертів.
        """
        self._alert = alert_callback

        # Час останнього виявлення для кожної цільової частоти
        self._freq_history: dict[float, float] = {
            f: 0.0
            for obj in TARGET_OBJECTS.values()
            for f in obj
        }

        # Час останнього алерту для кожного об'єкта
        self._last_obj_alert: dict[str, float] = {
            name: 0.0 for name in TARGET_OBJECTS
        }

        # Persistence-лічильники для протоколів (ключ: int(center_freq/1000)*1000)
        self._persistence_db: dict[int, np.ndarray] = {}

        # Стан для WiFi-алертів
        self._last_alert: float = 0.0
        self._last_wifi_alert: float = 0.0
        self._last_wifi_channels: list = []
        self._mask_warned: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, freqs: np.ndarray, psd: np.ndarray,
                noise_floor: float, center_freq: float,
                mask_enabled: bool,
                current_profile: dict | None,
                channel_activity: dict[int, float],
                detected_protocol: str | None,
                detected_bandwidth: float,
                detected_power: float) -> None:
        """
        Повний аналіз загроз за один тік.
        """
        self._detect_target_objects(freqs, psd, center_freq, mask_enabled)
        self._detect_protocols(
            freqs, psd, noise_floor, center_freq,
            current_profile, channel_activity,
            detected_protocol, detected_bandwidth, detected_power
        )

    # ── Private: Object detection ─────────────────────────────────────────────

    def _detect_target_objects(self, freqs: np.ndarray, psd: np.ndarray,
                                center_freq: float, mask_enabled: bool) -> None:
        if not mask_enabled and not self._mask_warned:
            self._mask_warned = True
            print("[!] УВАГА: Для найкращої детекції цілей встановіть МАСКУ (без генератора).")
            print("[!] Увімкнено режим пошуку по формі гребінки (без маски).")

        now = time.time()
        freq_step = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1000.0
        half_bw = max(10, int(4e6 / freq_step))
        margin = 5e6

        for target_f in self._freq_history:
            if not (freqs[0] - margin) <= target_f <= (freqs[-1] + margin):
                continue

            closest_idx = int(np.argmin(np.abs(freqs - target_f)))
            start_idx = max(0, closest_idx - half_bw)
            end_idx = min(len(psd), closest_idx + half_bw + 1)

            zone_psd = psd[start_idx:end_idx]
            if len(zone_psd) < 15:
                continue

            # scipy.ndimage.uniform_filter1d — швидше за np.convolve
            smoothed = uniform_filter1d(zone_psd.astype(float), size=3)
            local_base = float(np.percentile(smoothed, 10))
            power_above = float(np.max(smoothed)) - local_base

            # scipy.signal.find_peaks: знаходимо зубці гребінки
            # prominence — висота піку над сусідніми точками (2.5 dB)
            peaks_idx, _ = find_peaks(smoothed, prominence=2.5)
            peaks_count = len(peaks_idx)

            if mask_enabled:
                is_target = power_above > 3.0
                reason = f"+{power_above:.0f}dB (Mask)"
            else:
                is_target = (peaks_count >= 3) and (power_above >= 3.5)
                reason = f"+{power_above:.0f}dB, {peaks_count} зубців"

            if is_target:
                if self._freq_history[target_f] == 0 or \
                        (now - self._freq_history[target_f] > self.FREQ_HISTORY_WINDOW):
                    self._alert(
                        f" Ціль: {target_f / 1e6:.1f} МГц ({reason})", "#8888ff"
                    )
                    print(f"[*] Ціль: {target_f / 1e6:.0f} МГц ({reason}) center={center_freq / 1e6:.0f}")
                self._freq_history[target_f] = now

        # Перевірка комбінацій
        for obj_name, obj_freqs in TARGET_OBJECTS.items():
            ready = all(
                now - self._freq_history[f] <= self.FREQ_HISTORY_WINDOW
                for f in obj_freqs
            )
            if ready and (now - self._last_obj_alert[obj_name] > self.OBJ_REPEAT_INTERVAL):
                self._last_obj_alert[obj_name] = now
                self._alert(
                    f"🚨 ДЕТЕКЦІЯ: {obj_name} (Всі 3 частоти активні!)", "#ff0000"
                )

    # ── Private: Protocol detection ───────────────────────────────────────────

    def _detect_protocols(self, freqs: np.ndarray, psd: np.ndarray,
                           noise_floor: float, center_freq: float,
                           current_profile: dict | None,
                           channel_activity: dict[int, float],
                           proto: str | None,
                           bw: float, power: float) -> None:
        thresh = noise_floor + RSSI_THRESHOLD
        raw_mask = psd > thresh
        raw_mask[:5] = False
        raw_mask[-5:] = False

        # Persistence
        scan_key = int(center_freq / 1000) * 1000
        if scan_key not in self._persistence_db:
            self._persistence_db[scan_key] = np.zeros_like(raw_mask, dtype=int)

        p_counter = self._persistence_db[scan_key]
        if len(p_counter) != len(raw_mask):
            p_counter = np.zeros_like(raw_mask, dtype=int)

        hits = raw_mask.astype(int)
        p_counter = p_counter + hits - (1 - hits) * 2
        p_counter = np.clip(p_counter, 0, 15)
        self._persistence_db[scan_key] = p_counter

        confirmed_mask = p_counter > 5
        if np.sum(confirmed_mask) < 5:
            return

        idxs = np.where(confirmed_mask)[0]
        if len(idxs) < 2:
            return

        now = time.time()
        if now - self._last_alert < self.ALERT_COOLDOWN:
            return

        if current_profile and current_profile.get('channels') and channel_activity:
            # WiFi-режим
            active_channels = sorted(channel_activity.keys())
            changed = active_channels != self._last_wifi_channels
            time_ok = now - self._last_wifi_alert > self.WIFI_COOLDOWN

            if changed or time_ok:
                if proto and proto in PROTOCOL_SIGNATURES:
                    sig = PROTOCOL_SIGNATURES[proto]
                    icon, color, name = sig['icon'], sig['color'], sig['name']
                else:
                    icon, color, name = "", "#55ff55", "WiFi"

                ch_str = (
                    ', '.join(f"CH{ch}" for ch in active_channels)
                    if len(active_channels) <= 3
                    else f"{len(active_channels)}ch"
                )
                msg = f"{icon} {name} | {ch_str} | {power:.0f}dB"
                self._last_alert = now
                self._last_wifi_alert = now
                self._last_wifi_channels = active_channels
                self._alert(msg, color)

        elif proto and proto in PROTOCOL_SIGNATURES:
            sig = PROTOCOL_SIGNATURES[proto]
            msg = (
                f"{sig['icon']} {sig['name']} | "
                f"{center_freq / 1e6:.1f}MHz | "
                f"BW:{bw / 1e6:.1f}MHz | {power:.0f}dB"
            )
            self._last_alert = now
            self._alert(msg, sig['color'])

        elif proto:
            if bw < 2e6:
                msg = f" Narrowband | {center_freq / 1e6:.1f}MHz | BW:{bw / 1e3:.0f}kHz"
                color = "#ff5555"
            else:
                msg = f" Wideband | {center_freq / 1e6:.1f}MHz | BW:{bw / 1e6:.1f}MHz"
                color = "#ffaa00"
            self._last_alert = now
            self._alert(msg, color)
