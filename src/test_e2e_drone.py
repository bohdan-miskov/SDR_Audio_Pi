"""
test_e2e_drone.py — E2E тест пеленгації дрона.

Симулює реалістичний сценарій: дрон(и) летять з відомих напрямків,
передають на відомих частотах — система повинна визначити кут.

Сценарії:
  1. Один дрон, чистий ефір
  2. Дрон + WiFi-завада з іншого боку
  3. Мульти-частотний дрон (433 + 868 + 5800 МГц)
  4. Дрон що рухається (0 → 180°)
  5. Слабкий сигнал (далекий дрон, SNR = 0..5 dB)
  6. Два дрони з різних напрямків на різних частотах

Запуск: python test_e2e_drone.py
"""
import sys
import numpy as np

sys.path.insert(0, '.')

from core.config import SAMPLE_RATE, N_ANTENNAS, ARRAY_RADIUS_M, ANTENNA_STEP_MS
from dsp.direction_finder import DirectionFinder
from dsp.amplitude_sync import AmplitudeSyncDetector

C = 3e8
STEP_SAMPLES = int(SAMPLE_RATE * ANTENNA_STEP_MS / 1000)
CYCLE_SAMPLES = STEP_SAMPLES * (N_ANTENNAS + 1)  # 5 ant + 1 blank
N_CYCLES = 6
TOTAL_SAMPLES = CYCLE_SAMPLES * N_CYCLES


# ══════════════════════════════════════════════════════════════════════════════
# DroneSimulator — генерує IQ для усього пайплайну (5 антен + BLANK маркери)
# ══════════════════════════════════════════════════════════════════════════════

class DroneSimulator:
    """
    Симулює радіо-ефір для кільцевого масиву 5 антен.

    Використання:
        sim = DroneSimulator(center_freq=433e6)
        sim.add_drone(angle_deg=135, freq_hz=433e6, power_db=20, bw_hz=200e3)
        sim.add_interference(angle_deg=270, freq_hz=433.5e6, power_db=10)
        iq_buffer = sim.generate()  # -> np.ndarray з BLANK-маркерами
    """

    def __init__(self, center_freq: float = 433e6, noise_floor_db: float = -30):
        self._center_freq = center_freq
        self._noise_floor_db = noise_floor_db
        self._sources: list[dict] = []

    def add_drone(self, angle_deg: float, freq_hz: float,
                  power_db: float = 20, bw_hz: float = 200e3,
                  label: str = "DRONE") -> None:
        """Додає дрон (сигнал з певного напрямку)."""
        self._sources.append({
            'angle_deg': angle_deg,
            'freq_hz': freq_hz,
            'power_db': power_db,
            'bw_hz': bw_hz,
            'label': label,
        })

    def add_interference(self, angle_deg: float, freq_hz: float,
                         power_db: float = 15, bw_hz: float = 5e6,
                         label: str = "INTERF") -> None:
        """Додає завадне джерело (WiFi, TV, тощо)."""
        self._sources.append({
            'angle_deg': angle_deg,
            'freq_hz': freq_hz,
            'power_db': power_db,
            'bw_hz': bw_hz,
            'label': label,
        })

    def clear(self) -> None:
        """Очищає всі джерела."""
        self._sources.clear()

    def generate(self, n_cycles: int = N_CYCLES) -> np.ndarray:
        """
        Generates full IQ buffer with BLANK markers.

        Structure per cycle:
            [ANT_A: step][ANT_B: step]...[ANT_E: step][BLANK: step]

        During BLANK - only noise * 0.02 (deep dip).
        Uses global time axis and pre-generated modulation for coherence.
        """
        total = CYCLE_SAMPLES * n_cycles
        buf = np.zeros(total, dtype=np.complex128)

        # Global time axis for continuous phase
        t_global = np.arange(total) / SAMPLE_RATE

        noise_amp = 10 ** (self._noise_floor_db / 20)

        # Pre-generate modulation signals (shared across all antennas)
        # This models reality: all antennas hear the SAME signal waveform,
        # only spatial phase differs.
        src_modulations = []
        for src in self._sources:
            if src['bw_hz'] > 0:
                mod_freq = src['bw_hz'] / 2
                # One continuous random walk for the entire buffer
                phase_walk = np.cumsum(np.random.randn(total)) / SAMPLE_RATE
                mod = np.exp(1j * 2 * np.pi * mod_freq * phase_walk)
            else:
                mod = None
            src_modulations.append(mod)

        for cycle in range(n_cycles):
            cycle_start = cycle * CYCLE_SAMPLES

            for ant_idx in range(N_ANTENNAS):
                seg_start = cycle_start + ant_idx * STEP_SAMPLES
                seg_end = seg_start + STEP_SAMPLES

                if seg_end > total:
                    break

                phi_k = 2 * np.pi * ant_idx / N_ANTENNAS
                t_seg = t_global[seg_start:seg_end]

                # Base noise (independent per antenna — this is correct)
                noise = noise_amp * (np.random.randn(STEP_SAMPLES)
                                     + 1j * np.random.randn(STEP_SAMPLES))
                antenna_signal = noise.copy()

                # Add each source with continuous phase & shared modulation
                for si, src in enumerate(self._sources):
                    theta = np.radians(src['angle_deg'])
                    lam = C / src['freq_hz']
                    phase_const = 2 * np.pi * ARRAY_RADIUS_M / lam

                    spatial_phase = phase_const * np.cos(phi_k - theta)
                    sig_amp = 10 ** (src['power_db'] / 20)
                    freq_offset = src['freq_hz'] - self._center_freq

                    # Continuous carrier using global time
                    carrier = np.exp(1j * (2 * np.pi * freq_offset * t_seg + spatial_phase))

                    # Apply SHARED modulation (same waveform for all antennas)
                    if src_modulations[si] is not None:
                        carrier = carrier * src_modulations[si][seg_start:seg_end]

                    antenna_signal += sig_amp * carrier

                buf[seg_start:seg_end] = antenna_signal

            # BLANK step
            blank_start = cycle_start + N_ANTENNAS * STEP_SAMPLES
            blank_end = blank_start + STEP_SAMPLES
            if blank_end <= total:
                blank_noise = 0.02 * noise_amp * (
                    np.random.randn(STEP_SAMPLES)
                    + 1j * np.random.randn(STEP_SAMPLES)
                )
                buf[blank_start:blank_end] = blank_noise

        return buf.astype(np.complex64)



    def generate_per_antenna(self, n_samples: int = STEP_SAMPLES) -> dict[str, np.ndarray]:
        """
        Швидкий генератор: повертає IQ по антенах ( MiniBuf AmplitudeSync).
        Без BLANK-маркерів — для прямого estimate().
        """
        t = np.arange(n_samples) / SAMPLE_RATE
        result: dict[str, np.ndarray] = {}

        for ant_idx in range(N_ANTENNAS):
            name = 'ANT_%s' % chr(65 + ant_idx)
            phi_k = 2 * np.pi * ant_idx / N_ANTENNAS

            noise_amp = 10 ** (self._noise_floor_db / 20)
            sig = noise_amp * (np.random.randn(n_samples)
                               + 1j * np.random.randn(n_samples))

            for src in self._sources:
                theta = np.radians(src['angle_deg'])
                lam = C / src['freq_hz']
                spatial_phase = (2 * np.pi * ARRAY_RADIUS_M / lam) * np.cos(phi_k - theta)

                sig_amp = 10 ** (src['power_db'] / 20)
                freq_offset = src['freq_hz'] - self._center_freq
                carrier = sig_amp * np.exp(
                    1j * (2 * np.pi * freq_offset * t + spatial_phase)
                )
                sig += carrier

            result[name] = sig.astype(np.complex64)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def hr(ch='-', n=70):
    print(ch * n)

def angular_error(got: float, true: float) -> float:
    diff = abs(got - true) % 360
    return min(diff, 360 - diff)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Один дрон, чистий ефір
# ══════════════════════════════════════════════════════════════════════════════

def test_single_drone():
    hr('=')
    print("SCENARIO 1: Single drone, clean spectrum")
    print("  Drone at 135 deg, 433 MHz LoRa, power=20 dB")
    hr()

    sim = DroneSimulator(center_freq=433e6, noise_floor_db=-30)
    sim.add_drone(angle_deg=135, freq_hz=433e6, power_db=20, bw_hz=200e3)

    df = DirectionFinder(center_freq=433e6)
    ant_iq = sim.generate_per_antenna()

    sv = df.estimate(ant_iq)
    mu = df.estimate_music(ant_iq)
    err_sv = angular_error(sv, 135)
    err_mu = angular_error(mu, 135)

    print("  SV:    bearing=%.1f  err=%.1f  conf=%.3f" % (sv, err_sv, df.confidence))
    print("  MUSIC: bearing=%.1f  err=%.1f" % (mu, err_mu))

    ok = err_sv < 15
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Дрон + WiFi завада з іншого боку
# ══════════════════════════════════════════════════════════════════════════════

def test_drone_plus_wifi():
    hr('=')
    print("SCENARIO 2: Drone (135 deg) + WiFi interference (270 deg)")
    print("  Drone: 433 MHz, 20 dB")
    print("  WiFi:  433.5 MHz, 25 dB (stronger!)")
    hr()

    sim = DroneSimulator(center_freq=433e6, noise_floor_db=-30)
    sim.add_drone(angle_deg=135, freq_hz=433e6, power_db=20, bw_hz=200e3)
    sim.add_interference(angle_deg=270, freq_hz=433.5e6, power_db=25, bw_hz=5e6)

    df = DirectionFinder(center_freq=433e6)
    ant_iq = sim.generate_per_antenna()

    sv = df.estimate(ant_iq)
    err_drone = angular_error(sv, 135)
    err_wifi = angular_error(sv, 270)

    print("  bearing=%.1f" % sv)
    print("  Error to drone (135): %.1f" % err_drone)
    print("  Error to WiFi  (270): %.1f" % err_wifi)

    # Без BPF, сильніший сигнал (WiFi) може перетягнути кут
    if err_wifi < err_drone:
        print("  [EXPECTED] Bearing points to WiFi (stronger signal)")
        print("  -> This proves BPF is needed for real use!")
    else:
        print("  Bearing closer to drone despite WiFi")

    print("  Result: INFO (demonstrates BPF necessity)")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Мульти-частотний дрон
# ══════════════════════════════════════════════════════════════════════════════

def test_multi_freq_drone():
    hr('=')
    print("SCENARIO 3: Multi-frequency drone at 90 deg")
    print("  433 MHz (LoRa) + 868 MHz (ELRS) + 5800 MHz (Video)")
    hr()

    true_angle = 90
    freqs = [
        (433e6,  "433 MHz LoRa",   20, 200e3),
        (868e6,  "868 MHz ELRS",   25, 500e3),
        (5800e6, "5.8 GHz Video", 15, 8e6),
    ]

    print("  %16s  bearing  err    conf   2piR/lam  status" % "Frequency")
    hr('-', 70)

    bearings = []
    for freq, name, power, bw in freqs:
        sim = DroneSimulator(center_freq=freq, noise_floor_db=-30)
        sim.add_drone(angle_deg=true_angle, freq_hz=freq, power_db=power, bw_hz=bw)

        df = DirectionFinder(center_freq=freq)
        ant_iq = sim.generate_per_antenna()
        sv = df.estimate(ant_iq)
        err = angular_error(sv, true_angle)

        lam = C / freq
        ratio = 2 * np.pi * ARRAY_RADIUS_M / lam
        ambig = ratio > np.pi

        status = "AMBIG" if ambig else ("PASS" if err < 15 else "FAIL")
        print("  %16s  %6.1f  %5.1f  %.3f   %6.3f    %s" % (
            name, sv, err, df.confidence, ratio, status
        ))
        bearings.append((sv, err, ambig))

    # Порівнюємо пеленги неамбігуозних частот
    clean = [(b, e) for b, e, a in bearings if not a]
    if len(clean) >= 2:
        spread = max(b for b, _ in clean) - min(b for b, _ in clean)
        print()
        print("  Non-ambiguous bearings spread: %.1f deg" % spread)
        ok = spread < 30
        print("  Multi-freq consistency: %s" % ('PASS' if ok else 'FAIL'))
    else:
        ok = True
        print("  Not enough non-ambiguous freqs to compare")

    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Дрон що рухається
# ══════════════════════════════════════════════════════════════════════════════

def test_moving_drone():
    hr('=')
    print("SCENARIO 4: Moving drone (0 -> 180 deg, step 10)")
    hr()

    df = DirectionFinder(center_freq=433e6)
    angles = list(range(0, 190, 10))
    errors = []

    print("  True   Got    Err")
    hr('-', 35)

    for true_deg in angles:
        sim = DroneSimulator(center_freq=433e6, noise_floor_db=-30)
        sim.add_drone(angle_deg=true_deg, freq_hz=433e6, power_db=20, bw_hz=200e3)
        ant_iq = sim.generate_per_antenna()

        sv = df.estimate(ant_iq)
        err = angular_error(sv, true_deg)
        errors.append(err)
        marker = " !!!" if err > 15 else ""
        print("  %4d  %6.1f  %5.1f%s" % (true_deg, sv, err, marker))

    avg = float(np.mean(errors))
    mx = float(np.max(errors))
    print()
    print("  Avg error: %.2f deg" % avg)
    print("  Max error: %.2f deg" % mx)

    ok = mx < 15
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Слабкий сигнал (далекий дрон)
# ══════════════════════════════════════════════════════════════════════════════

def test_weak_signal():
    hr('=')
    print("SCENARIO 5: Weak signal (distant drone at 60 deg)")
    hr()

    df = DirectionFinder(center_freq=433e6)
    true_deg = 60
    snr_values = [-5, 0, 3, 5, 10, 15, 20]
    n_trials = 20

    print("  SNR(dB)  AvgErr  MaxErr  AvgConf  Status")
    hr('-', 55)

    high_snr_ok = True
    for snr in snr_values:
        errs = []
        confs = []
        for _ in range(n_trials):
            sim = DroneSimulator(center_freq=433e6, noise_floor_db=-snr)
            sim.add_drone(angle_deg=true_deg, freq_hz=433e6,
                          power_db=0, bw_hz=200e3)
            ant_iq = sim.generate_per_antenna()
            sv = df.estimate(ant_iq)
            errs.append(angular_error(sv, true_deg))
            confs.append(df.confidence)

        avg_e = float(np.mean(errs))
        max_e = float(np.max(errs))
        avg_c = float(np.mean(confs))
        ok = avg_e < 30 if snr < 5 else avg_e < 15
        if snr >= 10 and not ok:
            high_snr_ok = False
        print("  %5d    %5.1f   %5.1f   %.3f    %s" % (
            snr, avg_e, max_e, avg_c, 'PASS' if ok else 'FAIL'
        ))

    print()
    print("  High SNR (>=10 dB) all pass: %s" % ('YES' if high_snr_ok else 'NO'))
    print("  Result: %s" % ('PASS' if high_snr_ok else 'FAIL'))
    return high_snr_ok


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Два дрони з різних напрямків
# ══════════════════════════════════════════════════════════════════════════════

def test_two_drones():
    hr('=')
    print("SCENARIO 6: Two drones from different directions")
    print("  Drone A: 45 deg, 433 MHz, 20 dB")
    print("  Drone B: 225 deg, 868 MHz, 25 dB")
    hr()

    # Drone A: 433 MHz
    sim_a = DroneSimulator(center_freq=433e6, noise_floor_db=-30)
    sim_a.add_drone(angle_deg=45, freq_hz=433e6, power_db=20, bw_hz=200e3)

    df_a = DirectionFinder(center_freq=433e6)
    ant_a = sim_a.generate_per_antenna()
    bear_a = df_a.estimate(ant_a)
    err_a = angular_error(bear_a, 45)

    # Drone B: 868 MHz
    sim_b = DroneSimulator(center_freq=868e6, noise_floor_db=-30)
    sim_b.add_drone(angle_deg=225, freq_hz=868e6, power_db=25, bw_hz=500e3)

    df_b = DirectionFinder(center_freq=868e6)
    ant_b = sim_b.generate_per_antenna()
    bear_b = df_b.estimate(ant_b)
    err_b = angular_error(bear_b, 225)

    print("  Drone A (433 MHz): true=45   got=%.1f  err=%.1f" % (bear_a, err_a))
    print("  Drone B (868 MHz): true=225  got=%.1f  err=%.1f" % (bear_b, err_b))

    # Перевіряємо що кути розділені
    separation = angular_error(bear_a, bear_b)
    print("  Angular separation: %.1f deg (expected ~180)" % separation)

    ok = err_a < 15 and err_b < 15
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 7 (bonus): Full pipeline — sync + direction
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline():
    hr('=')
    print("SCENARIO 7: Full pipeline (AmplitudeSync -> DirectionFinder)")
    print("  Drone at 200 deg, 433 MHz, with BLANK markers in IQ")
    print("  Per-cycle bearing estimation + circular mean")
    hr()

    true_deg = 200

    sim = DroneSimulator(center_freq=433e6, noise_floor_db=-30)
    sim.add_drone(angle_deg=true_deg, freq_hz=433e6, power_db=20, bw_hz=0)

    # Full IQ with BLANK markers
    iq_buffer = sim.generate(n_cycles=N_CYCLES)

    # Step 1: AmplitudeSync — find markers
    sync = AmplitudeSyncDetector()
    markers = sync.detect_only(iq_buffer)

    print("  AmplitudeSync:")
    print("    Markers found : %d" % len(markers))
    print("    Quality       : %.3f" % sync.sync_quality)

    if len(markers) < 2:
        print("  FAIL: not enough markers")
        return False

    # Step 2: Per-cycle bearing
    # For each marker: extract one cycle (5 antenna steps after BLANK)
    df = DirectionFinder(center_freq=433e6)
    step = sync.step_samples
    guard = int(step * 0.20)
    ant_names = ['ANT_%s' % chr(65 + i) for i in range(N_ANTENNAS)]

    cycle_bearings = []

    print()
    print("  Per-cycle bearings:")
    for mi, marker in enumerate(markers):
        start = marker + step  # skip BLANK, start from ANT_A
        # Build per-antenna IQ for this single cycle
        cycle_iq = {}
        valid = True
        for ai, name in enumerate(ant_names):
            s = start + ai * step + guard
            e = start + (ai + 1) * step - guard
            if e > len(iq_buffer):
                valid = False
                break
            cycle_iq[name] = iq_buffer[s:e]

        if not valid:
            continue

        bearing = df.estimate(cycle_iq)
        if bearing is not None:
            err = angular_error(bearing, true_deg)
            cycle_bearings.append(bearing)
            print("    Cycle %d: bearing=%.1f  err=%.1f" % (mi, bearing, err))

    if not cycle_bearings:
        print("  FAIL: no valid cycles")
        return False

    # Step 3: Circular mean of bearings
    rads = np.radians(cycle_bearings)
    mean_sin = float(np.mean(np.sin(rads)))
    mean_cos = float(np.mean(np.cos(rads)))
    avg_bearing = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360

    err_avg = angular_error(avg_bearing, true_deg)
    print()
    print("  Circular mean bearing: %.1f deg (err=%.1f)" % (avg_bearing, err_avg))
    print("  Spread: %.1f deg" % float(np.std(cycle_bearings)))

    ok = err_avg < 20
    print()
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok



# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    hr('=')
    print("  E2E DRONE DIRECTION FINDING — TEST SUITE")
    hr('=')
    print("  N_ANTENNAS  : %d" % N_ANTENNAS)
    print("  ARRAY_RADIUS: %.0f cm" % (ARRAY_RADIUS_M * 100))
    print("  SAMPLE_RATE : %.0f MSPS" % (SAMPLE_RATE / 1e6))
    print("  STEP        : %.1f ms = %d samples" % (ANTENNA_STEP_MS, STEP_SAMPLES))
    print("  CYCLE        : %d steps = %d samples" % (N_ANTENNAS + 1, CYCLE_SAMPLES))
    print("  BUFFER       : %d cycles = %d samples" % (N_CYCLES, TOTAL_SAMPLES))
    hr('=')

    tests = [
        test_single_drone,
        test_drone_plus_wifi,
        test_multi_freq_drone,
        test_moving_drone,
        test_weak_signal,
        test_two_drones,
        test_full_pipeline,
    ]

    results = []
    for fn in tests:
        print()
        try:
            ok = fn()
        except Exception as exc:
            print("  EXCEPTION: %s" % exc)
            import traceback
            traceback.print_exc()
            ok = False
        results.append(ok)

    print()
    hr('=')
    passed = sum(1 for r in results if r)
    print("  SUMMARY: %d/%d tests passed" % (passed, len(tests)))
    print()
    for i, (fn, ok) in enumerate(zip(tests, results)):
        print("  Scenario %d: [%s] %s" % (i + 1, 'PASS' if ok else 'FAIL', fn.__name__))
    hr('=')

    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
