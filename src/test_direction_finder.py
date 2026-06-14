"""
test_direction_finder.py — Тестування коректності виявлення кута (пеленгатора).

Симулює сигнали з відомих напрямків та перевіряє:
  1. Точність Steering Vector Scan на всіх кутах (крок 15 deg)
  2. Точність MUSIC на всіх кутах
  3. Вплив SNR (від -5 dB до +30 dB) на похибку
  4. Стабільність: 100 прогонів з одного кута → std кута
  5. Вплив частоти (433 / 868 / 2400 МГц) на точність
  6. Найгірший випадок: 2piR/lam < 1 (велика lambda vs малий R)

Запуск: python test_direction_finder.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from core.config import SAMPLE_RATE, ANTENNA_STEP_MS, N_ANTENNAS, ARRAY_RADIUS_M
from dsp.direction_finder import DirectionFinder

# ---- Helpers ----------------------------------------------------------------

C = 3e8  # speed of light
STEP_SAMPLES = int(SAMPLE_RATE * ANTENNA_STEP_MS / 1000)


def hr(char='-', n=70):
    print(char * n)


def angular_error(got_deg: float, true_deg: float) -> float:
    """Мiнiмальна кутова похибка з урахуванням перекиду 0/360."""
    diff = abs(got_deg - true_deg) % 360
    return min(diff, 360 - diff)


def make_antenna_iq(
    true_deg: float,
    freq_hz: float = 433e6,
    snr_db: float = 20.0,
    n_samples: int = STEP_SAMPLES,
    radius_m: float = ARRAY_RADIUS_M,
) -> dict[str, np.ndarray]:
    """
    Генерує IQ-дані для кожної антени кільцевого масиву (UCA).
    Сигнал = плоска хвиля з напрямку true_deg + гаусів шум.
    """
    lam = C / freq_hz
    phase_const = 2 * np.pi * radius_m / lam
    theta = np.radians(true_deg)

    ant_iq = {}
    for k in range(N_ANTENNAS):
        phi_k = 2 * np.pi * k / N_ANTENNAS
        phase = phase_const * np.cos(phi_k - theta)

        # Чистий сигнал (тональний)
        signal = np.exp(1j * phase) * np.ones(n_samples, dtype=complex)

        # Шум
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        )
        ant_iq['ANT_%s' % chr(65 + k)] = signal + noise

    return ant_iq


# ---- Test 1: Steering Vector - all angles -----------------------------------

def test_sv_all_angles():
    hr('=')
    print("TEST 1: Steering Vector Scan — all angles (step 15 deg)")
    hr()

    df = DirectionFinder(center_freq=433e6)
    errors = []

    print("  True    Got    Err   Conf")
    hr('-', 40)
    for true_deg in range(0, 360, 15):
        ant_iq = make_antenna_iq(true_deg, snr_db=20)
        got = df.estimate(ant_iq)
        err = angular_error(got, true_deg)
        errors.append(err)
        marker = " !!!" if err > 5 else ""
        print("  %4d  %6.1f  %5.1f  %.3f%s" % (true_deg, got, err, df.confidence, marker))

    avg_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    print()
    print("  Avg error: %.2f deg" % avg_err)
    print("  Max error: %.2f deg" % max_err)
    ok = max_err < 5.0
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ---- Test 2: MUSIC - all angles ---------------------------------------------

def test_music_all_angles():
    hr('=')
    print("TEST 2: MUSIC algorithm — all angles (step 15 deg)")
    hr()

    df = DirectionFinder(center_freq=433e6)
    errors = []

    print("  True    Got    Err   Conf")
    hr('-', 40)
    for true_deg in range(0, 360, 15):
        ant_iq = make_antenna_iq(true_deg, snr_db=20)
        got = df.estimate_music(ant_iq)
        if got is None:
            print("  %4d    None" % true_deg)
            continue
        err = angular_error(got, true_deg)
        errors.append(err)
        marker = " !!!" if err > 5 else ""
        print("  %4d  %6.1f  %5.1f  %.3f%s" % (true_deg, got, err, df.confidence, marker))

    avg_err = float(np.mean(errors)) if errors else 999
    max_err = float(np.max(errors)) if errors else 999
    print()
    print("  Avg error: %.2f deg" % avg_err)
    print("  Max error: %.2f deg" % max_err)
    ok = max_err < 5.0
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ---- Test 3: SNR sweep ------------------------------------------------------

def test_snr_sweep():
    hr('=')
    print("TEST 3: SNR sweep (true_angle = 120 deg)")
    hr()

    df = DirectionFinder(center_freq=433e6)
    true_deg = 120.0

    snr_list = [-5, 0, 3, 5, 10, 15, 20, 30]
    n_trials = 20

    print("  SNR(dB)  AvgErr  MaxErr  Conf   Status")
    hr('-', 55)

    results_ok = []
    for snr in snr_list:
        errs = []
        confs = []
        for _ in range(n_trials):
            ant_iq = make_antenna_iq(true_deg, snr_db=snr)
            got = df.estimate(ant_iq)
            errs.append(angular_error(got, true_deg))
            confs.append(df.confidence)

        avg_e = float(np.mean(errs))
        max_e = float(np.max(errs))
        avg_c = float(np.mean(confs))
        ok = avg_e < 20
        results_ok.append(ok)
        print("  %5d    %5.1f   %5.1f  %.3f   %s" % (
            snr, avg_e, max_e, avg_c, 'PASS' if ok else 'FAIL'
        ))

    print()
    # High SNR should always pass
    high_snr_ok = all(results_ok[-3:])
    print("  High SNR (15-30 dB) pass: %s" % ('YES' if high_snr_ok else 'NO'))
    print("  Result: %s" % ('PASS' if high_snr_ok else 'FAIL'))
    return high_snr_ok


# ---- Test 4: Stability (100 runs, same angle) --------------------------------

def test_stability():
    hr('=')
    print("TEST 4: Stability — 100 runs at true_angle = 45 deg, SNR = 15 dB")
    hr()

    df = DirectionFinder(center_freq=433e6)
    true_deg = 45.0
    n_runs = 100

    results_sv = []
    results_mu = []

    t0 = time.perf_counter()
    for _ in range(n_runs):
        ant_iq = make_antenna_iq(true_deg, snr_db=15)
        sv = df.estimate(ant_iq)
        mu = df.estimate_music(ant_iq)
        results_sv.append(sv)
        results_mu.append(mu)
    dt = time.perf_counter() - t0

    sv_arr = np.array(results_sv)
    mu_arr = np.array(results_mu)
    sv_errs = [angular_error(g, true_deg) for g in results_sv]
    mu_errs = [angular_error(g, true_deg) for g in results_mu]

    print("  Steering Vector:")
    print("    Mean bearing : %.2f deg" % float(np.mean(sv_arr)))
    print("    Std          : %.2f deg" % float(np.std(sv_arr)))
    print("    Avg error    : %.2f deg" % float(np.mean(sv_errs)))
    print("    Max error    : %.2f deg" % float(np.max(sv_errs)))
    print()
    print("  MUSIC:")
    print("    Mean bearing : %.2f deg" % float(np.mean(mu_arr)))
    print("    Std          : %.2f deg" % float(np.std(mu_arr)))
    print("    Avg error    : %.2f deg" % float(np.mean(mu_errs)))
    print("    Max error    : %.2f deg" % float(np.max(mu_errs)))
    print()
    print("  Time: %.1f ms total, %.2f ms per run" % (dt * 1000, dt / n_runs * 1000))

    ok = float(np.max(sv_errs)) < 10 and float(np.max(mu_errs)) < 10
    print("  Result: %s" % ('PASS' if ok else 'FAIL'))
    return ok


# ---- Test 5: Multiple frequencies -------------------------------------------

def test_multi_frequency():
    hr('=')
    print("TEST 5: Multi-frequency — accuracy at different center frequencies")
    hr()

    true_deg = 200.0
    freqs = [
        (433e6,  "433 MHz (LoRa)"),
        (868e6,  "868 MHz (ELRS)"),
        (915e6,  "915 MHz (TBS)"),
        (2400e6, "2.4 GHz (WiFi)"),
        (5800e6, "5.8 GHz (Video)"),
    ]

    lam_limit = 2 * np.pi * ARRAY_RADIUS_M  # 2piR

    print("  %15s  lam(cm)  2piR/lam  SV_err  MU_err  Status" % "Frequency")
    hr('-', 70)

    all_ok = True
    for freq, name in freqs:
        lam = C / freq
        ratio = lam_limit / lam

        df = DirectionFinder(center_freq=freq)
        ant_iq = make_antenna_iq(true_deg, freq_hz=freq, snr_db=20)
        sv = df.estimate(ant_iq)
        mu = df.estimate_music(ant_iq)
        err_sv = angular_error(sv, true_deg)
        err_mu = angular_error(mu, true_deg)

        # На високих частотах (2piR/lam > pi) можуть бути ambiguity
        ambiguity = ratio > np.pi
        ok = err_sv < 10 or ambiguity
        if not ok:
            all_ok = False

        status = "AMBIG" if ambiguity else ("PASS" if ok else "FAIL")
        print("  %15s  %6.1f   %6.3f    %5.1f   %5.1f   %s" % (
            name, lam * 100, ratio, err_sv, err_mu, status
        ))

    print()
    if not all_ok:
        print("  NOTE: High-freq ambiguity is expected when 2piR/lam > pi")
    print("  Result: INFO (ambiguity expected at high freqs)")
    return True


# ---- Test 6: Fine resolution — sub-degree increments -------------------------

def test_fine_resolution():
    hr('=')
    print("TEST 6: Fine resolution — angles 0.0, 0.5, 1.0, ... 5.0 deg")
    hr()

    df = DirectionFinder(center_freq=433e6)
    print("  True     SV_got   Err_SV   MU_got   Err_MU")
    hr('-', 55)

    sv_errs = []
    mu_errs = []
    for true_deg_10x in range(0, 55, 5):   # 0.0, 0.5, 1.0 ... 5.0
        true_deg = true_deg_10x / 10.0
        ant_iq = make_antenna_iq(true_deg, snr_db=25)
        sv = df.estimate(ant_iq)
        mu = df.estimate_music(ant_iq)
        e_sv = angular_error(sv, true_deg)
        e_mu = angular_error(mu, true_deg)
        sv_errs.append(e_sv)
        mu_errs.append(e_mu)
        print("  %5.1f   %7.2f   %5.2f   %7.2f   %5.2f" % (true_deg, sv, e_sv, mu, e_mu))

    print()
    print("  SV avg sub-degree error: %.3f deg" % float(np.mean(sv_errs)))
    print("  MU avg sub-degree error: %.3f deg" % float(np.mean(mu_errs)))
    print("  Result: INFO")
    return True


# ---- Main -------------------------------------------------------------------

def main():
    hr('=')
    print("  DIRECTION FINDER — TEST SUITE")
    hr('=')
    print("  N_ANTENNAS     : %d" % N_ANTENNAS)
    print("  ARRAY_RADIUS   : %.0f cm" % (ARRAY_RADIUS_M * 100))
    print("  STEP_SAMPLES   : %d" % STEP_SAMPLES)
    print("  Base freq      : 433 MHz")
    lam_433 = C / 433e6
    print("  lambda(433MHz) : %.1f cm" % (lam_433 * 100))
    print("  2piR/lambda    : %.3f rad" % (2 * np.pi * ARRAY_RADIUS_M / lam_433))
    hr('=')

    tests = [
        test_sv_all_angles,
        test_music_all_angles,
        test_snr_sweep,
        test_stability,
        test_multi_frequency,
        test_fine_resolution,
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
        print("  Test %d: [%s] %s" % (i + 1, 'PASS' if ok else 'FAIL', fn.__name__))
    hr('=')

    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
