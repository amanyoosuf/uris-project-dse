"""Pure-NumPy load profiles (no pandapower needed). All values are synthetic."""
import numpy as np

N_TIMESTEPS = 144          # 24 h at 10-minute resolution
STEP_H = 1 / 6             # exactly 10 minutes

# Default deterministic profile
BASE_AMP, BASE_PEAK_H = 0.20, 18.0   # +/-20 % daily swing, evening residential peak
# --- EV charging model (scenario assumptions; see DATA_PROVENANCE.md) ---
BASE_LOAD_KW = 3715.0          # case33bw total active load
N_HOUSEHOLDS = 2000            # assumed households on the feeder
CHARGER_KW = 7.0               # assumed home AC charger
PEAK_CHARGING_FRACTION = 0.25  # assumed share of EVs charging at the evening peak
EV_PENETRATION = 0.25          # default EVs per household (single-day profile)
EV_PEAK_H, EV_SIGMA_H = 19.5, 1.5
NIGHT_PEAK_H, NIGHT_SIGMA_H, NIGHT_FRAC = 1.0, 2.0, 0.3  # smaller overnight bump


def ev_amp_from_penetration(pen):
    """Peak EV load as a multiple of base load."""
    return N_HOUSEHOLDS * pen * CHARGER_KW * PEAK_CHARGING_FRACTION / BASE_LOAD_KW


EV_AMP = ev_amp_from_penetration(EV_PENETRATION)


def time_axis():
    """144 points, 0.0 .. 23.83 h, exactly 10 min apart (no duplicate midnight)."""
    return np.arange(N_TIMESTEPS) * STEP_H


def base_curve(t, amp=BASE_AMP, peak_h=BASE_PEAK_H):
    """Residential demand multiplier; sine peaks at peak_h."""
    return 1.0 + amp * np.sin(2 * np.pi * (t - peak_h + 6) / 24)


def _circ(t, peak):
    """Signed distance in hours on a 24 h circle, so the overnight bump wraps past midnight."""
    return (t - peak + 12) % 24 - 12


def ev_curve(t, amp=EV_AMP, peak_h=EV_PEAK_H, sigma=EV_SIGMA_H):
    """Extra load multiplier from EV charging: evening peak plus a smaller overnight bump."""
    main = np.exp(-_circ(t, peak_h) ** 2 / (2 * sigma ** 2))
    night = NIGHT_FRAC * np.exp(-_circ(t, NIGHT_PEAK_H) ** 2 / (2 * NIGHT_SIGMA_H ** 2))
    return amp * (main + night)


def total_multiplier(t, base_amp=BASE_AMP, base_peak_h=BASE_PEAK_H,
                     ev_amp=EV_AMP, ev_peak_h=EV_PEAK_H):
    return base_curve(t, base_amp, base_peak_h) + ev_curve(t, ev_amp, ev_peak_h)


def random_params(rng):
    """Random profile parameters for one trajectory (used for multi-trajectory data)."""
    pen = rng.uniform(0.10, 0.40)
    return dict(base_amp=rng.uniform(0.15, 0.25), base_peak_h=rng.uniform(17.0, 20.0),
                ev_penetration=pen, ev_amp=ev_amp_from_penetration(pen),
                ev_peak_h=rng.normal(19.5, 0.75))