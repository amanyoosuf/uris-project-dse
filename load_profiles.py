"""Pure-NumPy load profiles (no pandapower needed). All values are synthetic."""
import numpy as np

N_TIMESTEPS = 144          # 24 h at 10-minute resolution
STEP_H = 1 / 6             # exactly 10 minutes

# Default deterministic profile
BASE_AMP, BASE_PEAK_H = 0.20, 18.0   # +/-20 % daily swing, evening residential peak
EV_AMP, EV_PEAK_H, EV_WIDTH = 0.25, 19.0, 4.0  # Gaussian-shaped EV charging bump


def time_axis():
    """144 points, 0.0 .. 23.83 h, exactly 10 min apart (no duplicate midnight)."""
    return np.arange(N_TIMESTEPS) * STEP_H


def base_curve(t, amp=BASE_AMP, peak_h=BASE_PEAK_H):
    """Residential demand multiplier; sine peaks at peak_h."""
    return 1.0 + amp * np.sin(2 * np.pi * (t - peak_h + 6) / 24)


def ev_curve(t, amp=EV_AMP, peak_h=EV_PEAK_H, width=EV_WIDTH):
    """Extra load multiplier from evening EV charging."""
    return amp * np.exp(-((t - peak_h) ** 2) / width)


def total_multiplier(t, base_amp=BASE_AMP, base_peak_h=BASE_PEAK_H,
                     ev_amp=EV_AMP, ev_peak_h=EV_PEAK_H):
    return base_curve(t, base_amp, base_peak_h) + ev_curve(t, ev_amp, ev_peak_h)


def random_params(rng):
    """Random profile parameters for one trajectory (used for multi-trajectory data)."""
    return dict(base_amp=rng.uniform(0.15, 0.25), base_peak_h=rng.uniform(17.0, 20.0),
                ev_amp=rng.uniform(0.10, 0.35), ev_peak_h=rng.normal(19.5, 0.75))
