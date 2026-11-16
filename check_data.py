"""Sanity checks on the generated datasets. Exits with code 1 if any check fails."""
import os, sys
import numpy as np

SINGLE, MULTI, TIME = ("ieee33_ev_voltage_dataset.npy",
                       "ieee33_ev_voltage_dataset_multi.npy", "simulation_time_axis.npy")
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


v, t = np.load(SINGLE), np.load(TIME)
print(f"Voltage dataset {v.shape}, time axis {t.shape}")
check("voltage shape is (144, 33)", v.shape == (144, 33))
check("time axis has 144 points, 10 min apart", t.shape == (144,) and np.allclose(np.diff(t), 1 / 6))
check("no NaN or inf values", np.isfinite(v).all())
check("slack bus (index 0) is 1.05 p.u. at every step", np.allclose(v[:, 0], 1.05))
check("all voltages between 0.5 and 1.05 p.u.", ((v > 0.5) & (v <= 1.05 + 1e-9)).all())
step, bus = np.unravel_index(np.argmin(v), v.shape)
print(f"Minimum voltage {v.min():.4f} p.u. at bus index {bus} (IEEE bus {bus + 1}), t = {t[step]:.2f} h")
print(f"Readings below 0.95 p.u.: {(v < 0.95).sum()} of {v.size}")

if os.path.exists(MULTI):
    m = np.load(MULTI)
    print(f"Multi-trajectory dataset {m.shape}")
    check("multi dataset shape is (N, 144, 33)", m.ndim == 3 and m.shape[1:] == (144, 33))
    check("multi dataset has no NaN or inf", np.isfinite(m).all())
    check("trajectories differ from each other", m.shape[0] < 2 or not np.allclose(m[0], m[1]))
sys.exit(0 if all(ok) else 1)
