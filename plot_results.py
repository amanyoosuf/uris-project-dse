"""Make report figures into ./figures (load profile always; voltage plot if data exists)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import load_profiles as lp

os.makedirs("figures", exist_ok=True)
t = lp.time_axis()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, lp.base_curve(t), label="Baseline residential")
ax.plot(t, lp.ev_curve(t), label="EV charging component")
ax.plot(t, lp.total_multiplier(t), lw=2.5, label="Total load multiplier")
ax.set(xlabel="Hour of day", ylabel="Multiplier on base load", title="Synthetic load profile (24 h, 10-min steps)")
ax.set_xlim(0, 24); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figures/load_profile.png", dpi=200); plt.close(fig)
print("Saved figures/load_profile.png")

if os.path.exists("ieee33_ev_voltage_dataset.npy"):
    v = np.load("ieee33_ev_voltage_dataset.npy")
    w = int(np.argmin(v.min(axis=0)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, v[:, w], lw=2, label=f"Weakest bus (index {w}, IEEE bus {w + 1})")
    ax.plot(t, v[:, 4], label="Bus index 4 (IEEE bus 5)")
    ax.axhline(0.95, ls="--", color="red", label="0.95 p.u. limit")
    ax.set(xlabel="Hour of day", ylabel="Voltage (p.u.)", title="Voltage over 24 h with EV charging peak")
    ax.set_xlim(0, 24); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/voltage_profile.png", dpi=200); plt.close(fig)
    print("Saved figures/voltage_profile.png")
else:
    print("Run ev_load_generator.py first to get the voltage plot.")
