"""Time-series AC power flow on the IEEE 33-bus feeder with an EV charging peak.

python ev_load_generator.py               -> one deterministic 24 h trajectory
python ev_load_generator.py --n-traj 100  -> also 100 randomised trajectories
All data is synthetic: public test network + invented load shapes + AC power flow.
"""
import argparse, json, datetime
import numpy as np
import pandapower as pp
import pandapower.networks as nw
import load_profiles as lp

SINGLE, MULTI, TIME = ("ieee33_ev_voltage_dataset.npy",
                       "ieee33_ev_voltage_dataset_multi.npy", "simulation_time_axis.npy")
SLACK_V = 1.05   # substation voltage (p.u.)


def simulate(multipliers):
    """Run an AC power flow per step. Failed steps stay NaN so they are never hidden."""
    net = nw.case33bw()
    net.ext_grid["vm_pu"] = SLACK_V
    p0, q0 = net.load.p_mw.copy(), net.load.q_mvar.copy()
    v = np.full((len(multipliers), len(net.bus)), np.nan)
    for i, m in enumerate(multipliers):
        net.load.p_mw = p0.values * m
        net.load.q_mvar = q0.values * m
        try:
            pp.runpp(net, numba=False)
            v[i] = net.res_bus.vm_pu.values
        except Exception as exc:
            print(f"  power flow failed at step {i} (multiplier {np.mean(m):.3f}): {exc}")
    return v


def report(v, mult, t):
    step, bus = np.unravel_index(np.argmin(v), v.shape)
    hh, mm = int(t[step]), round((t[step] % 1) * 60)
    print(f"Shape: {v.shape} (timesteps x buses)")
    print(f"Peak load multiplier: {mult.max():.3f} x base load")
    print(f"Minimum voltage: {v.min():.4f} p.u. at bus index {bus} (IEEE bus {bus + 1}), {hh:02d}:{mm:02d}")
    print(f"Readings below 0.95 p.u.: {(v < 0.95).sum()} of {v.size}")
    print(f"Slack voltage set to {SLACK_V} p.u.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=0, help="also generate N randomised trajectories")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t = lp.time_axis()
    mult = lp.total_multiplier(t)
    print(f"Running {lp.N_TIMESTEPS} power flows...")
    v = simulate(mult)
    if np.isnan(v).any():
        raise SystemExit("Some power flows did not converge; dataset NOT saved. Reduce the load multipliers.")
    report(v, mult, t)
    np.save(SINGLE, v)
    np.save(TIME, t)

    info = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
            "pandapower": pp.__version__, "numpy": np.__version__, "network": "pandapower case33bw","slack_voltage_pu": SLACK_V,
            "default_profile": {"base_amp": lp.BASE_AMP, "base_peak_h": lp.BASE_PEAK_H,
                                "ev_amp": lp.EV_AMP, "ev_peak_h": lp.EV_PEAK_H, "ev_width": lp.EV_WIDTH}}
    if args.n_traj:
        rng = np.random.default_rng(args.seed)
        n_load = len(nw.case33bw().load)
        trajs, plist, attempts = [], [], 0
        while len(trajs) < args.n_traj and attempts < 3 * args.n_traj:
            attempts += 1
            p = lp.random_params(rng)
            base = lp.base_curve(t, p["base_amp"], p["base_peak_h"])
            ev = lp.ev_curve(t, p["ev_amp"], p["ev_peak_h"])
            scale = 1 + rng.normal(0, 0.05, n_load)            # fixed per-load size variation
            n_ev = int(rng.integers(8, 13))                     # 8-12 loads have EV charging
            ev_idx = rng.choice(n_load, size=n_ev, replace=False)
            ev_w = np.zeros(n_load)
            ev_w[ev_idx] = n_load / n_ev                        # keeps total EV energy similar
            noise = 1 + rng.normal(0, 0.01, (t.size, n_load))   # small per-load, per-step noise
            m = (base[:, None] * scale + ev[:, None] * ev_w) * noise
            vt = simulate(m)
            if not np.isnan(vt).any():
                trajs.append(vt)
                plist.append({**{key: float(val) for key, val in p.items()},
                              "ev_load_idx": [int(i) for i in ev_idx]})
        np.save(MULTI, np.stack(trajs))
        info.update(n_trajectories=len(trajs), attempts=attempts, trajectory_params=plist)
        print(f"Saved {len(trajs)} randomised trajectories to {MULTI}")
    with open("generation_params.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Saved datasets and generation_params.json")


if __name__ == "__main__":
    main()
