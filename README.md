# URIS Project: Data-Driven Dynamic State Estimation for Modern Power Grids

Student: Amanur Rahman Yoosuf. Supervisor: Prof. Siqi Bu (EEE, PolyU).
Goal: forecast bus voltages on the IEEE 33-bus feeder with an LSTM under EV charging load, then flag expected violations.

**All data in this repository is synthetic.** See `DATA_PROVENANCE.md`.

## Setup
```
conda env create -f environment.yml
conda activate uris_env
conda env export --no-builds > environment.lock.yml   # record exact versions, then commit it
```

## Run order
| Step | Command | What it does |
|---|---|---|
| 1 | `python test_ieee33.py` | Loads the IEEE 33-bus network and runs one AC power flow |
| 2 | `python validate_base_case.py` | Compares the base case with published values |
| 3 | `python ev_load_generator.py` | 24 h, 144 steps, one deterministic trajectory |
| 3b | `python ev_load_generator.py --n-traj 100` | Adds 100 randomised trajectories (Phase 2) |
| 4 | `python check_data.py` | Sanity checks on the saved datasets |
| 5 | `python plot_results.py` | Figures in `figures/` |
| 6 | `python train_lstm.py` | LSTM vs persistence baseline (meaningful only with step 3b data) |

## Files
- `load_profiles.py`: baseline and EV load curves (NumPy only, all parameters in one place)
- `ev_load_generator.py`: time-series power flow, saves `.npy` datasets and `generation_params.json`
- `train_lstm.py`: sliding-window dataset, LSTM, training loop, persistence baseline
- Datasets: `ieee33_ev_voltage_dataset.npy` (144 x 33), `simulation_time_axis.npy` (144), optional `..._multi.npy` (N x 144 x 33)

## Status
Phase 1 (environment, IEEE 33-bus model, load generator): complete.
Next: calibrate EV profile to Hong Kong data, simulate sparse sensors, evaluate the LSTM on many trajectories.

## Use of AI tools
See `AI_USE_LOG.md`.
