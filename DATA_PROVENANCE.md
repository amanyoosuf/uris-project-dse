# Data provenance and validation

## Where the data comes from
| Item | Source | Real or synthetic |
|---|---|---|
| Network (33 buses, 37 lines, base loads) | Pandapower built-in `case33bw`, from Baran & Wu (1989) | Standard public test case |
| Baseline daily load curve | Sine function chosen by me (+/-20 %, peak 18:00) | Synthetic |
| EV charging curve | Gaussian bump chosen by me (peak 19:00) | Synthetic. NOT yet calibrated to Hong Kong data |
| Bus voltages | AC power flow (Pandapower) at each timestep | Computed from the model, not measured |

No measured or private data is used. All loads are scaled by one common multiplier.

## How correctness is checked
1. `validate_base_case.py`: base-case loads, minimum voltage and losses compared with commonly reported values.
2. `check_data.py`: shapes, 10-minute spacing, no NaN, slack bus at 1.0 p.u., voltages in a plausible range.
3. `ev_load_generator.py` refuses to save if any power flow fails to converge.
4. Every dataset is regenerated from code. `generation_params.json` stores the seed, parameters and library versions.

## Known limitations
- All loads scale together, but real EV load is clustered at certain buses.
- The test feeder already sags below 0.95 p.u. at base load, so violations are partly a property of the test case.
- No solar or other distributed generation, and no model error.
- One feeder topology only.
- The deterministic dataset is one day. Use the randomised set for any learning or evaluation.

## References
- M. E. Baran and F. F. Wu, "Network reconfiguration in distribution systems for loss reduction and load balancing," IEEE Trans. Power Delivery, vol. 4, no. 2, pp. 1401-1407, Apr. 1989.
- L. Thurner et al., "pandapower: An open-source Python tool for convenient modeling, analysis, and optimization of electric power systems," IEEE Trans. Power Systems, vol. 33, no. 6, pp. 6510-6521, Nov. 2018.
- Hong Kong EV and solar figures used in the proposal: see its reference list. Add the sources used for calibration here.
