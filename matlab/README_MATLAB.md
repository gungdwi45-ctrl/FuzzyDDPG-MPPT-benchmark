# MATLAB / Octave version of the MPPT benchmark

Same plant, controllers and scenarios as the Python package. Tested in GNU Octave 8; runs unchanged in MATLAB.

## Files
- `mppt_params.m` plant and converter parameters (CS6P-250P fit, Isc coeff. 0.053 %/degC)
- `pv_module_iv.m`, `pv_string_iv.m` single-diode modules, 4-module string with bypass diodes, GMPP
- `boost_period.m` one 10 ms MPPT period, `'avg'` averaged model or `'sw'` 20 kHz switching model (Ron 0.05 ohm, diode 0.7 V)
- `ctrl_po.m`, `ctrl_inc.m`, `ctrl_pso_init.m` + `ctrl_pso.m` baselines (tuned PSO)
- `ctrl_ddpg_init.m` + `ctrl_ddpg.m` trained DDPG / Fuzzy-DDPG policy, weights in `policies/*.mat`
- `mppt_scenarios.m` the 10 test scenarios, `run_scenario.m` one run, `main_compare.m` full comparison

## Run
```
main_compare        % all controllers x scenarios x 3 initial duty cycles, averaged and switching
```
The switching model uses a 0.25 us step, so it is about 10x slower than the averaged model. MATLAB is much faster than Octave for these loops.

## Cross-check against Python (PS4, 1 s, D0 = 0.5, averaged model)
| Controller | Python | Octave |
|---|---|---|
| P&O | 47.89% | 47.93% |
| DDPG (seed 1) | 91.68% | 91.69% |
| Adaptive F-DDPG-TW (seed 4) | 94.81% | 94.12% |

Small differences come from interpolation details in the I-V curve construction. PSO uses MATLAB's own random generator, so its exact trajectory differs from Python.
