# Do Fuzzy Features Help DDPG-Based MPPT? Multi-seed benchmark code

Code and results for the paper "Do Fuzzy Features Help DDPG-Based MPPT? A Multi-Seed Benchmark Under Partial Shading" (submitted to the Journal of Electrical Engineering & Technology). A MATLAB re-implementation of the plant and controllers is in `matlab/` (see `matlab/README_MATLAB.md`).

## How to reproduce
```
pip install -r requirements.txt
python3 train_one.py adaptive tw 0 400 main     # variant: none|static|adaptive, reward: conv|tw, seed 0-9, episodes, tag
# all 50 training runs (5 variants x 10 seeds): see jobs_seeds.txt and pipeline2.sh
python3 eval_all.py main                        # all controllers x 10 scenarios x 5 initial duty cycles -> results_main.json
python3 summarize.py main                       # mean ± SD over seeds -> summary_main.csv
python3 figs.py main                            # figures
```
On 1 CPU core, each 400-episode run takes about 5–7 minutes.

## Plant (pvenv.py)
- 4 × Canadian Solar CS6P-250P in series, each module with a bypass diode (-0.6 V).
- Isc temperature coefficient 0.053 %/°C (datasheet). Single-diode model fitted to the datasheet: n = 1.183, Rs = 0.240 Ω, Rsh = 894 Ω. Model Pmax at STC is 249.8 W (datasheet 250 W).
- Averaged boost converter: L = 1 mH, rL = 0.05 Ω, C_in = 220 µF, fixed 200 V DC bus. Forward Euler with dt = 2 µs.
- MPPT sampling period is 10 ms. Duty cycle is limited to 0.20–0.95.
- Limitation: the averaged model has no switching ripple and assumes ideal semiconductors apart from rL.

## Controllers
- P&O and INC: fixed step ΔD = 0.01.
- PSO: 4 particles, w = 0.3, c1 = 1.2, c2 = 1.6, convergence when ptp < 0.02 or after 6 iterations, restart on a 5% power change. These settings were tuned over 6 configurations, so this is a fair (not crippled) PSO.
- DDPG family (rl.py), all sharing the same setup:
  - Sensor-free 10-dimensional observation: V, I, P, ΔV, ΔI, ΔP, D, best P since the last detected irradiance change, D_best − D, and steps since the change.
  - Action ΔD within ±0.15.
  - Actor/critic 64-64, γ = 0.97, 400 episodes of 200 steps each.
  - Training uses randomised uniform, 2-level and 3-level shading at 20–50 °C. The test patterns are held out.
  - Fuzzy layer: 5 Gaussian membership functions per input, normalised firing strengths concatenated with the raw input. "Static" keeps the membership functions fixed, as in García-Luna et al. 2025. "Adaptive" trains the centres and widths end-to-end.
  - Rewards. "conv" = η − 1 − 0.5|ΔD|. "tw" (time-weighted) = −w(k)(1−η) − |ΔD|·[η > 0.97] − 0.2·max(0, i_L,peak/i_pv − 1.2), with w(k) = 1 + 2e^(−k/15).

## Evaluation
- Each RL variant is trained with 10 seeds (0–9). Trained actors are in `models/main/`. Every controller is run from 5 initial duty cycles (0.3–0.7).
- Efficiency η is the energy ratio ΣP / ΣP_GMPP.

## Main result (energy efficiency %, mean over 10 scenarios; ± is SD across 10 seeds)
| Method | Mean η |
|---|---|
| P&O | 85.8 |
| INC | 84.9 |
| PSO (tuned) | 95.7 |
| DDPG | 91.2 ± 2.4 |
| DDPG + TW reward | 86.0 ± 3.6 |
| Static Fuzzy-DDPG | 89.7 ± 2.8 |
| Adaptive Fuzzy-DDPG | 90.8 ± 3.3 |
| Adaptive Fuzzy-DDPG + TW reward | 89.1 ± 2.9 |

Full per-scenario results are in `summary_main.csv` and `results_main.json`. The earlier 5-seed and 1000-episode runs are in `results_long.json` and `summary_long.csv`.

## Main findings
1. All DDPG variants beat P&O/INC under partial shading, mainly on PS4 and PS5, where hill-climbing gets trapped at a local peak.
2. None of the RL variants beats a tuned PSO overall. PSO loses only on the irradiance ramp (S3), and the RL variants are close to it under dynamic shading (S5).
3. Adaptive membership functions give **no statistically meaningful gain** over plain DDPG, and the time-weighted reward lowers mean efficiency (86.0 % vs 91.2 % for plain DDPG). A 1000-episode check with 3 seeds (`summary_long.csv`) did not change this.
4. Adaptive fuzzy variants show larger steady-state oscillation than plain DDPG (5.9–6.9 W vs 2.9 W, ramp excluded).
5. RL results vary strongly between seeds (PS4 range 54.8–98.8 % across seeds for the adaptive TW variant). Single-seed RL MPPT results in the literature should be read with that in mind.

Figures: fig_pv_curves, fig_training, fig_eff_bars, fig_traces_S5 (median seed, not the best seed), fig_mf.
