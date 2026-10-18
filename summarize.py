"""Aggregate results_<tag>.json: RL variants mean ± SD over seeds (each seed = mean over 5 initial duty cycles)."""
import json, sys, numpy as np, re, csv
tag = sys.argv[1]
R = json.load(open(f"results_{tag}.json"))
groups = {}
for k, v in R.items():
    m = re.match(r"(.+)_s(\d+)$", k)
    groups.setdefault(m.group(1) if m else k, []).append(v)
scen = list(next(iter(R.values())).keys())
order = ["P&O", "INC", "PSO", "none_conv", "none_tw", "static_conv", "adaptive_conv", "adaptive_tw"]
rows = []
for metric in ["eta", "ss_eta", "settle_ms", "osc_W", "n_unsettled"]:
    print(f"\n=== {metric} ===")
    print(f"{'method':15s}" + "".join(f"{s[:14]:>16s}" for s in scen) + f"{'MEAN':>14s}")
    for g in order:
        if g not in groups: continue
        L = groups[g]
        vals = np.array([[r[s][metric] for s in scen] for r in L])       # seeds x scen
        mu, sd = np.nanmean(vals, 0), np.nanstd(vals, 0)
        overall = np.nanmean(vals, 1)
        line = f"{g:15s}" + "".join(f"{a:9.1f}±{b:<5.1f}" for a, b in zip(mu, sd)) + f"{np.nanmean(overall):8.2f}±{np.nanstd(overall):.2f}"
        print(line)
        rows.append([metric, g, len(L)] + [f"{a:.2f}±{b:.2f}" for a, b in zip(mu, sd)] + [f"{np.nanmean(overall):.2f}±{np.nanstd(overall):.2f}"])
with open(f"summary_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "method", "n_seeds"] + scen + ["mean"]); w.writerows(rows)
