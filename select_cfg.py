import json, glob, collections, numpy as np
R = collections.defaultdict(list)
for f in glob.glob("models/tune/*_val.json"):
    d = json.load(open(f)); R[(d["fz"], d["rk"], d["lr_a"], d["lr_c"], d["gamma"])].append(d["val_eta"])
best = {}
for k, v in R.items():
    var = (k[0], k[1])
    if var not in best or np.mean(v) > best[var][1]:
        best[var] = (k, np.mean(v))
json.dump({f"{a}_{b}": {"lr_a": k[2], "lr_c": k[3], "gamma": k[4], "val_eta": m, "all": {f"{kk[2]:g}/{kk[3]:g}/{kk[4]:g}": float(np.mean(R[kk])) for kk in R if kk[:2] == (a, b)}} for (a, b), (k, m) in best.items()}, open("tuning_result.json", "w"), indent=1)
with open("jobs_final.txt", "w") as f:
    for (a, b), (k, m) in best.items():
        for s in range(10):
            f.write(f"{a} {b} {s} 400 final {k[2]} {k[3]} {k[4]}\n")
print(open("tuning_result.json").read())
