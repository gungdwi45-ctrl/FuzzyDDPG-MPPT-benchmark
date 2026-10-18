"""train_cfg.py fuzzy rkind seed episodes tag lr_a lr_c gamma"""
import sys, torch, numpy as np, os, json
from rl import train
fz, rk, seed, eps, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
lr_a, lr_c, gamma = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
os.makedirs(f"models/{tag}", exist_ok=True)
_name = f"models/{tag}/{fz}_{rk}_a{lr_a:g}_c{lr_c:g}_g{gamma:g}_s{seed}"
if os.path.exists(_name + ".pt") and (tag != "tune" or os.path.exists(_name + "_val.json")):
    print("skip, already done", _name); sys.exit(0)
actor, hist = train(None if fz == "none" else fz, rk, seed, eps, gamma=gamma, lr_a=lr_a, lr_c=lr_c, log=lambda m: print(m, flush=True))
name = f"models/{tag}/{fz}_{rk}_a{lr_a:g}_c{lr_c:g}_g{gamma:g}_s{seed}"
torch.save(actor, name + ".pt"); np.save(name + "_hist.npy", hist)
if tag == "tune":   # evaluate on validation set right away
    from scenarios import val_scenarios
    from evaluate import run, metrics
    from rl import RLController
    S = val_scenarios(); c = RLController(actor, "x")
    eta = float(np.mean([metrics(run(c, sc, D0=d))["eta"] for sc in S.values() for d in (0.3, 0.5, 0.7)]))
    json.dump({"fz": fz, "rk": rk, "seed": seed, "lr_a": lr_a, "lr_c": lr_c, "gamma": gamma, "val_eta": eta}, open(name + "_val.json", "w"))
print("saved", name, flush=True)
