"""Compare averaged vs switching converter model for every controller (main models).
usage: python3 validate_sw.py <tag>"""
import sys, glob, json, numpy as np, torch
from pvenv import PVSystem, TS
from pvenv_sw import PVSystemSW
from scenarios import scenarios
from evaluate import metrics
from classic import PO, INC, PSO
from rl import RLController, Actor, Critic, Fuzzify  # noqa

def run(ctrl, scen, plant_cls, D0):
    ctrl.D0 = D0; plant = plant_cls(); D = ctrl.reset()
    out = {k: [] for k in ["P", "Pmax"]}
    for Gs, T in scen:
        plant.set_irradiance(Gs, T); v, i, p = plant.step(D)
        out["P"].append(p); out["Pmax"].append(plant.Pmax); D = ctrl.act(v, i, p)
    return {k: np.array(x) for k, x in out.items()}

tag = sys.argv[1] if len(sys.argv) > 1 else "main"
S = scenarios(); D0S = [0.3, 0.5, 0.7]
ctrls = {"P&O": PO, "INC": INC, "PSO": PSO}
for f in sorted(glob.glob(f"models/{tag}/*.pt")):
    a = torch.load(f, weights_only=False); nm = f.split("/")[-1][:-3]
    ctrls[nm] = (lambda a=a, nm=nm: RLController(a, nm))
res = {}
for cn, mk in ctrls.items():
    res[cn] = {}
    for sn, sc in S.items():
        r = {}
        for pname, pc in (("avg", PVSystem), ("sw", PVSystemSW)):
            r[pname] = float(np.mean([metrics(run(mk(), sc, pc, d))["eta"] for d in D0S]))
        res[cn][sn] = r
    print(cn, round(np.mean([v["avg"] for v in res[cn].values()]), 2), round(np.mean([v["sw"] for v in res[cn].values()]), 2), flush=True)
json.dump(res, open(f"validate_sw_{tag}.json", "w"), indent=1)
