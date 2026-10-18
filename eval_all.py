"""Evaluate controllers over all scenarios and 5 initial duty cycles."""
import sys, glob, torch, numpy as np, json
from scenarios import scenarios
from classic import PO, INC, PSO
from evaluate import run, metrics
from rl import RLController, Actor, Critic, Fuzzify  # noqa (for torch.load)

D0S = [0.3, 0.4, 0.5, 0.6, 0.7]

def eval_ctrl(make, scen):
    ms = [metrics(run(make(), scen, D0=d)) for d in D0S]
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}

if __name__ == "__main__":
    tag = sys.argv[1]
    S = scenarios()
    res = {}
    ctrls = {"P&O": PO, "INC": INC, "PSO": PSO}
    for f in sorted(glob.glob(f"models/{tag}/*.pt")):
        a = torch.load(f, weights_only=False)
        nm = f.split("/")[-1][:-3]
        ctrls[nm] = (lambda a=a, nm=nm: RLController(a, nm))
    for cn, mk in ctrls.items():
        res[cn] = {sn: eval_ctrl(mk, sc) for sn, sc in S.items()}
        print(cn, " ".join(f"{sn.split()[0]}{sn.split()[1] if sn.startswith('S4') else ''}={v['eta']:.1f}" for sn, v in res[cn].items()), flush=True)
    json.dump(res, open(f"results_{tag}.json", "w"), indent=1)
