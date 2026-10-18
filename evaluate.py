import numpy as np
from pvenv import PVSystem, TS

def run(ctrl, scen, D0=0.5):
    if hasattr(ctrl, "D0"):
        ctrl.D0 = D0
    """Run controller over scenario. Returns dict of traces."""
    plant = PVSystem()
    D = ctrl.reset()
    out = {k: [] for k in ["t", "V", "I", "P", "Pmax", "D", "Vmpp"]}
    for k, (Gs, T) in enumerate(scen):
        plant.set_irradiance(Gs, T)
        v, i, p = plant.step(D)
        out["t"].append((k + 1) * TS); out["V"].append(v); out["I"].append(i); out["P"].append(p)
        out["Pmax"].append(plant.Pmax); out["D"].append(D); out["Vmpp"].append(plant.Vmpp)
        D = ctrl.act(v, i, p)
    return {k: np.array(x) for k, x in out.items()}

def metrics(tr, band=0.02):
    P, Pm = tr["P"], tr["Pmax"]
    eta = P.sum() / Pm.sum() * 100
    # segment boundaries where Pmax changes markedly (step changes)
    chg = np.where(np.abs(np.diff(Pm)) > 0.02 * Pm[:-1])[0] + 1
    segs = np.split(np.arange(len(P)), chg)
    ts_list, ss_eta, osc = [], [], []
    for s in segs:
        if len(s) < 20:
            continue
        e = P[s] / Pm[s]
        tail = s[int(len(s) * 0.7):]
        ss_eta.append(np.mean(P[tail] / Pm[tail]) * 100)
        osc.append(np.std(P[tail] - Pm[tail]))
        ok = e >= 1 - band
        # settling: first index after which always within band
        bad = np.where(~ok)[0]
        if len(bad) == 0:
            ts_list.append(0.0)
        elif bad[-1] == len(s) - 1:
            ts_list.append(np.nan)            # never settles at GMPP
        else:
            ts_list.append((bad[-1] + 1) * TS * 1000)
    return dict(eta=eta, ss_eta=np.mean(ss_eta), settle_ms=np.nanmean(ts_list) if not np.all(np.isnan(ts_list)) else np.nan,
                n_unsettled=int(np.sum(np.isnan(ts_list))), osc_W=np.mean(osc))
