import json, glob, sys, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pvenv import string_IV, gmpp
from scenarios import PS, scenarios
from evaluate import run
from classic import PO, INC, PSO
from rl import RLController, Actor, Critic, Fuzzify  # noqa

TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                     "legend.frameon": False, "savefig.dpi": 300})
VAR = {"adaptive_tw": "Proposed (adaptive Fuzzy-DDPG, TW reward)", "adaptive_conv": "Adaptive Fuzzy-DDPG, conv. reward",
       "static_conv": "Static Fuzzy-DDPG [ref]", "none_conv": "DDPG", "none_tw": "DDPG, TW reward"}

def fig_pv():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for j, (k, g) in enumerate(PS.items()):
        Vg, Ig = string_IV(g); P = Vg * Ig; pm, vm = gmpp(Vg, Ig)
        ax.plot(Vg, P, color=C[j], label=f"{k}  {g} W/m²")
        ax.plot([vm], [pm], "o", ms=6, color=C[j], mec="white", mew=1.5)
    Vg, Ig = string_IV([1000] * 4); ax.plot(Vg, Vg * Ig, color=MUTED, ls="--", lw=1.2, label="Uniform 1000 W/m²")
    ax.set_xlabel("String voltage (V)"); ax.set_ylabel("Power (W)"); ax.set_xlim(0, 155); ax.set_ylim(0, 1050)
    ax.legend(fontsize=7.5, loc="upper left"); fig.tight_layout(); fig.savefig("fig_pv_curves.png"); plt.close(fig)

def load_models():
    M = {}
    for f in sorted(glob.glob(f"models/{TAG}/*.pt")):
        nm = f.split("/")[-1][:-3]; v, s = nm.rsplit("_s", 1)
        M.setdefault(v, {})[int(s)] = torch.load(f, weights_only=False)
    return M

def fig_training():
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for j, v in enumerate(VAR):
        H = [np.load(f) for f in sorted(glob.glob(f"models/{TAG}/{v}_s*_hist.npy"))]
        if not H: continue
        H = np.array([np.convolve(h, np.ones(20) / 20, "valid") for h in H])
        m, s = H.mean(0), H.std(0); x = np.arange(len(m)) + 20
        ax.plot(x, m, color=C[j], label=VAR[v]); ax.fill_between(x, m - s, m + s, color=C[j], alpha=0.15, lw=0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode return (20-ep. mean)")
    ax.set_title("TW and conventional rewards have different scales; compare within reward type", fontsize=7, color=MUTED, loc="left")
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, 0.08)); fig.tight_layout(); fig.savefig("fig_training.png"); plt.close(fig)

def fig_traces(M, seed_p, seed_d):
    sc = scenarios()["S5 dynamic shading"]
    ctrls = [(f"Adaptive Fuzzy-DDPG (median seed)", RLController(M["adaptive_tw"][seed_p], "p"), C[0]),
             ("DDPG (median seed)", RLController(M["none_conv"][seed_d], "d"), C[4]),
             ("PSO", PSO(), C[3]), ("P&O", PO(), C[1])]
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True)
    tr0 = None
    for nm, c, col in ctrls:
        tr = run(c, sc, D0=0.5); tr0 = tr
        axs[0].plot(tr["t"], tr["P"], color=col, label=nm, lw=1.3)
        axs[1].plot(tr["t"], tr["V"], color=col, lw=1.3)
    axs[0].plot(tr0["t"], tr0["Pmax"], color=INK, ls=":", lw=1.2, label="GMPP")
    axs[1].plot(tr0["t"], tr0["Vmpp"], color=INK, ls=":", lw=1.2)
    for t in [1, 2, 3]:
        for a in axs: a.axvline(t, color=MUTED, lw=0.6)
    for x, lab in [(0.5, "Uniform"), (1.5, "PS4"), (2.5, "PS3"), (3.5, "Uniform 700")]:
        axs[0].text(x, 1035, lab, ha="center", fontsize=7.5, color=MUTED)
    axs[0].set_ylabel("PV power (W)"); axs[0].set_ylim(0, 1120); axs[1].set_ylabel("PV voltage (V)")
    axs[1].set_xlabel("Time (s)"); axs[0].legend(ncol=5, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.08), handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(); fig.savefig("fig_traces_S5.png"); plt.close(fig)

def fig_mf(M, seed):
    a = M["adaptive_tw"][seed]; f = a.f
    names = ["V", "I", "P", "ΔV", "ΔI", "ΔP", "D", "P_best", "ΔD_best", "k"]
    x = torch.linspace(-1, 1.5, 300)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
    for ax, idx in zip(axs, [2, 5, 8]):
        c0 = torch.linspace(-1, 1.5, 5); s0 = (2.5 / 4 / 1.2)
        for m in range(5):
            ax.plot(x, torch.exp(-0.5 * ((x - c0[m]) / s0) ** 2), color=MUTED, ls="--", lw=0.9)
            c, s = f.c[idx, m].item(), f.log_s[idx, m].exp().item()
            ax.plot(x, torch.exp(-0.5 * ((x - c) / s) ** 2), color=C[0], lw=1.4)
        ax.set_title(f"input {names[idx]} (normalised)", fontsize=8); ax.set_xlabel("x")
    axs[0].set_ylabel("Membership")
    axs[0].plot([], [], color=MUTED, ls="--", label="initial"); axs[0].plot([], [], color=C[0], label="learned")
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("fig_mf.png"); plt.close(fig)

def fig_bars():
    import re
    R = json.load(open(f"results_{TAG}.json"))
    meth = [("P&O", "P&O"), ("INC", "INC"), ("PSO", "PSO (tuned)"), ("none_conv", "DDPG"),
            ("static_conv", "Static Fuzzy-DDPG"), ("adaptive_tw", "Adaptive Fuzzy-DDPG + TW")]
    scen = ["S1 uniform STC", "S2 uniform steps", "S3 ramp 100 W/m2/s", "S4 PS1", "S4 PS2", "S4 PS3", "S4 PS4", "S4 PS5", "S4 PS4 @45C", "S5 dynamic shading"]
    lab = ["S1", "S2", "S3", "PS1", "PS2", "PS3", "PS4", "PS5", "PS4\n45°C", "S5"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0)); W = 0.8 / len(meth); x = np.arange(len(scen))
    for j, (k, nm) in enumerate(meth):
        keys = [kk for kk in R if kk == k or re.fullmatch(re.escape(k) + r"_s\d+", kk)]
        V = np.array([[R[kk][s]["eta"] for s in scen] for kk in keys])
        ax.bar(x - 0.4 + W * (j + 0.5), V.mean(0), W * 0.88, color=C[j], label=nm,
               yerr=V.std(0) if len(keys) > 1 else None, error_kw=dict(lw=0.7, capsize=1.2, ecolor=INK))
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(40, 101); ax.set_ylabel("Energy efficiency (%)")
    ax.legend(ncol=6, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.9); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig("fig_eff_bars.png"); plt.close(fig)

if __name__ == "__main__":
    fig_pv(); fig_training()
    M = load_models()
    res = json.load(open(f"results_{TAG}.json")) if glob.glob(f"results_{TAG}.json") else None
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    fig_traces(M, 6, 1); fig_mf(M, 6); fig_bars()
