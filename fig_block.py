import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
INK, MUTED = "#1f1f1e", "#6b6a63"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a998f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(-1, 56); ax.axis("off")
def box(x, y, w, h, title, sub, col, fill="#ffffff", dash=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.3, ec=col, fc=fill, ls="--" if dash else "-"))
    ax.text(x + w/2, y + h - 2.3, title, ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
    ax.text(x + w/2, y + h - 6.3, sub, ha="center", va="top", fontsize=7.2, color=MUTED, linespacing=1.3)
def arr(p, q, col=INK, txt=None, tpos=None, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text(*tpos, txt, fontsize=7.2, color=col, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.6))
# plant row
box(2, 40, 24, 14, "PV string", "4 × 250 W modules in series\nbypass diode per module\nG(t), T(t) per module", INK)
box(38, 40, 24, 14, "Boost converter", "averaged model\nL, r_L, C_in, Euler step 2 µs\nD ∈ [0.20, 0.95]", INK)
box(74, 40, 24, 14, "DC bus", "fixed 200 V\nv ≈ (1 − D)·V_bus", INK)
arr((26.8, 47), (37.2, 47), txt="v, i", tpos=(32, 49.5))
arr((62.8, 47), (73.2, 47))
# controller row
box(2, 13, 22, 17, "Observation builder", "10 inputs from v, i and D\nevery 10 ms (no irradiance\nsensor)", BLUE, "#f3f8fe")
box(29, 13, 20, 17, "Fuzzy layer", "none | static Gaussian MFs |\nadaptive MFs (trained\nend-to-end)", BLUE, "#f3f8fe")
box(54, 13, 18, 17, "DDPG actor", "MLP 64–64\noutput ΔD\n+ Gaussian noise\n(training only)", BLUE, "#f3f8fe")
box(77, 13, 21, 17, "Duty-cycle update", "D(k+1) = D(k) + ΔD\nclipped to [0.20, 0.95]", BLUE, "#f3f8fe")
arr((24.8, 21.5), (28.2, 21.5), BLUE); arr((49.8, 21.5), (53.2, 21.5), BLUE); arr((72.8, 21.5), (76.2, 21.5), BLUE)
arr((14, 39.2), (14, 30.8), BLUE, txt="measured v, i", tpos=(14, 35))
arr((87.5, 30.8), (55, 39.2), BLUE, txt="D", tpos=(73, 35.5), rad=0.0)
# training-only and baselines
box(29, 0.0, 43, 11, "Training only", "reward (conventional or time-weighted), critic MLP 64–64,\nreplay buffer, Adam optimiser", ORANGE, "#fff6f1", dash=True)
arr((63, 11.6), (63, 12.4), ORANGE, ls="--")
box(77, 0.0, 21, 11, "Baselines", "P&O, INC, tuned PSO\n(replace the agent row)", GREEN, "#f1fbf6", dash=True)
fig.tight_layout(pad=0.2); fig.savefig("fig_block.png"); print("ok")
