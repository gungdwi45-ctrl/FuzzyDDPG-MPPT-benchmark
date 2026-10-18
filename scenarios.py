"""Test scenarios. Each returns list of (Gs[4], T) per control step (Ts = 10 ms)."""
import numpy as np

STEPS_PER_S = 100

def hold(Gs, T, sec):
    return [(list(Gs), T)] * int(sec * STEPS_PER_S)

def ramp(G0, G1, T, rate):  # uniform ramp, rate in W/m2/s
    n = int(abs(G1 - G0) / rate * STEPS_PER_S)
    return [([g] * 4, T) for g in np.linspace(G0, G1, n)]

# Five partial-shading patterns on the 4-module string (held-out from training)
PS = {
    "PS1": [1000, 1000, 600, 300],   # GMPP mid (3 peaks)
    "PS2": [1000, 800, 600, 400],    # 4 peaks, close local peak at high V
    "PS3": [1000, 300, 300, 300],    # GMPP at high V, deep local peak at low V
    "PS4": [900, 900, 200, 200],     # GMPP at low V, local peak at high V (P&O trap)
    "PS5": [1000, 1000, 700, 250],   # 3 peaks, GMPP mid-voltage
}

def scenarios():
    S = {}
    S["S1 uniform STC"] = hold([1000] * 4, 25, 2.0)
    S["S2 uniform steps"] = hold([1000] * 4, 25, 1.0) + hold([400] * 4, 25, 1.0) + hold([800] * 4, 25, 1.0)
    S["S3 ramp 100 W/m2/s"] = hold([1000] * 4, 25, 0.5) + ramp(1000, 300, 25, 100) + hold([300] * 4, 25, 0.5) + ramp(300, 1000, 25, 100)
    for k, g in PS.items():
        S[f"S4 {k}"] = hold(g, 25, 2.0)
    S["S4 PS4 @45C"] = hold(PS["PS4"], 45, 2.0)
    S["S5 dynamic shading"] = hold([1000] * 4, 25, 1.0) + hold(PS["PS4"], 25, 1.0) + hold(PS["PS3"], 25, 1.0) + hold([700] * 4, 25, 1.0)
    return S


def val_scenarios(n=12, seed=2026):
    """Validation set for hyperparameter tuning: random patterns, disjoint from the fixed test set."""
    from rl import random_pattern
    rng = np.random.default_rng(seed); S = {}
    for k in range(n):
        T = float(rng.uniform(20, 50))
        S[f"V{k}"] = hold(random_pattern(rng), T, 1.0) + hold(random_pattern(rng), T, 1.0)
    return S
