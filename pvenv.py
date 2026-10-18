"""
PV string (4 x CS6P-250P, series, bypass diodes) + averaged boost converter
feeding a fixed DC bus. MPPT controller acts on duty cycle D every Ts.
"""
import numpy as np
from numba import njit

q = 1.602176634e-19; kB = 1.380649e-23
# --- CS6P-250P single-diode parameters (fitted to datasheet STC: Pmax 250 W,
# Vmp 30.1 V, Imp 8.30 A, Voc 37.2 V, Isc 8.87 A, Ns = 60)
NS, N_IDEAL, RS, RSH = 60, 1.18307, 0.23997, 894.22
ISC_STC, VOC_STC, KI, KV = 8.87, 37.2, 0.00053, -0.0031   # KI,KV: per degC (relative)
EG = 1.12
N_MOD = 4
V_BYPASS = -0.6

# --- converter (averaged model)
L_IND, R_L, C_IN, V_BUS = 1.0e-3, 0.05, 220e-6, 200.0
DT = 2e-6
TS = 0.01            # MPPT sampling period 10 ms
D_MIN, D_MAX = 0.20, 0.95
P_STC_STRING = 4 * 249.83


def module_params(G, T):
    Tk = T + 273.15; Tr = 298.15
    a = N_IDEAL * NS * kB * Tk / q
    a_r = N_IDEAL * NS * kB * Tr / q
    Isc = ISC_STC * (G / 1000.0) * (1 + KI * (T - 25))
    Voc_r = VOC_STC
    I0_r = (ISC_STC * (1 + RS / RSH) - Voc_r / RSH) / (np.exp(Voc_r / a_r) - np.exp(ISC_STC * RS / a_r))
    I0 = I0_r * (Tk / Tr) ** 3 * np.exp(q * EG / (N_IDEAL * kB) * (1 / Tr - 1 / Tk))
    Iph = Isc * (1 + RS / RSH)
    return Iph, I0, a


def module_IV(G, T, npts=300):
    """Return (V, I) on V in [0, Voc] for one module."""
    if G < 1.0:
        return np.array([0.0, 1e-3]), np.array([0.0, 0.0])
    Iph, I0, a = module_params(G, T)
    # Voc estimate by Newton on I=0
    Voc = a * np.log(Iph / I0 + 1)
    for _ in range(30):
        f = Iph - I0 * (np.exp(Voc / a) - 1) - Voc / RSH
        df = -I0 * np.exp(Voc / a) / a - 1 / RSH
        Voc -= f / df
    V = np.linspace(0, Voc, npts)
    I = np.full(npts, Iph)
    for _ in range(50):
        e = np.exp((V + I * RS) / a)
        f = Iph - I0 * (e - 1) - (V + I * RS) / RSH - I
        df = -I0 * e * RS / a - RS / RSH - 1
        I = I - f / df
    I[-1] = 0.0
    return V, np.maximum(I, 0.0)


def string_IV(Gs, T=25.0, n_i=800):
    """Series string with ideal-ish bypass diodes. Returns V grid (ascending) and I(V)."""
    Imax = max(ISC_STC * max(Gs) / 1000.0 * (1 + KI * (T - 25)) * 1.02, 1e-3)
    Ig = np.linspace(0, Imax, n_i)
    Vs = np.zeros(n_i)
    for G in Gs:
        V, I = module_IV(G, T)
        isc = I[0]
        # module voltage as function of current (I is decreasing in V)
        Vk = np.interp(Ig, I[::-1], V[::-1])
        Vk[Ig > isc] = V_BYPASS
        Vs += Vk
    # Vs decreasing in I -> build I(V) on monotone grid
    order = np.argsort(Vs)
    Vsorted, Isorted = Vs[order], Ig[order]
    Vgrid = np.linspace(Vsorted[0], Vsorted[-1], 1500)
    Igrid = np.interp(Vgrid, Vsorted, Isorted)
    return Vgrid, Igrid


def gmpp(Vgrid, Igrid):
    P = Vgrid * Igrid
    k = int(np.argmax(P))
    return P[k], Vgrid[k]


@njit(cache=True)
def _ipv(v, Vg, Ig):
    if v <= Vg[0]:
        return Ig[0]
    if v >= Vg[-1]:
        return 0.0
    n = Vg.shape[0]
    x = (v - Vg[0]) / (Vg[-1] - Vg[0]) * (n - 1)
    j = int(x); f = x - j
    if j >= n - 1:
        return Ig[n - 1]
    return Ig[j] * (1 - f) + Ig[j + 1] * f


@njit(cache=True)
def simulate_period(v, iL, D, Vg, Ig, nsteps, dt, L, rL, C, Vbus):
    """Forward-Euler averaged boost; returns final state and averages over last 30%."""
    sv = 0.0; si = 0.0; cnt = 0; ipk = 0.0
    navg = int(nsteps * 0.3)
    for k in range(nsteps):
        ipv = _ipv(v, Vg, Ig)
        dv = (ipv - iL) / C
        di = (v - rL * iL - (1.0 - D) * Vbus) / L
        v += dv * dt
        iL += di * dt
        if iL < 0.0:
            iL = 0.0        # diode blocks (DCM approx.)
        if v < 0.0:
            v = 0.0
        if iL > ipk:
            ipk = iL
        if k >= nsteps - navg:
            sv += v; si += _ipv(v, Vg, Ig); cnt += 1
    return v, iL, sv / cnt, si / cnt, ipk


class PVSystem:
    """Plant: call set_irradiance(Gs, T) then step(D) -> (Vpv, Ipv, Ppv)."""

    def __init__(self, T=25.0):
        self.T = T
        self._cache = {}
        self.reset()

    def reset(self, D0=0.5):
        self.v = 0.0; self.iL = 0.0; self.D = D0
        self.Gs = None

    def set_irradiance(self, Gs, T=None):
        if T is not None:
            self.T = T
        key = (tuple(int(round(g)) for g in Gs), round(self.T, 1))
        if key not in self._cache:
            Vg, Ig = string_IV([float(g) for g in key[0]], self.T)
            self._cache[key] = (Vg, Ig, *gmpp(Vg, Ig))
            if len(self._cache) > 20000:
                self._cache.clear()
        self.Vg, self.Ig, self.Pmax, self.Vmpp = self._cache[key]
        self.Gs = key[0]
        if self.v == 0.0 and self.iL == 0.0:      # start at open circuit
            self.v = self.Vg[-1]

    def step(self, D):
        D = float(np.clip(D, D_MIN, D_MAX))
        self.D = D
        nsteps = int(round(TS / DT))
        self.v, self.iL, vm, im, self.iL_peak = simulate_period(self.v, self.iL, D, self.Vg, self.Ig,
                                                  nsteps, DT, L_IND, R_L, C_IN, V_BUS)
        return vm, im, vm * im
