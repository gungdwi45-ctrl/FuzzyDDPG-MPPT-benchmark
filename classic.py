"""Conventional MPPT baselines operating on duty cycle (boost: V_pv = (1-D) V_bus)."""
import numpy as np
from pvenv import D_MIN, D_MAX

class PO:
    name = "P&O"
    def __init__(self, dD=0.01, D0=0.5):
        self.dD, self.D0 = dD, D0
    def reset(self):
        self.D = self.D0; self.pv = None; self.pp = None; self.dir = 1
        return self.D
    def act(self, v, i, p):
        if self.pp is not None:
            if p < self.pp:
                self.dir = -self.dir
        self.pp = p
        self.D = float(np.clip(self.D + self.dir * self.dD, D_MIN, D_MAX))
        return self.D

class INC:
    name = "INC"
    def __init__(self, dD=0.01, D0=0.5, eps=0.02):
        self.dD, self.D0, self.eps = dD, D0, eps
    def reset(self):
        self.D = self.D0; self.vp = None; self.ip = None
        return self.D
    def act(self, v, i, p):
        if self.vp is None:
            dD = self.dD
        else:
            dv, di = v - self.vp, i - self.ip
            if abs(dv) < 1e-3:
                dD = 0.0 if abs(di) < 1e-3 else (-self.dD if di > 0 else self.dD)
            else:
                g = di / dv + i / max(v, 1e-3)          # dP/dV / V
                if abs(g) < self.eps * i / max(v, 1e-3):
                    dD = 0.0
                else:
                    dD = -self.dD if g > 0 else self.dD  # dP/dV>0 -> raise V -> lower D
        self.vp, self.ip = v, i
        self.D = float(np.clip(self.D + dD, D_MIN, D_MAX))
        return self.D


class PSO:
    """Standard PSO-MPPT on duty cycle; re-initialises when power changes while holding."""
    name = "PSO"
    def __init__(self, n=4, D0=0.5, w=0.3, c1=1.2, c2=1.6, seed=0, tol=0.02, itmax=6):
        self.n, self.D0, self.w, self.c1, self.c2 = n, D0, w, c1, c2
        self.rng = np.random.default_rng(seed); self.tol, self.itmax = tol, itmax
    def _init_swarm(self):
        self.x = np.linspace(0.27, 0.9, self.n) ; self.vel = np.zeros(self.n)
        self.pb = self.x.copy(); self.pbv = np.full(self.n, -1.0)
        self.gb = self.x[0]; self.gbv = -1.0; self.j = 0; self.it = 0; self.hold = False
    def reset(self):
        self._init_swarm(); self.D = self.x[0]; self.ph = None
        return self.D
    def act(self, v, i, p):
        if self.hold:
            self.kh += 1
            if self.kh == 3:                      # reference power after converter settles
                self.ph = p
            elif self.ph is not None and abs(p - self.ph) / max(self.ph, 1) > 0.05:
                self._init_swarm(); self.D = self.x[0]; self.ph = None
                return self.D
            return self.D
        # evaluation of particle j
        if p > self.pbv[self.j]:
            self.pbv[self.j], self.pb[self.j] = p, self.x[self.j]
        if p > self.gbv:
            self.gbv, self.gb = p, self.x[self.j]
        self.j += 1
        if self.j == self.n:
            self.j = 0; self.it += 1
            r1, r2 = self.rng.random(self.n), self.rng.random(self.n)
            self.vel = self.w * self.vel + self.c1 * r1 * (self.pb - self.x) + self.c2 * r2 * (self.gb - self.x)
            self.x = np.clip(self.x + self.vel, D_MIN, D_MAX)
            if np.ptp(self.x) < self.tol or self.it >= self.itmax:
                self.hold = True; self.D = self.gb; self.ph = None; self.kh = 0
                return self.D
        self.D = float(self.x[self.j])
        return self.D
