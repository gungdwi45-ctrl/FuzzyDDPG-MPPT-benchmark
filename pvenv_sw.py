"""Switching (cycle-by-cycle) boost converter model for validating the averaged model.
20 kHz PWM, ideal gate timing, MOSFET Ron, diode forward drop, DCM handled."""
import numpy as np
from numba import njit
from pvenv import PVSystem, _ipv, TS, L_IND, R_L, C_IN, V_BUS, D_MIN, D_MAX

F_SW = 20e3
DT_SW = 0.25e-6
R_ON = 0.05
V_F = 0.7


@njit(cache=True)
def simulate_period_sw(v, iL, D, Vg, Ig, nsteps, dt, L, rL, C, Vbus, fsw, Ron, Vf, t0):
    sv = 0.0; si = 0.0; cnt = 0; ipk = 0.0
    navg = int(nsteps * 0.3)
    Tsw = 1.0 / fsw
    t = t0
    for k in range(nsteps):
        ipv = _ipv(v, Vg, Ig)
        phase = (t % Tsw) / Tsw
        if phase < D:                      # switch on
            vL = v - (rL + Ron) * iL
        else:                              # switch off, diode conducts if iL > 0
            vL = v - rL * iL - (Vbus + Vf)
            if iL <= 0.0 and vL < 0.0:
                vL = 0.0
        iL += vL / L * dt
        if iL < 0.0:
            iL = 0.0
        v += (ipv - iL) / C * dt
        if v < 0.0:
            v = 0.0
        if iL > ipk:
            ipk = iL
        if k >= nsteps - navg:
            sv += v; si += _ipv(v, Vg, Ig); cnt += 1
        t += dt
    return v, iL, sv / cnt, si / cnt, ipk, t


class PVSystemSW(PVSystem):
    def reset(self, D0=0.5):
        super().reset(D0); self.t = 0.0

    def step(self, D):
        D = float(np.clip(D, D_MIN, D_MAX)); self.D = D
        n = int(round(TS / DT_SW))
        self.v, self.iL, vm, im, self.iL_peak, self.t = simulate_period_sw(
            self.v, self.iL, D, self.Vg, self.Ig, n, DT_SW, L_IND, R_L, C_IN, V_BUS, F_SW, R_ON, V_F, self.t)
        return vm, im, vm * im
