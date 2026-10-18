"""DDPG / Fuzzy-DDPG (static or adaptive membership functions) for MPPT."""
import numpy as np, torch, torch.nn as nn, copy, time
from pvenv import PVSystem, D_MIN, D_MAX, P_STC_STRING

torch.set_num_threads(1)
OBS_DIM, A_MAX = 10, 0.15


class ObsBuilder:
    """Sensor-free observation from V, I (and the controller's own D).
    Keeps a short memory (best power seen since last detected irradiance change)."""
    def __init__(self):
        self.reset(0.5)

    def reset(self, D0):
        self.vp = self.ip = self.pp = None
        self.D = D0; self.dDprev = 0.0
        self.pbest = 0.0; self.dbest = D0; self.k = 0

    def detect_change(self, v, i):
        if self.vp is None:
            return True
        return abs(v - self.vp) / max(self.vp, 1) < 0.03 and abs(i - self.ip) / max(self.ip, 0.2) > 0.08 \
            and abs(self.dDprev) < 0.03

    def __call__(self, v, i, p):
        if self.detect_change(v, i):
            self.pbest, self.dbest, self.k = 0.0, self.D, 0
        if p > self.pbest:
            self.pbest, self.dbest = p, self.D
        dv = 0 if self.vp is None else (v - self.vp)
        di = 0 if self.ip is None else (i - self.ip)
        dp = 0 if self.pp is None else (p - self.pp)
        o = np.array([v / 150, i / 9, p / P_STC_STRING, dv / 20, di / 2, dp / 200,
                      self.D, self.pbest / P_STC_STRING, (self.dbest - self.D) * 2, min(self.k / 50, 1.0)],
                     dtype=np.float32)
        self.vp, self.ip, self.pp = v, i, p
        self.k += 1
        return o

    def apply(self, a):
        """a in [-1,1] -> new duty."""
        dD = float(a) * A_MAX
        Dn = float(np.clip(self.D + dD, D_MIN, D_MAX))
        self.dDprev = Dn - self.D
        self.D = Dn
        return Dn


class Fuzzify(nn.Module):
    """Gaussian membership layer. adaptive=True -> centres and widths are trained end-to-end."""
    def __init__(self, n_in, n_mf=5, adaptive=True, lo=-1.0, hi=1.5):
        super().__init__()
        c = torch.linspace(lo, hi, n_mf).repeat(n_in, 1)
        s = torch.full((n_in, n_mf), (hi - lo) / (n_mf - 1) / 1.2)
        if adaptive:
            self.c = nn.Parameter(c); self.log_s = nn.Parameter(s.log())
        else:
            self.register_buffer("c", c); self.register_buffer("log_s", s.log())
        self.out_dim = n_in * (n_mf + 1)

    def forward(self, x):
        s = self.log_s.exp().clamp(min=0.02)
        mu = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.c) / s) ** 2)
        mu = mu / (mu.sum(-1, keepdim=True) + 1e-6)            # normalised firing strengths
        return torch.cat([x, mu.flatten(1)], dim=1)


def mlp(i, o, h=64, out_act=None):
    L = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)]
    if out_act: L.append(out_act)
    return nn.Sequential(*L)


class Actor(nn.Module):
    def __init__(self, fuzzy):
        super().__init__()
        self.f = Fuzzify(OBS_DIM, adaptive=(fuzzy == "adaptive")) if fuzzy else None
        d = self.f.out_dim if self.f else OBS_DIM
        self.net = mlp(d, 1, out_act=nn.Tanh())
    def forward(self, s):
        return self.net(self.f(s) if self.f else s)


class Critic(nn.Module):
    def __init__(self, fuzzy):
        super().__init__()
        self.f = Fuzzify(OBS_DIM, adaptive=(fuzzy == "adaptive")) if fuzzy else None
        d = (self.f.out_dim if self.f else OBS_DIM) + 1
        self.net = mlp(d, 1)
    def forward(self, s, a):
        return self.net(torch.cat([self.f(s) if self.f else s, a], 1))


def reward(kind, p, pmax, dD, k_since, iL_peak, i_avg):
    eta = p / max(pmax, 1.0)
    if kind == "conv":                       # efficiency + oscillation penalty (conventional)
        return eta - 1.0 - 0.5 * abs(dD)
    # proposed: time-weighted efficiency error + steady-state oscillation + current-overshoot ratio
    w = 1.0 + 2.0 * np.exp(-k_since / 15.0)
    osc = 1.0 * abs(dD) if eta > 0.97 else 0.0
    os_ratio = max(0.0, iL_peak / max(i_avg, 0.5) - 1.2)
    return -w * (1.0 - eta) - osc - 0.2 * os_ratio


def random_pattern(rng):
    """Uniform (30%), two-level shading (50%) or three-level shading (20%)."""
    u = rng.random(); gh = rng.uniform(500, 1000)
    if u < 0.3:
        return [rng.uniform(200, 1000)] * 4
    if u < 0.8:
        k = int(rng.integers(1, 4)); gl = rng.uniform(100, 0.6 * gh)
        return [gh] * k + [gl] * (4 - k)
    gm = rng.uniform(0.4 * gh, 0.8 * gh); gl = rng.uniform(100, 0.6 * gm)
    k1 = int(rng.integers(1, 3)); k2 = int(rng.integers(1, 4 - k1))
    return [gh] * k1 + [gm] * k2 + [gl] * (4 - k1 - k2)


class TrainEnv:
    def __init__(self, rkind, rng):
        self.rkind, self.rng = rkind, rng
        self.plant = PVSystem(); self.ob = ObsBuilder()

    def reset(self):
        r = self.rng
        self.T = r.uniform(20, 50)
        self.p1, self.p2 = random_pattern(r), random_pattern(r)
        self.n = 200; self.tc = int(r.integers(60, 140)) if r.random() < 0.7 else 10**9
        self.t = 0; self.k_since = 0
        self.plant.reset(); self.plant.set_irradiance(self.p1, self.T)
        D0 = r.uniform(0.25, 0.8); self.ob.reset(D0)
        v, i, p = self.plant.step(D0)
        return self.ob(v, i, p)

    def step(self, a):
        D = self.ob.apply(a); dD = self.ob.dDprev
        self.t += 1; self.k_since += 1
        if self.t == self.tc:
            self.plant.set_irradiance(self.p2, self.T); self.k_since = 0
        v, i, p = self.plant.step(D)
        r = reward(self.rkind, p, self.plant.Pmax, dD, self.k_since, self.plant.iL_peak, i)
        return self.ob(v, i, p), r, self.t >= self.n


class Buffer:
    def __init__(self, n=200000):
        self.s = np.zeros((n, OBS_DIM), np.float32); self.a = np.zeros((n, 1), np.float32)
        self.r = np.zeros((n, 1), np.float32); self.s2 = np.zeros((n, OBS_DIM), np.float32)
        self.d = np.zeros((n, 1), np.float32); self.n = n; self.i = 0; self.full = False
    def add(self, s, a, r, s2, d):
        j = self.i; self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (j + 1) % self.n; self.full |= self.i == 0
    def sample(self, b, rng):
        idx = rng.integers(0, self.n if self.full else self.i, b)
        return [torch.from_numpy(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d)]


def train(fuzzy=None, rkind="conv", seed=0, episodes=400, gamma=0.97, log=None, lr_a=3e-4, lr_c=1e-3):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    actor, critic = Actor(fuzzy), Critic(fuzzy)
    at, ct = copy.deepcopy(actor), copy.deepcopy(critic)
    oa = torch.optim.Adam(actor.parameters(), lr_a); oc = torch.optim.Adam(critic.parameters(), lr_c)
    env = TrainEnv(rkind, rng); buf = Buffer(); steps = 0; hist = []
    t0 = time.time()
    for ep in range(episodes):
        s = env.reset(); done = False; R = 0; effs = []
        sig = max(0.05, 0.4 * (1 - ep / (0.7 * episodes)))
        while not done:
            if steps < 3000:
                a = rng.uniform(-1, 1)
            else:
                with torch.no_grad():
                    a = actor(torch.from_numpy(s)[None])[0, 0].item()
                a = float(np.clip(a + rng.normal(0, sig), -1, 1))
            s2, r, done = env.step(a)
            buf.add(s, [a], r, s2, float(done)); s = s2; R += r; steps += 1
            effs.append(env.plant.Pmax)
            if steps >= 1000:
                S, A, Rr, S2, Dn = buf.sample(128, rng)
                with torch.no_grad():
                    y = Rr + gamma * (1 - Dn) * ct(S2, at(S2))
                lc = ((critic(S, A) - y) ** 2).mean()
                oc.zero_grad(); lc.backward(); oc.step()
                la = -critic(S, actor(S)).mean()
                oa.zero_grad(); la.backward(); oa.step()
                with torch.no_grad():
                    for net, tgt in ((actor, at), (critic, ct)):
                        for p_, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(0.995).add_(0.005 * p_)
        hist.append(R)
        if log and (ep + 1) % 50 == 0:
            log(f"[{fuzzy}/{rkind}/s{seed}] ep {ep+1} R(avg50)={np.mean(hist[-50:]):.2f} t={time.time()-t0:.0f}s")
    return actor, np.array(hist)


class RLController:
    """Deterministic deployment of a trained actor."""
    def __init__(self, actor, name, D0=0.5):
        self.actor, self.name, self.D0 = actor.eval(), name, D0
        self.ob = ObsBuilder(); self.first = True
    def reset(self):
        self.ob.reset(self.D0); return self.D0
    def act(self, v, i, p):
        s = self.ob(v, i, p)
        with torch.no_grad():
            a = self.actor(torch.from_numpy(s)[None])[0, 0].item()
        return self.ob.apply(a)
