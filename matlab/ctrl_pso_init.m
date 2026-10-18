function s = ctrl_pso_init(seed)
% Tuned PSO: 4 particles, w 0.3, c1 1.2, c2 1.6, tol 0.02, 6 iterations, restart on 5% power change
s.n = 4; s.w = 0.3; s.c1 = 1.2; s.c2 = 1.6; s.tol = 0.02; s.itmax = 6;
s.rng = seed;  s = pso_reset(s);
end
function s = pso_reset(s)
s.x = linspace(0.27, 0.9, s.n); s.vel = zeros(1, s.n); s.pb = s.x; s.pbv = -ones(1, s.n);
s.gb = s.x(1); s.gbv = -1; s.j = 1; s.it = 0; s.hold = false; s.ph = []; s.kh = 0; s.D = s.x(1);
end
