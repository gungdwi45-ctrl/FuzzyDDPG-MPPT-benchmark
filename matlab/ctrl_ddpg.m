function [D, s] = ctrl_ddpg(s, v, i, p)
% Sensor-free DDPG / Fuzzy-DDPG policy (same observation as rl.py ObsBuilder)
Pstc = 4 * 249.83;
chg = isempty(s.vp) || (abs(v - s.vp)/max(s.vp, 1) < 0.03 && abs(i - s.ip)/max(s.ip, 0.2) > 0.08 && abs(s.dDprev) < 0.03);
if chg, s.pbest = 0; s.dbest = s.D; s.k = 0; end
if p > s.pbest, s.pbest = p; s.dbest = s.D; end
if isempty(s.vp), dv = 0; di = 0; dp = 0; else, dv = v - s.vp; di = i - s.ip; dp = p - s.pp; end
o = [v/150, i/9, p/Pstc, dv/20, di/2, dp/200, s.D, s.pbest/Pstc, (s.dbest - s.D)*2, min(s.k/50, 1)];
s.vp = v; s.ip = i; s.pp = p; s.k = s.k + 1;
W = s.W;  x = o(:);
if isfield(W, 'c')                       % fuzzy layer: normalised Gaussian firing strengths
  sg = max(exp(W.log_s), 0.02);
  mu = exp(-0.5*((x - W.c)./sg).^2);     % 10 x 5
  mu = mu ./ (sum(mu, 2) + 1e-6);
  x = [x; reshape(mu.', [], 1)];         % row-major flatten, same as torch .flatten(1)
end
h = max(W.W0*x + W.b0(:), 0);  h = max(W.W1*h + W.b1(:), 0);  a = tanh(W.W2*h + W.b2(:));
Dn = min(max(s.D + a*0.15, 0.20), 0.95);
s.dDprev = Dn - s.D;  s.D = Dn;  D = Dn;
end
