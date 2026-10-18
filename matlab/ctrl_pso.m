function [D, s] = ctrl_pso(s, v, i, p)
if s.hold
  s.kh = s.kh + 1;
  if s.kh == 3, s.ph = p;
  elseif ~isempty(s.ph) && abs(p - s.ph)/max(s.ph, 1) > 0.05
    s.x = linspace(0.27, 0.9, s.n); s.vel = zeros(1, s.n); s.pb = s.x; s.pbv = -ones(1, s.n);
    s.gb = s.x(1); s.gbv = -1; s.j = 1; s.it = 0; s.hold = false; s.ph = []; s.D = s.x(1);
  end
  D = s.D; return;
end
if p > s.pbv(s.j), s.pbv(s.j) = p; s.pb(s.j) = s.x(s.j); end
if p > s.gbv, s.gbv = p; s.gb = s.x(s.j); end
s.j = s.j + 1;
if s.j > s.n
  s.j = 1; s.it = s.it + 1;
  r1 = rand(1, s.n); r2 = rand(1, s.n);
  s.vel = s.w*s.vel + s.c1*r1.*(s.pb - s.x) + s.c2*r2.*(s.gb - s.x);
  s.x = min(max(s.x + s.vel, 0.20), 0.95);
  if (max(s.x) - min(s.x)) < s.tol || s.it >= s.itmax
    s.hold = true; s.D = s.gb; s.ph = []; s.kh = 0; D = s.D; return;
  end
end
s.D = s.x(s.j);  D = s.D;
end
