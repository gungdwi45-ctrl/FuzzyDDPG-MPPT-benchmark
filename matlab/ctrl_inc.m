function [D, s] = ctrl_inc(s, v, i, p)
% Incremental conductance (init: s = struct('D',D0,'dD',0.01,'eps',0.02,'vp',[],'ip',[]))
if isempty(s.vp), dD = s.dD; else
  dv = v - s.vp;  di = i - s.ip;
  if abs(dv) < 1e-3
    if abs(di) < 1e-3, dD = 0; elseif di > 0, dD = -s.dD; else, dD = s.dD; end
  else
    g = di/dv + i/max(v, 1e-3);
    if abs(g) < s.eps*i/max(v, 1e-3), dD = 0; elseif g > 0, dD = -s.dD; else, dD = s.dD; end
  end
end
s.vp = v; s.ip = i;  s.D = min(max(s.D + dD, 0.20), 0.95);  D = s.D;
end
