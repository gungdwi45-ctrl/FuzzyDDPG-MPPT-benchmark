function [D, s] = ctrl_po(s, v, i, p)
% Perturb and observe on duty cycle, fixed step s.dD (init: s = struct('D',D0,'dD',0.01,'pp',[],'dir',1))
if ~isempty(s.pp) && p < s.pp, s.dir = -s.dir; end
s.pp = p;  s.D = min(max(s.D + s.dir*s.dD, 0.20), 0.95);  D = s.D;
end
