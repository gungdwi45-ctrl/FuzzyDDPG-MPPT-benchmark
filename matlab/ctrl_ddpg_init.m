function s = ctrl_ddpg_init(matfile, D0)
% Load an exported actor (export_policies.py) and initialise the observation memory
s.W = load(matfile);  s.D = D0;  s.dDprev = 0;
s.vp = []; s.ip = []; s.pp = []; s.pbest = 0; s.dbest = D0; s.k = 0;
end
