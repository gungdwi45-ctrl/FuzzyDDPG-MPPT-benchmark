% Reproduce the controller comparison in MATLAB (averaged and switching converter models).
% Requires the exported policies in ./policies (from export_policies.py).
clear; P = mppt_params(); S = mppt_scenarios();
models = {'avg', 'sw'};  D0s = [0.3 0.5 0.7];
pol = dir(fullfile('policies', '*.mat'));
names = [{'P&O', 'INC', 'PSO'}, {pol.name}];
eta = nan(numel(names), size(S,1), numel(models));
for c = 1:numel(names)
  for sidx = 1:size(S,1)
    for m = 1:numel(models)
      e = zeros(size(D0s));
      for d = 1:numel(D0s)
        D0 = D0s(d); rand('seed', 0);
        switch names{c}
          case 'P&O', s = struct('D', D0, 'dD', 0.01, 'pp', [], 'dir', 1); ct = 'po';
          case 'INC', s = struct('D', D0, 'dD', 0.01, 'eps', 0.02, 'vp', [], 'ip', []); ct = 'inc';
          case 'PSO', s = ctrl_pso_init(0); ct = 'pso';
          otherwise,  s = ctrl_ddpg_init(fullfile('policies', names{c}), D0); ct = 'ddpg';
        end
        R = run_scenario(ct, s, S{sidx,2}, S{sidx,3}, P, models{m});  e(d) = R.eta;
      end
      eta(c, sidx, m) = mean(e);
    end
    fprintf('%-28s %-12s avg %.1f  sw %.1f\n', names{c}, S{sidx,1}, eta(c,sidx,1), eta(c,sidx,2));
  end
end
save('matlab_results.mat', 'eta', 'names', 'S');
