function R = run_scenario(ctrl, s, G, T, P, model)
% ctrl = 'po' | 'inc' | 'pso' | 'ddpg' ; s = controller state ; model = 'avg' | 'sw'
N = size(G, 1);  R.P = zeros(N,1); R.Pmax = zeros(N,1); R.V = zeros(N,1);
st.v = 0; st.iL = 0; st.t = 0;  D = s.D;  lastkey = '';
for k = 1:N
  key = sprintf('%d_', round([G(k,:) 10*T(k)]));
  if ~strcmp(key, lastkey), [Vg, Ig, Pm] = pv_string_iv(round(G(k,:)), round(T(k)*10)/10, P); lastkey = key; end
  if k == 1, st.v = Vg(end); end
  [st, v, i] = boost_period(st, D, Vg, Ig, P, model);  p = v*i;
  R.P(k) = p; R.Pmax(k) = Pm; R.V(k) = v;
  switch ctrl
    case 'po',   [D, s] = ctrl_po(s, v, i, p);
    case 'inc',  [D, s] = ctrl_inc(s, v, i, p);
    case 'pso',  [D, s] = ctrl_pso(s, v, i, p);
    case 'ddpg', [D, s] = ctrl_ddpg(s, v, i, p);
  end
end
R.eta = 100 * sum(R.P) / sum(R.Pmax);
end
