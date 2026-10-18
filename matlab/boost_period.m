function [st, vm, im] = boost_period(st, D, Vg, Ig, P, model)
% One MPPT sampling period of the boost converter. model = 'avg' or 'sw'.
% st.v, st.iL, st.t ; returns V and I averaged over the last 30% of the period.
D = min(max(D, P.Dmin), P.Dmax);
if strcmp(model, 'avg'), dt = P.dt; else, dt = P.dt_sw; end
n = round(P.Ts/dt);  navg = floor(0.3*n);  sv = 0; si = 0;  v = st.v;  iL = st.iL;  t = st.t;
Tsw = 1/P.fsw;
for k = 1:n
  ipv = ipv_lookup(v, Vg, Ig);
  if strcmp(model, 'avg')
    vL = v - P.rL*iL - (1 - D)*P.Vbus;
  else
    if mod(t, Tsw)/Tsw < D
      vL = v - (P.rL + P.Ron)*iL;
    else
      vL = v - P.rL*iL - (P.Vbus + P.Vf);
      if iL <= 0 && vL < 0, vL = 0; end
    end
  end
  iL = max(iL + vL/P.L*dt, 0);
  v = max(v + (ipv - iL)/P.Cin*dt, 0);
  if k > n - navg, sv = sv + v; si = si + ipv_lookup(v, Vg, Ig); end
  t = t + dt;
end
st.v = v; st.iL = iL; st.t = t;  vm = sv/navg;  im = si/navg;
end
