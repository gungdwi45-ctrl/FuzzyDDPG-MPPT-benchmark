function [V, I] = pv_module_iv(G, T, P)
% I-V curve of one module on V in [0, Voc] (single-diode model, Newton iteration)
if G < 1, V = [0 1e-3]; I = [0 0]; return; end
Tk = T + 273.15; Tr = 298.15;
a = P.n * P.Ns * P.kB * Tk / P.q;  a_r = P.n * P.Ns * P.kB * Tr / P.q;
Isc = P.Isc_stc * (G / 1000) * (1 + P.KI * (T - 25));
I0r = (P.Isc_stc * (1 + P.Rs/P.Rsh) - P.Voc_stc/P.Rsh) / (exp(P.Voc_stc/a_r) - exp(P.Isc_stc*P.Rs/a_r));
I0 = I0r * (Tk/Tr)^3 * exp(P.q*P.Eg/(P.n*P.kB) * (1/Tr - 1/Tk));
Iph = Isc * (1 + P.Rs/P.Rsh);
Voc = a * log(Iph/I0 + 1);
for k = 1:30
  f = Iph - I0*(exp(Voc/a) - 1) - Voc/P.Rsh;  df = -I0*exp(Voc/a)/a - 1/P.Rsh;  Voc = Voc - f/df;
end
V = linspace(0, Voc, 300);  I = Iph * ones(size(V));
for k = 1:50
  e = exp((V + I*P.Rs)/a);
  f = Iph - I0*(e - 1) - (V + I*P.Rs)/P.Rsh - I;
  df = -I0*e*P.Rs/a - P.Rs/P.Rsh - 1;
  I = I - f./df;
end
I(end) = 0;  I = max(I, 0);
end
