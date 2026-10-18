function [Vg, Ig, Pmax, Vmpp] = pv_string_iv(Gs, T, P)
% Series string with bypass diodes; returns I(V) on a monotone voltage grid and the GMPP
Imax = max(P.Isc_stc * max(Gs)/1000 * (1 + P.KI*(T - 25)) * 1.02, 1e-3);
Igr = linspace(0, Imax, 800);  Vs = zeros(size(Igr));
for m = 1:numel(Gs)
  [V, I] = pv_module_iv(Gs(m), T, P);
  Vk = interp1(fliplr(I) + (0:numel(I)-1)*1e-12, fliplr(V), Igr, 'linear', 'extrap');
  Vk(Igr > I(1)) = P.Vbypass;  Vk(Igr <= I(end)) = V(end);
  Vs = Vs + Vk;
end
[Vsorted, ix] = sort(Vs);  Isorted = Igr(ix);
[Vsorted, iu] = unique(Vsorted);  Isorted = Isorted(iu);
Vg = linspace(Vsorted(1), Vsorted(end), 1500);
Ig = interp1(Vsorted, Isorted, Vg);
[Pmax, k] = max(Vg .* Ig);  Vmpp = Vg(k);
end
