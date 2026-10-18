function i = ipv_lookup(v, Vg, Ig)
if v <= Vg(1), i = Ig(1); elseif v >= Vg(end), i = 0; else
  x = (v - Vg(1)) / (Vg(end) - Vg(1)) * (numel(Vg) - 1);  j = floor(x);  f = x - j;
  i = Ig(j+1)*(1 - f) + Ig(j+2)*f;
end
end
