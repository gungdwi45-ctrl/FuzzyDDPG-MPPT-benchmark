function S = mppt_scenarios()
% Test scenarios: cell array {name, G (Nx4), T (Nx1)}, one row per 10 ms period
PS = {[1000 1000 600 300], [1000 800 600 400], [1000 300 300 300], [900 900 200 200], [1000 1000 700 250]};
hold_ = @(g, T, sec) deal(repmat(g, round(sec*100), 1), T*ones(round(sec*100), 1));
S = {};
[G, T] = hold_([1000 1000 1000 1000], 25, 2); S(end+1, :) = {'S1 uniform', G, T};
[G1, T1] = hold_(1000*ones(1,4), 25, 1); [G2, T2] = hold_(400*ones(1,4), 25, 1); [G3, T3] = hold_(800*ones(1,4), 25, 1);
S(end+1, :) = {'S2 steps', [G1; G2; G3], [T1; T2; T3]};
r1 = linspace(1000, 300, 700)'; r2 = linspace(300, 1000, 700)';
G = [1000*ones(50,4); repmat(r1, 1, 4); 300*ones(50,4); repmat(r2, 1, 4)];  S(end+1, :) = {'S3 ramp', G, 25*ones(size(G,1),1)};
for k = 1:5, [G, T] = hold_(PS{k}, 25, 2); S(end+1, :) = {sprintf('PS%d', k), G, T}; end
[G, T] = hold_(PS{4}, 45, 2); S(end+1, :) = {'PS4 45C', G, T};
[G1, T1] = hold_(1000*ones(1,4), 25, 1); [G2, T2] = hold_(PS{4}, 25, 1); [G3, T3] = hold_(PS{3}, 25, 1); [G4, T4] = hold_(700*ones(1,4), 25, 1);
S(end+1, :) = {'S5 dynamic', [G1; G2; G3; G4], [T1; T2; T3; T4]};
end
