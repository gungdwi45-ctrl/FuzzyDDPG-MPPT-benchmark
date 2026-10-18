function P = mppt_params()
% Plant and converter parameters (identical to the Python package, pvenv.py)
P.q = 1.602176634e-19; P.kB = 1.380649e-23;
P.Ns = 60; P.n = 1.18307; P.Rs = 0.23997; P.Rsh = 894.22;   % CS6P-250P single-diode fit
P.Isc_stc = 8.87; P.Voc_stc = 37.2; P.KI = 0.00053;          % Isc temp. coeff. 0.053 %/degC
P.Eg = 1.12; P.Vbypass = -0.6; P.Nmod = 4;
P.L = 1e-3; P.rL = 0.05; P.Cin = 220e-6; P.Vbus = 200;     % averaged boost
P.dt = 2e-6; P.Ts = 0.01; P.Dmin = 0.20; P.Dmax = 0.95;
P.fsw = 20e3; P.dt_sw = 0.25e-6; P.Ron = 0.05; P.Vf = 0.7;  % switching model
P.Pstc_string = 4 * 249.83;
end
