def charger(power_fromheater, temp_frompy):
    import matlab.engine as me
    import numpy as np

    eng = me.start_matlab()
    s = eng.genpath("C:/Users/taimo/OneDrive - Universite de Lorraine/Bureau/DENSYS/Master Thesis/Thesis Work/Python Code/PDE Solution")
    eng.addpath(s, nargout=0)

    T = 273.15 + (750 + 320)/2
    T_inf = 9.8 + 273
    T_in = 750 + 273
    e = 0.4
    ps = 2083
    cs = 827.69 + 0.3339*T

    pa = 0.4369
    ca = 1100
    ka = 0.05769

    dp = 0.035
    rho_c_eff = e*pa*ca + ps*cs*(1-e)
    eff0 = 0.85
    E_des = 30*3.6e9
    DT_TES = (750-300)

    V = E_des/(rho_c_eff*DT_TES*eff0)
    asp_ratio = 1.25
    D = ((4*V)/(np.pi*asp_ratio))**(1/3)
    H = asp_ratio*D
    A = np.pi*(D**2)/4
    heater_eff = 0.95

    m_flow = power_fromheater*1e6/(ca*heater_eff*(T_in - T_inf))
    G = m_flow/A
    alpha_p = (700/(6*(1-e)))*(G**0.76)*(dp**0.24)
    h = 700*(G/dp)**0.76

    U = 0.678

    m = 0
    dt = 60
    n_mesh = 500 + 1
    t_max  = 3600
    n_step = t_max/dt + 1
    print('hello ji', n_step)
    xmesh = np.linspace(0, H, n_mesh)
    tstep = np.linspace(0, t_max, int(n_step))
    options = eng.odeset('RelTol', 1e-1, 'AbsTol', 1e-2)
    sol = eng.pdepe(m, Charge_eq, ini_cd, boundary, xmesh, tstep, options, nargout = 1)

    print(len(sol))