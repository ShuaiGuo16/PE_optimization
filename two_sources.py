def thermal_distribution_maxT_twosources(X, Data):
    
    # Unpack design variables
    Q_first, Q_second, d, b, L, c, L_duct, n, t, Xc_first, Yc_first, Xc_second, Yc_second = X

    # Unpack data
    Ti, c_source, d_source, c_module, d_module = Data

    # Define constants and calculate parameters
    lambda_air = 0.02551
    lambda_HS = 237
    visc_air_K = 156.2e-7
    visc_air = 184.9e-7
    density_air = 1.184
    density_Al = 2700
    c_air = 1007
    Pr = visc_air * c_air / lambda_air

    # Fan properties
    Fan_height = 40e-3
    Fan_weight = 50.8e-3
    Fan_power = 17.4
    Fan_A0 = 1081;
    Fan_A1 = -1.603e4;
    Fan_A2 = -2.797e6;
    Fan_A3 = 2.061e8;
    Fan_A4 = -5.034e9;
    Fan_A5 = 3.923e10;
    VF_max = 0.0466 / 3
    N_fan = ceil(b / Fan_height)
    
    width = b / n
    s = width - t
    x_fan = c * L_duct / (Fan_height - c)
    alpha = atan(c / x_fan)

    # Pressure drop and thermal calculations
    V = np.arange(0.0005, VF_max * N_fan + 0.0005, 0.0005)
    P_hs, P_duct, P_acc, P_fan = [], [], [], []

    for V_dot in V:
        dh = 2 * s * c / (s + c)
        Uhs = V_dot / (n * s * c)
        EPS = s / c
        Kse = (1 - (1 - (n + 1) * t / b) ** 2) ** 2
        Ksc = 0.42 * (1 - (1 - (n + 1) * t / b) ** 2)
        fRe_fd = 12 / (sqrt(EPS) * (1 + EPS) * (1 - 192 * EPS * tanh(pi / 2 / EPS) / pi ** 5))
        fRe = sqrt(11.8336 * V_dot / L / n / visc_air_K + fRe_fd ** 2)
        fapp = n * visc_air_K * sqrt(c * s) * fRe / V_dot
        P_hs.append((fapp * L / dh + Kse + Ksc) * density_air / 2 * Uhs ** 2)

        dh_duct = 2 * b * (b + c) / (3 * b + c)
        L_duct_calc = (b - c) / 2 / tan(alpha)
        K_venturi = 0.2
        U_duct = V_dot / b / c
        EPS_duct = (b + c) / 2 / c
        fRe_fd_duct = 12 / (sqrt(EPS_duct) * (1 + EPS_duct) * (1 - 192 * EPS_duct * tanh(pi / 2 / EPS_duct) / pi ** 5))
        fapp_duct = visc_air_K * sqrt(b * (b + c)) / sqrt(2) / V_dot * sqrt(11.8336 * V_dot / L_duct_calc / visc_air_K + fRe_fd_duct ** 2)
        P_duct.append((fapp_duct * L_duct_calc / dh_duct * 0.25 + K_venturi) * density_air / 2 * U_duct ** 2)

        # Acceleration pressure drop
        P_acc.append(density_air / 2 * V_dot ** 2 * (1 / (n * s * c) ** 2 - 1 / b ** 4))
        
        # Fan pressure curve
        P_fan.append(Fan_A0 + Fan_A1 * (V_dot / N_fan * 3) + Fan_A2 * (V_dot / N_fan * 3) ** 2 + Fan_A3 * (V_dot / N_fan * 3) ** 3 + Fan_A4 * (V_dot / N_fan * 3) ** 4 + Fan_A5 * (V_dot / N_fan * 3) ** 5)

    # Total pressure drop
    P_tot = np.array(P_hs) + np.array(P_duct) + np.array(P_acc)
    P_fan = np.array(P_fan)

    # Interpolate the curves
    interp_fan = interp1d(V, P_fan, kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_tot = interp1d(V, P_tot, kind='cubic', bounds_error=False, fill_value="extrapolate")
    def diff_function(x):
        return interp_fan(x) - interp_tot(x)
    
    intersection_points = []
    for i in range(len(V) - 1):
        if diff_function(V[i]) * diff_function(V[i+1]) < 0:
            root_result = root_scalar(diff_function, bracket=[V[i], V[i+1]], method='brentq')
            if root_result.converged:
                intersection_points.append(root_result.root)
    
    # Check if intersection_points is empty
    if not intersection_points:
        print(f"No valid fan operating point! @d[mm]= {d:.4f}, b[mm]= {b:.4f}, L[mm]= {L:.4f}")
        print(f"c[mm]= {c:.4f}, L_duct[mm]= {L_duct:.4f}, n = {n}, t[mm]= {t:.4f}")
    else:
        V_cal = intersection_points[0]  # Proceed with the first intersection point

    # Fluid Dynamic Entry Length Calculation
    Lh_plus = 0.0822 * EPS * (1 + EPS) ** 2 * (1 - 192 * EPS * np.tanh(np.pi / 2 / EPS) / np.pi ** 5)
    Lh = Lh_plus * V_cal / n / visc_air_K
    
    # Thermal resistance
    Ahs = L * b
    Rth_d = d / lambda_HS / Ahs
    
    dh = 2 * s * c / (s + c)
    
    C1, C2, C3, C4 = 3.24, 1.5, 0.409, 2
    Cons = -0.3
    m = 2.27 + 1.65 * Pr ** 1.3
    z_star = L * n * visc_air_K / Pr / V_cal
    
    fRe_fd_th = 12 / (np.sqrt(EPS) * (1 + EPS) * (1 - 192 * EPS * np.tanh(np.pi / 2 / EPS) / np.pi ** 5))
    fRe_th = np.sqrt(11.8336 * V_cal / L / n / visc_air_K + fRe_fd_th ** 2)
    f_Pr = 0.564 / ((1 + (1.664 * Pr ** (1/6)) ** (9/2)) ** (2/9))
    Nu = ((C4 * f_Pr / np.sqrt(z_star)) ** m + ((C1 * fRe_th / 8 / np.sqrt(np.pi) / EPS ** Cons) ** 5 + (C2 * C3 * (fRe_th / z_star) ** (1/3)) ** 5) ** (m/5)) ** (1/m)
    
    h = Nu * lambda_air / dh
    
    eff_fin = np.tanh(np.sqrt(2 * h * (t + L) / lambda_HS / t / L) * c) / np.sqrt(2 * h * (t + L) / lambda_HS / t / L) / c
    Aeff = n * (2 * c * eff_fin + s) * L
    Rth_conv = 1 / (density_air * c_air * V_cal * (1 - np.exp(-h * Aeff / density_air / c_air / V_cal)))

    k = 237  # Thermal conductivity of heatsink (Aluminum)
    density_air = 1.184
    c_air = 1007
    
    Max_iter = 50
    
    Temperature_surface = []

    # Iterate over the surface
    for x in np.arange(0.001, b, 0.005):
        for y in np.arange(0.001, L, 0.005):
            m_dot = density_air * V_cal  # Mass flow
            A0 = (Q_first + Q_second) * (d / (k * b * L) + Rth_conv)
            T_diff = A0
            
            # Iterate for m1 and n1 series
            for m1 in range(1, Max_iter + 1):
                lamtha = m1 * pi / b
                phi_m = (lamtha * sinh(lamtha * d) + h / k * cosh(lamtha * d)) / (lamtha * cosh(lamtha * d) + h / k * sinh(lamtha * d))
                Am_1 = 2 * Q_first * (sin((2 * Xc_first + c_source) / 2 * lamtha) - sin((2 * Xc_first - c_source) / 2 * lamtha)) / (b * L * c_source * k * lamtha ** 2 * phi_m)
                Am_2 = 2 * Q_second * (sin((2 * Xc_second + c_source) / 2 * lamtha) - sin((2 * Xc_second - c_source) / 2 * lamtha)) / (b * L * c_source * k * lamtha ** 2 * phi_m)
                T_diff += cos(lamtha * x) * (Am_1 + Am_2)
    
            for n1 in range(1, Max_iter + 1):
                thelta = n1 * pi / L
                phi_n = (thelta * sinh(thelta * d) + h / k * cosh(thelta * d)) / (thelta * cosh(thelta * d) + h / k * sinh(thelta * d))
                An_1 = 2 * Q_first * (sin((2 * Yc_first + d_source) / 2 * thelta) - sin((2 * Yc_first - d_source) / 2 * thelta)) / (b * L * d_source * k * thelta ** 2 * phi_m)
                An_2 = 2 * Q_second * (sin((2 * Yc_second + d_source) / 2 * thelta) - sin((2 * Yc_second - d_source) / 2 * thelta)) / (b * L * d_source * k * thelta ** 2 * phi_m)
                T_diff += cos(thelta * y) * (An_1 + An_2)
    
            # Double summation for m_mn and n_mn series
            for m_mn in range(1, 51):
                for n_mn in range(1, 51):
                    lamtha_mn = m_mn * pi / b
                    thelta_mn = n_mn * pi / L
                    beta = sqrt(lamtha_mn ** 2 + thelta_mn ** 2)
                    phi_mn = (beta * sinh(beta * d) + h / k * cosh(beta * d)) / (beta * cosh(beta * d) + h / k * sinh(beta * d))
                    Amn_1 = 16 * Q_first * cos(lamtha_mn * Xc_first) * sin(lamtha_mn * c_source / 2) * cos(thelta_mn * Yc_first) * sin(thelta_mn * d_source / 2) / (b * L * c_source * d_source * k * beta * lamtha_mn * thelta_mn * phi_mn)
                    Amn_2 = 16 * Q_second * cos(lamtha_mn * Xc_second) * sin(lamtha_mn * c_source / 2) * cos(thelta_mn * Yc_second) * sin(thelta_mn * d_source / 2) / (b * L * c_source * d_source * k * beta * lamtha_mn * thelta_mn * phi_mn)
                    T_diff += cos(lamtha_mn * x) * cos(thelta_mn * y) * (Amn_1 + Amn_2)
    
            Tf = Ti + (Q_first + Q_second) * y / m_dot / c_air / L
            T_diff += Tf
            Temperature_surface.append([x, y, T_diff])
    
    # Extract Tz (temperature values) and find maximum temperature
    Tz = np.array(Temperature_surface)[:, 2]
    T_max = np.max(Tz)
    T_max += Q_first*0.117

    return T_max