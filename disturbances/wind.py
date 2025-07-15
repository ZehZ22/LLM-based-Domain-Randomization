import numpy as np
from ship_params import ShipParams


# === 风力模型 1：基于 Isherwood 经验公式 ===
def wind_func1(time, V_wind, Psi_wind, eta, nu, ship: ShipParams):
    gamma_r = Psi_wind - eta[2]  # 相对风向角
    V_r = V_wind  # 简化：直接用平均风速作为相对风速

    rho_a = ship.rho_air
    gamma_r_deg = np.degrees(gamma_r)

    # 定义风力系数表格数据
    CX_data = np.array([
        [0, 2.152, -5.00, 0.243, -0.164, 0, 0, 0],
        [10, 1.714, -3.33, 0.145, -0.121, 0, 0, 0],
        [20, 1.818, -3.97, 0.211, -0.143, 0, 0, 0.033],
        [30, 1.965, -4.81, 0.243, -0.154, 0, 0, 0.041],
        [40, 2.333, -5.99, 0.247, -0.190, 0, 0, 0.042],
        [50, 1.726, -6.54, 0.189, -0.173, 0.348, 0, 0.048],
        [60, 0.913, -4.68, 0, -0.104, 0.482, 0, 0.052],
        [70, 0.457, -2.88, 0, -0.068, 0.346, 0, 0.043],
        [80, 0.341, -0.91, 0, -0.031, 0, 0, 0.032],
        [90, 0.355, 0, 0, 0, -0.247, 0, 0.018],
        [100, 0.601, 0, 0, 0, -0.372, 0, -0.020],
        [110, 0.651, 1.29, 0, 0, -0.582, 0, -0.031],
        [120, 0.564, 2.54, 0, 0, -0.748, 0, -0.024],
        [130, -0.142, 3.58, 0, 0.047, -0.700, 0, -0.028],
        [140, -0.677, 3.64, 0, 0.069, -0.529, 0, -0.032],
        [150, -0.723, 3.14, 0, 0.064, -0.475, 0, -0.032],
        [160, -2.148, 2.56, 0, 0.081, 0, 1.27, -0.027],
        [170, -2.707, 3.97, -0.175, 0.126, 0, 1.81, 0],
        [180, -2.529, 3.76, -0.174, 0.128, 0, 1.55, 0]
    ])

    CY_data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0.096, 0.22, 0, 0, 0, 0, 0],
        [20, 0.176, 0.71, 0, 0, 0, 0, 0],
        [30, 0.225, 1.38, 0, 0.023, 0, -0.29, 0],
        [40, 0.329, 1.82, 0, 0.043, 0, -0.59, 0],
        [50, 1.164, 1.26, 0.121, 0, -0.242, -0.95, 0],
        [60, 1.163, 0.96, 0.101, 0, -0.177, -0.88, 0],
        [70, 0.916, 0.53, 0.069, 0, 0, -0.65, 0],
        [80, 0.844, 0.55, 0.082, 0, 0, -0.54, 0],
        [90, 0.889, 0, 0.138, 0, 0, -0.66, 0],
        [100, 0.799, 0, 0.155, 0, 0, -0.55, 0],
        [110, 0.797, 0, 0.151, 0, 0, -0.55, 0],
        [120, 0.996, 0, 0.184, 0, -0.212, -0.66, 0.34],
        [130, 1.014, 0, 0.191, 0, -0.280, -0.69, 0.44],
        [140, 0.784, 0, 0.166, 0, -0.209, -0.53, 0.38],
        [150, 0.536, 0, 0.176, -0.029, -0.163, 0, 0.27],
        [160, 0.251, 0, 0.106, -0.022, 0, 0, 0],
        [170, 0.125, 0, 0.046, -0.012, 0, 0, 0],
        [180, 0, 0, 0, 0, 0, 0, 0]
    ])

    CN_data = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [10, 0.0596, 0.061, 0, 0, 0, -0.074],
        [20, 0.1106, 0.204, 0, 0, 0, -0.170],
        [30, 0.2258, 0.245, 0, 0, 0, -0.380],
        [40, 0.2017, 0.457, 0, 0.0067, 0, -0.472],
        [50, 0.1759, 0.573, 0, 0.0118, 0, -0.523],
        [60, 0.1925, 0.480, 0, 0.0115, 0, -0.546],
        [70, 0.2133, 0.315, 0, 0.0081, 0, -0.526],
        [80, 0.1827, 0.254, 0, 0.0053, 0, -0.443],
        [90, 0.2627, 0, 0, 0, 0, -0.508],
        [100, 0.2102, 0, -0.0195, 0, 0.0335, -0.492],
        [110, 0.1567, 0, -0.0258, 0, 0.0497, -0.457],
        [120, 0.0801, 0, -0.0311, 0, 0.0740, -0.396],
        [130, -0.0189, 0, -0.0488, 0.0101, 0.1128, -0.420],
        [140, 0.0256, 0, -0.0422, 0.0100, 0.0889, -0.463],
        [150, 0.0552, 0, -0.0381, 0.0109, 0.0689, -0.476],
        [160, 0.0881, 0, -0.0306, 0.0091, 0.0366, -0.415],
        [170, 0.0851, 0, -0.0122, 0.0025, 0, -0.220],
        [180, 0, 0, 0, 0, 0, 0]
    ])

    # === 插值函数 ===
    def interp(data, col):
        return np.interp(gamma_r_deg, data[:, 0], data[:, col])

    # CX 插值
    A0 = interp(CX_data, 1)
    A1 = interp(CX_data, 2)
    A2 = interp(CX_data, 3)
    A3 = interp(CX_data, 4)
    A4 = interp(CX_data, 5)
    A5 = interp(CX_data, 6)
    A6 = interp(CX_data, 7)

    # CY 插值
    B0 = interp(CY_data, 1)
    B1 = interp(CY_data, 2)
    B2 = interp(CY_data, 3)
    B3 = interp(CY_data, 4)
    B4 = interp(CY_data, 5)
    B5 = interp(CY_data, 6)
    B6 = interp(CY_data, 7)

    # CN 插值
    C0 = interp(CN_data, 1)
    C1 = interp(CN_data, 2)
    C2 = interp(CN_data, 3)
    C3 = interp(CN_data, 4)
    C4 = interp(CN_data, 5)
    C5 = interp(CN_data, 6)

    # === 风力系数 ===
    CX = -(A0 + A1 * 2 * ship.ALw / ship.Loa ** 2
           + A2 * 2 * ship.AFw / ship.B ** 2
           + A3 * (ship.Loa / ship.B)
           + A4 * (ship.S / ship.Loa)
           + A5 * (ship.C / ship.Loa)
           + A6 * ship.M)

    CY = B0 + B1 * 2 * ship.ALw / ship.Loa ** 2 \
         + B2 * 2 * ship.AFw / ship.B ** 2 \
         + B3 * (ship.Loa / ship.B) \
         + B4 * (ship.S / ship.Loa) \
         + B5 * (ship.C / ship.Loa) \
         + B6 * (ship.A_SS / ship.ALw)

    CN = C0 + C1 * 2 * ship.ALw / ship.Loa ** 2 \
         + C2 * 2 * ship.AFw / ship.B ** 2 \
         + C3 * (ship.Loa / ship.B) \
         + C4 * (ship.S / ship.Loa) \
         + C5 * (ship.C / ship.Loa)

    # 风力和力矩计算
    q = 0.5 * rho_a * V_r ** 2  # 动压项
    tau_X = q * CX * ship.AFw
    tau_Y = q * CY * ship.ALw
    tau_N = q * CN * ship.ALw * ship.Loa

    tau_wind = [tau_X, tau_Y, tau_N]

    return tau_wind, CX, CY, CN


# === 风力模型 2：基于Xie的动态风模型 ===
def wind_func2(time, V_wind, Psi_wind, eta, nu, ship: ShipParams):
    A_D = np.pi / 180
    rho_a = ship.rho_air

    z = 19.4
    Psi_r = Psi_wind - eta[2]
    u_r = V_wind * np.cos(Psi_r) - nu[0]
    v_r = V_wind * np.sin(Psi_r) - nu[1]
    V_r = np.sqrt(u_r ** 2 + v_r ** 2)

    N = 100
    omega = np.linspace(0.001, 1, N)
    delta_omega = omega[1] - omega[0]

    alpha = -0.125 if z <= 20 else -0.275
    sigma_u = 0.15 * (z / 20) ** alpha * V_r
    omega_p = 2 * np.pi * 0.0025 * V_r

    V_rt, V_rtl = 0, 0
    for i in range(N):
        epsilon = np.random.uniform(0, 2 * np.pi)
        S_omega = sigma_u ** 2 / (omega_p * (1 + 1.5 * omega[i] / omega_p) ** (5.0 / 3.0))
        dS_omega = -5 * np.pi * sigma_u ** 2 / omega_p ** 2 * (1 + 1.5 * omega[i] / omega_p) ** (-8.0 / 3.0)
        Sv_omega = 0.5 * (S_omega - omega[i] / (2 * np.pi) * dS_omega)
        V_rt += np.sqrt(2 * S_omega * delta_omega) * np.cos(omega[i] * time + epsilon)
        V_rtl += np.sqrt(2 * Sv_omega * delta_omega) * np.cos(omega[i] * time + epsilon)

    V_r_total = np.sqrt((V_rt + V_r) ** 2 + V_rtl ** 2)
    Psi_r_total = Psi_r + np.arctan2(V_rtl, V_rt + V_r)

    # 经验阻力系数表
    degrees = np.arange(0, 185, 5)

    Cx = np.array([
        -0.348756, -0.39009, -0.431425, -0.472759, -0.475342, -0.477926, -0.480509,
        -0.446925, -0.413341, -0.379757, -0.310006, -0.240254, -0.170503, -0.138211,
        -0.105918, -0.0736262, -0.041334, -0.00904175, 0.0232505, 0.0516676, 0.0800847,
        0.108502, 0.136919, 0.165336, 0.193753, 0.271255, 0.348756, 0.426258, 0.50376,
        0.581261, 0.658763, 0.630345, 0.601928, 0.573511, 0.545093, 0.516676, 0.488258
    ])

    Cy = np.array([
        -0.0, -0.211633, -0.423266, -0.634899, -0.779354, -0.92381, -1.06827, -1.19722,
        -1.32617, -1.45512, -1.47556, -1.496, -1.51644, -1.49944, -1.48243, -1.46543,
        -1.44842, -1.43142, -1.41441, -1.44649, -1.47856, -1.51064, -1.54272, -1.57479,
        -1.60687, -1.50074, -1.3946, -1.28847, -1.18234, -1.07621, -0.970075, -0.809695,
        -0.649315, -0.488935, -0.328555, -0.168175, -0.00779514
    ])

    Cn = np.array([
        0.0, -0.0572305, -0.114461, -0.171691, -0.208618, -0.245545, -0.282471,
        -0.294315, -0.306158, -0.318002, -0.308347, -0.298692, -0.289037, -0.256093,
        -0.223148, -0.190204, -0.157259, -0.124315, -0.0913704, -0.0655932, -0.0398159,
        -0.0140387, 0.0117385, 0.0375158, 0.063293, 0.0687676, 0.0742422, 0.0797168,
        0.0851914, 0.090666, 0.0961406, 0.0819052, 0.0676698, 0.0534344, 0.039199,
        0.0249636, 0.0107281
    ])

    Cx_psi = np.interp(np.degrees(Psi_r_total), degrees, Cx)
    Cy_psi = np.interp(np.degrees(Psi_r_total), degrees, Cy)
    Cn_psi = np.interp(np.degrees(Psi_r_total), degrees, Cn)

    if Psi_r_total < 0:
        Cy_psi *= -1
        Cn_psi *= -1

    A_f = ship.AFw
    A_s = ship.ALw
    L_oa = ship.Loa

    tau_X = 0.5 * rho_a * A_f * V_r_total ** 2 * Cx_psi
    tau_Y = 0.5 * rho_a * A_s * V_r_total ** 2 * Cy_psi
    tau_N = 0.5 * rho_a * A_s * L_oa * V_r_total ** 2 * Cn_psi

    tau_wind = [tau_X, tau_Y, tau_N]
    return tau_wind, Cx_psi, Cy_psi, Cn_psi


# === 统一接口 wind_model ===
def wind_model(method, time, V_wind, Psi_wind, eta, nu, ship: ShipParams):
    if method == 'func1':
        return wind_func1(time, V_wind, Psi_wind, eta, nu, ship)
    elif method == 'func2':
        return wind_func2(time, V_wind, Psi_wind, eta, nu, ship)
    else:
        raise ValueError(f"Unsupported wind model method: {method}")
