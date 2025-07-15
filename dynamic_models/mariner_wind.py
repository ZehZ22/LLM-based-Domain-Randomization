import numpy as np
from disturbances.isherwood72 import isherwood72
from ship_params import ship_params

def marinerwind(x, ui, U0=7.7175, wind_speed=0.0, wind_direction=0.0, scale_ratio=1.0):
    if len(x) != 7:
        raise ValueError('x-vector must have dimension 7!')
    if not np.isscalar(ui):
        raise ValueError('ui must be a scalar input!')

    # 船舶基本参数
    L = 160.93  # 原始船长度
    U = np.sqrt((U0 + x[0]) ** 2 + x[1] ** 2)  # 总速度

    # 无量纲化后的状态变量
    u = x[0] / U
    v = x[1] / U
    r = x[2] * L / U
    psi = x[5]
    delta = x[6]

    # 舵角限制
    delta_max = 35 * np.pi / 180  # 最大舵角（弧度）
    Ddelta_max = 5 * np.pi / 180  # 最大舵角变化率（弧度/s）

    delta_c = -ui
    delta_c = np.clip(delta_c, -delta_max, delta_max)
    delta_dot = delta_c - delta
    delta_dot = np.clip(delta_dot, -Ddelta_max, Ddelta_max)

    # 船舶质量、惯性矩等参数
    m = 798e-5
    Iz = 39.2e-5
    xG = -0.023

    Xudot = -42e-5
    Yvdot = -748e-5
    Nvdot = 4.646e-5
    Xu = -184e-5
    Yrdot = -9.354e-5
    Nrdot = -43.8e-5
    Xuu = -110e-5
    Yv = -1160e-5
    Nv = -264e-5
    Xuuu = -215e-5
    Yr = -499e-5
    Nr = -166e-5
    Xvv = -899e-5
    Yvvv = -8078e-5
    Nvvv = 1636e-5
    Xrr = 18e-5
    Yvvr = 15356e-5
    Nvvr = -5483e-5
    Xdd = -95e-5
    Yvu = -1160e-5
    Nvu = -264e-5
    Xudd = -190e-5
    Yru = -499e-5
    Nru = -166e-5
    Xrv = 798e-5
    Yd = 278e-5
    Nd = -139e-5
    Xvd = 93e-5
    Yddd = -90e-5
    Nddd = 45e-5
    Xuvd = 93e-5
    Yud = 556e-5
    Nud = -278e-5
    Yuud = 278e-5
    Nuud = -139e-5
    Yvdd = -4e-5
    Nvdd = 13e-5
    Yvvd = 1190e-5
    Nvvd = -489e-5
    Y0 = -4e-5
    N0 = 3e-5
    Y0u = -8e-5
    N0u = 6e-5
    Y0uu = -4e-5
    N0uu = 3e-5

    # 质量矩阵和惯性矩
    m11 = m - Xudot
    m22 = m - Yvdot
    m23 = m * xG - Yrdot
    m32 = m * xG - Nvdot
    m33 = Iz - Nrdot

    # 计算风力矩
    V_ship_x = U * np.cos(psi)  # 船速在 x 方向的分量
    V_ship_y = U * np.sin(psi)  # 船速在 y 方向的分量
    V_wind_x = wind_speed * np.cos(wind_direction)
    V_wind_y = wind_speed * np.sin(wind_direction)

    V_r_x = V_wind_x - V_ship_x
    V_r_y = V_wind_y - V_ship_y

    V_r = np.sqrt(V_r_x ** 2 + V_r_y ** 2)
    gamma_r = np.arctan2(V_r_y, V_r_x) - psi

    # 船体参数 (风力相关)
    Loa = ship_params.Loa
    B = ship_params.B
    ALw = ship_params.ALw
    AFw = ship_params.AFw
    A_SS = ship_params.A_SS
    S = ship_params.S
    C = ship_params.C
    M = ship_params.M

    tau_w, _, _, _ = isherwood72(gamma_r, V_r, Loa, B, ALw, AFw, A_SS, S, C, M)

    # 适当缩放风力矩并无量纲化处理
    tau_w[0] /= (0.5 * 1000 * U ** 2 * Loa ** 2)
    tau_w[1] /= (0.5 * 1000 * U ** 2 * Loa ** 2)
    tau_w[2] /= (0.5 * 1000 * U ** 2 * Loa ** 3)

    # 计算系统的力和力矩，叠加风力矩
    X = Xu * u + Xuu * u ** 2 + Xuuu * u ** 3 + Xvv * v ** 2 + Xrr * r ** 2 + Xrv * r * v + Xdd * delta ** 2 + \
        Xudd * u * delta ** 2 + Xvd * v * delta + Xuvd * u * v * delta + tau_w[0]

    Y = Yv * v + Yr * r + Yvvv * v ** 3 + Yvvr * v ** 2 * r + Yvu * v * u + Yru * r * u + Yd * delta + \
        Yddd * delta ** 3 + Yud * u * delta + Yuud * u ** 2 * delta + Yvdd * v * delta ** 2 + \
        Yvvd * v ** 2 * delta + (Y0 + Y0u * u + Y0uu * u ** 2) + tau_w[1]

    N = Nv * v + Nr * r + Nvvv * v ** 3 + Nvvr * v ** 2 * r + Nvu * v * u + Nru * r * u + Nd * delta + \
        Nddd * delta ** 3 + Nud * u * delta + Nuud * u ** 2 * delta + Nvdd * v * delta ** 2 + \
        Nvvd * v ** 2 * delta + (N0 + N0u * u + N0uu * u ** 2) + tau_w[2]

    # 计算状态导数
    detM22 = m22 * m33 - m23 * m32

    xdot = np.array([
        X * (U ** 2 / L) / m11,
        -(-m33 * Y + m23 * N) * (U ** 2 / L) / detM22,
        (-m32 * Y + m22 * N) * (U ** 2 / L ** 2) / detM22,
        (np.cos(psi) * (U0 / U + u) - np.sin(psi) * v) * U,
        (np.sin(psi) * (U0 / U + u) + np.cos(psi) * v) * U,
        r * (U / L),
        delta_dot
    ])

    return xdot, tau_w
