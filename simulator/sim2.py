import numpy as np
from disturbances.isherwood72 import isherwood72
from ship_params import ship_params


def marinerwind(x, ui, U0=7.7175, wind_speed=10.0, wind_direction=330.0):
    """
    x: 状态向量 [u, v, r, x, y, psi, delta]
    ui: 命令舵角 (rad)
    U0: 标称速度 (m/s)
    wind_speed: 风速 (m/s)
    wind_direction: 风向 (rad)
    """

    L = 160.93  # 船舶长度
    U = np.sqrt((U0 + x[0]) ** 2 + x[1] ** 2)  # 总速度

    # 从 x 中提取速度、位移和舵角
    u = x[0] / U
    v = x[1] / U
    r = x[2] * L / U
    psi = x[5]
    delta = x[6]

    # 最大舵角和舵角变化率
    delta_max = 35 * np.pi / 180  # 最大舵角
    Ddelta_max = 5 * np.pi / 180  # 最大舵角变化率

    # 舵角限制
    delta_c = -ui  # 命令舵角
    delta_c = np.clip(delta_c, -delta_max, delta_max)
    delta_dot = delta_c - delta
    delta_dot = np.clip(delta_dot, -Ddelta_max, Ddelta_max)

    m = 798e-5
    Izz = 39.2e-5
    xG = -0.023
    Xudot = -42e-5
    Yvdot = -748e-5
    Yrdot = -9.354e-5
    Nvdot = 4.646e-5
    Nrdot = -43.8e-5


    # para_x 的参数
    Xu = -0.00119926039448010
    Xuu = 0.000455555819115393
    Xuuu = -9.46461544509614e-05
    Xvv = -0.00907037041784122
    Xrr = 0.000167141252312789
    Xdd = -0.000947707422346588
    Xudd = -2.54612043552858e-06
    Xvd = 0.00791904293452310
    Xuvd = 0.000934311710584476
    Xrv = 3.19820259617234e-06

    # para_y 的参数
    Yv = -0.0112404832826235
    Yr = -0.00476564418672964
    Yvvv = -0.0935434664025616
    Yvvr = 0.148974866722004
    Yvu = -0.00723967698495130
    Yru = -0.00232867042877100
    Yd = 0.00286138730867823
    Yddd = -0.000786060411494002
    Yud = 0.00716681597865635
    Yuud = 0.00632594383011116
    Yvdd = -0.000316847597309361
    Yvvd = 0.0132808934334867

    # para_n 的参数
    Nv = -0.00288680107928400
    Nr = -0.00181671827746091
    Nvvv = 0.0262811068093262
    Nvvr = -0.0510116342562566
    Nvu = -0.00571477412815948
    Nru = -0.00353481558611082
    Nd = -0.00144854628809892
    Nddd = 0.000369831116460640
    Nud = -0.00393539536791339
    Nuud = -0.00399333602812417
    Nvdd = 0.000314523611317317
    Nvvd = -0.00569141098105009
    # 质量矩阵和惯性矩
    m11 = m - Xudot
    m22 = m - Yvdot
    m23 = m * xG - Yrdot
    m32 = m * xG - Nvdot
    m33 = Izz - Nrdot

    # 风力影响计算
    V_ship_x = U * np.cos(psi)
    V_ship_y = U * np.sin(psi)
    V_wind_x = wind_speed * np.cos(wind_direction)
    V_wind_y = wind_speed * np.sin(wind_direction)

    V_r_x = V_wind_x - V_ship_x
    V_r_y = V_wind_y - V_ship_y

    V_r = np.sqrt(V_r_x ** 2 + V_r_y ** 2)
    gamma_r = np.arctan2(V_r_y, V_r_x) - psi

    # 获取船体几何参数 (用于计算风力矩)
    Loa = ship_params.Loa
    B = ship_params.B
    ALw = ship_params.ALw
    AFw = ship_params.AFw
    A_SS = ship_params.A_SS
    S = ship_params.S
    C = ship_params.C
    M = ship_params.M

    # 通过 isherwood72 模型计算风力矩
    tau_w, _, _, _ = isherwood72(gamma_r, V_r, Loa, B, ALw, AFw, A_SS, S, C, M)

    # Scale the wind force and make non-dimensional
    tau_w[0] /= (0.5 * 1000 * U ** 2 * Loa ** 2)
    tau_w[1] /= (0.5 * 1000 * U ** 2 * Loa ** 2)
    tau_w[2] /= (0.5 * 1000 * U ** 2 * Loa ** 3)

    # 叠加风力矩到原始的力矩方程中
    X = (Xu * u + Xuu * u ** 2 + Xuuu * u ** 3 + Xvv * v ** 2 + Xrr * r ** 2 + Xrv * r * v +
         Xdd * delta ** 2 + Xudd * u * delta ** 2 + Xvd * v * delta + Xuvd * u * v * delta) + tau_w[0]

    Y = (Yv * v + Yr * r + Yvvv * v ** 3 + Yvvr * v ** 2 * r + Yvu * v * u + Yru * r * u +
         Yd * delta + Yddd * delta ** 3 + Yud * u * delta + Yuud * u ** 2 * delta +
         Yvdd * v * delta ** 2 + Yvvd * v ** 2 * delta) + tau_w[1]

    N = (Nv * v + Nr * r + Nvvv * v ** 3 + Nvvr * v ** 2 * r + Nvu * v * u + Nru * r * u +
         Nd * delta + Nddd * delta ** 3 + Nud * u * delta + Nuud * u ** 2 * delta +
         Nvdd * v * delta ** 2 + Nvvd * v ** 2 * delta) + tau_w[2]

    # 计算状态导数
    psi = x[5]
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
