import numpy as np

def current(x, y, U0):
    """
    动态计算洋流速度和方向。
    """
    V_c = 1.5 / U0  # 动态速度，例如随位置变化
    V_angle = np.radians(155)  # 动态方向角
    return V_c, V_angle


def decompose_current(beta_c, V_c, psi, U0):
    x = np.cos(beta_c) * V_c * U0
    y = np.sin(beta_c) * V_c * U0
    u_c = (np.cos(psi) * x - np.sin(psi) * y) / U0
    v_c = (np.sin(psi) * x + np.cos(psi) * y) / U0
    return u_c, v_c