import numpy as np
from ship_params import ShipParams


# === wave_func1：基于waveresponse ===
def wave_func1(t, a, beta, T_0, zeta4, T4, GMT, Cb, U, L, B, T, **kwargs):
    g = 9.81  # 重力加速度
    rho_water = kwargs.get('rho_water', 1025)

    nabla = Cb * L * B * T
    w_0 = 2 * np.pi / T_0
    k = w_0 ** 2 / g
    w_e = w_0 - k * U * np.cos(beta)
    k_e = abs(k * np.cos(beta))
    sigma = k_e * L / 2
    kappa = np.exp(-k_e * T)

    alpha = w_e / w_0
    A = 2 * np.sin(k * B * alpha ** 2 / 2) * np.exp(-k * T * alpha ** 2)
    f = np.sqrt((1 - k * T) ** 2 + (A ** 2 / (k * B * alpha ** 3)) ** 2)
    F = kappa * f * np.sin(sigma) / sigma
    G = kappa * f * (6 / L) * (1 / sigma) * (np.sin(sigma) / sigma - np.cos(sigma))

    wn = np.sqrt(g / (2 * T))
    zeta = (A ** 2 / (B * alpha ** 3)) * np.sqrt(1 / (8 * k ** 3 * T))

    Z3 = np.sqrt((2 * wn * zeta) ** 2 + (1 / w_e ** 2) * (wn ** 2 - w_e ** 2) ** 2)
    eps3 = np.arctan2(2 * w_e * wn * zeta, wn ** 2 - w_e ** 2)
    z_heave = (a * F * wn ** 2 / (Z3 * w_e)) * np.cos(w_e * t + eps3)

    Z5 = Z3
    eps5 = eps3
    theta_pitch = (a * G * wn ** 2 / (Z5 * w_e)) * np.sin(w_e * t + eps5)
    theta_pitch_deg = np.degrees(theta_pitch)

    w4 = 2 * np.pi / T4
    C44 = rho_water * g * nabla * GMT
    M44 = C44 / w4 ** 2
    B44 = 2 * zeta4 * w4 * M44
    M = np.sin(beta) * np.sqrt(B44 * rho_water * g ** 2 / max(w_e, 1e-6))

    Z4 = np.sqrt((2 * w4 * zeta4) ** 2 + (1 / w_e ** 2) * (w4 ** 2 - w_e ** 2) ** 2)
    eps4 = np.arctan2(2 * w_e * w4 * zeta4, w4 ** 2 - w_e ** 2)
    phi_roll = ((M / C44) * w4 ** 2 / (Z4 * w_e)) * np.cos(w_e * t + eps4)
    phi_roll_deg = np.degrees(phi_roll)

    return z_heave, phi_roll_deg, theta_pitch_deg, None  # 保持接口一致，多返回一个 None 占位


# === wave_func2：高频波浪响应，基于Xie的Wave_Func 中的高频波浪力

def fwave_func(Wlambda, omega_e, sigma, z_state, eta_wave_state, gau):
    dz_dt = -sigma * z_state + omega_e ** 2 * eta_wave_state + gau
    return dz_dt


def wave_func2(t, dt, state, V_wind, z_0, eta, nu, Psi_wind, Wlambda, sigma_wave, gau_noise, ship: ShipParams):
    eta_wave_state, z_state = state

    # 高频波参数
    g = ship.g
    omega_0 = 2 * np.pi / 7  # 峰值周期默认 7s（也可以根据 Wave_Func 计算）

    beta = np.arctan2(nu[1], nu[0]) if np.linalg.norm(nu[:2]) > 1e-6 else 0
    omega_e = abs(omega_0 - omega_0 ** 2 / g * np.sqrt(nu[0] ** 2 + nu[1] ** 2) * np.cos(Psi_wind - eta[2] - beta))

    # Runge-Kutta 4 integration
    k1 = fwave_func(Wlambda, omega_e, sigma_wave, z_state, eta_wave_state, gau_noise)
    k2 = fwave_func(Wlambda, omega_e, sigma_wave, z_state + 0.5 * dt * k1,
                   eta_wave_state + 0.5 * dt * z_state, gau_noise)
    k3 = fwave_func(Wlambda, omega_e, sigma_wave, z_state + 0.5 * dt * k2,
                   eta_wave_state + 0.5 * dt * z_state + 0.25 * dt ** 2 * k1, gau_noise)
    k4 = fwave_func(Wlambda, omega_e, sigma_wave, z_state + dt * k3,
                   eta_wave_state + dt * z_state + 0.5 * dt ** 2 * k2, gau_noise)

    new_eta_wave = eta_wave_state + dt * z_state + (dt ** 2 / 6.0) * (k1 + k2 + k3)
    new_z_state = z_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 当前步的输出就是 eta_wave_state
    z_heave = new_eta_wave[0]
    phi_roll_deg = np.degrees(new_eta_wave[1])
    theta_pitch_deg = np.degrees(new_eta_wave[2])

    return z_heave, phi_roll_deg, theta_pitch_deg, (new_eta_wave, new_z_state)


# === 统一接口 ===
def wave_model(method, *args, **kwargs):
    if method == 'func1':
        return wave_func1(*args, **kwargs)
    elif method == 'func2':
        return wave_func2(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported wave model method: {method}")
