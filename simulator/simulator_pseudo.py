import random
import numpy as np
import torch
from gym import spaces
from disturbances.isherwood72 import isherwood72 
from dynamic_models.mariner_wind import marinerwind
from env_params import env_params


def set_seed(seeds):
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seeds)
        torch.cuda.manual_seed_all(seeds)


# 设置随机种子
seed = 20
set_seed(seed)

# 创建独立的随机生成器，不受全局随机种子的影响
no_seed_random = random.Random()


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


class USVState:
    def __init__(self, waypoints, current_index, x, ui, model_func=marinerwind, wind_speed=3.0, wind_direction=60.0,
                 current_speed=3.0, current_direction=100.0,  # 洋流参数
                 wave_height=3.0, wave_period=3.0, beta=70 * np.pi / 180, T4=3.0, GMT=1.0, Cb=0.65, U=7.7175, L=160.93,
                 B=30.0, T=6, a=2.0, zeta4=0.2,
                 wind_mode="fixed", randomize_params=False, k1=1.0, k2=1.0, k3=1.0, w_chi=0.4, w_ey=0.5,
                 w_sigma_delta=0.0, U0=7.7175, mass=798e-5):

        """
        初始化 USVState，支持风速、洋流速度和船舶参数的域随机化。
        :param current_speed: 初始洋流速度
        :param current_direction: 初始洋流方向（角度）
        :param U0: 船舶的静态速度，默认为 7.7175
        :param wind_speed: 风速，默认为 10.0
        :param wind_direction: 风向，默认为 10.0
        :param wave_height: 波浪高度，默认为 2.0
        :param wave_period: 波浪周期，默认为 3.0
        """
        # 初始化洋流和波浪参数
        self.current_speed = current_speed
        self.current_direction = current_direction
        self.wave_height = wave_height
        self.wave_period = wave_period
        self.V_c = current_speed / U0  # 无量纲化洋流速度
        self.V_angle = np.radians(current_direction)  # 转换为弧度制
        self.V_c, self.V_angle = current(x[3], x[4], U0)
        # 波浪相关参数
        self.beta = beta
        self.T4 = T4
        self.GMT = GMT
        self.Cb = Cb
        self.U = U
        self.L = L
        self.B = B
        self.T = T
        self.a = a
        self.zeta4 = zeta4

        self.waypoints = waypoints
        self.current_index = current_index
        self.x = x
        self.ui = ui
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.w_chi = w_chi
        self.w_ey = w_ey
        self.w_sigma_delta = w_sigma_delta
        self.delta_history = []
        self.model_func = model_func

        # 风力参数
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wind_mode = wind_mode

        self.randomize_params = randomize_params

        # 随机化参数
        if randomize_params:
            self.wind_speed = self.randomize_value(1.0, 5.0, no_seed_random)  # 风速范围 [5,7]knot
            self.wind_direction = self.randomize_value(0.0, 90.0, no_seed_random)
            self.current_speed = self.randomize_value(0.5, 2.0, no_seed_random)  # 洋流速度范围 [2,3] knot
            self.current_direction = self.randomize_value(0.0, 90.0, no_seed_random)
            self.wave_height = self.randomize_value(2.0, 5.0, no_seed_random)  # 波高范围 [0.5,3] m
            self.beta = np.radians(self.randomize_value(0.0, 90.0, no_seed_random))  # 转换为弧度

        self.U0 = U0  # 静态速度
        self.prev_waypoint = waypoints[current_index - 1]
        self.current_waypoint = waypoints[current_index]

        self.path_angle, self.waypoint_angle, self.distance_to_waypoint = self.calculate_path_parameters()
        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()

        self.action_space = spaces.Box(
            low=np.array([-35.0], dtype=np.float32),
            high=np.array([35.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -100.0, -35.0], dtype=np.float32),
            high=np.array([np.pi, 100.0, 35.0], dtype=np.float32),
            dtype=np.float32
        )
        # 这里定义了一个单独的时间变量
        self.time = 0.0  # 初始化时间为0

    def randomize_value(self, min_value, max_value, rng):
        """
        在指定的最小值和最大值之间生成随机数。
        :param min_value: 随机化的最小值
        :param max_value: 随机化的最大值
        :param rng: 独立的随机生成器
        :return: 随机化后的值
        """
        return rng.uniform(min_value, max_value)

    def generate_wave_effect(self, t, wave_height, wave_period):
        # 确保 beta 以弧度表示
        beta = np.radians(self.beta)

        Hs = wave_height
        Tp = wave_period
        T4 = self.T4
        GMT = self.GMT
        Cb = self.Cb
        U = self.U
        L = self.L
        B = self.B
        T = self.T
        a = wave_height / 2
        zeta4 = self.zeta4

        # Constants
        g = 9.81  # 重力加速度 (m/s^2)
        rho = 1025  # 水的密度 (kg/m^3)

        # 船舶参数
        nabla = Cb * L * B * T  # 排水体积 (m^3)
        w_0 = 2 * np.pi / Tp  # 波浪峰值频率 (rad/s)
        k = w_0 ** 2 / g  # 波数
        w_e = max(abs(w_0 - k * U * np.cos(beta)), 1e-3)  # 避免 w_e 过小
        k_e = np.abs(k * np.cos(beta))  # 有效波数
        sigma = k_e * L / 2
        kappa = np.exp(-k_e * T)

        # 纵摇和横摇模型 (Jensen et al., 2004)
        alpha = w_e / w_0
        A = 2 * np.sin(k * B * alpha ** 2 / 2) * np.exp(-k * T * alpha ** 2)
        f = np.sqrt((1 - k * T) ** 2 + (A ** 2 / (k * B * alpha ** 3)) ** 2)
        F = kappa * f * np.sin(sigma) / sigma
        G = kappa * f * (6 / L) * (1 / sigma) * (np.sin(sigma) / sigma - np.cos(sigma))

        # 自然频率 (Jensen et al., 2004)
        wn = np.sqrt(g / (2 * T))
        zeta = (A ** 2 / (B * alpha ** 3)) * np.sqrt(1 / (8 * k ** 3 * T))

        # 横滚模型 (简化版)
        w4 = 2 * np.pi / T4  # 自然频率
        C44 = rho * g * nabla * GMT  # 弹簧系数
        M44 = C44 / w4 ** 2  # 含附质量的转动惯量
        B44 = 2 * zeta4 * w4 * M44  # 阻尼系数
        M = np.sin(beta) * np.sqrt(B44 * rho * g ** 2 / w_e)  # 横滚力矩幅度

        # 纵摇
        Z3 = np.sqrt(max((2 * wn * zeta) ** 2 + max(wn ** 2 - w_e ** 2, 0) ** 2 / max(w_e ** 2, 1e-6), 0))
        eps3 = np.arctan(2 * w_e * wn * zeta / max(wn ** 2 - w_e ** 2, 1e-6))
        Z3_wave = ((a * F * wn ** 2 / (Z3 * w_e)) * np.cos(w_e * t + eps3))

        # 横摇
        Z5 = np.sqrt(max((2 * wn * zeta) ** 2 + max(wn ** 2 - w_e ** 2, 0) ** 2 / max(w_e ** 2, 1e-6), 0))
        eps5 = np.arctan(2 * w_e * wn * zeta / max(wn ** 2 - w_e ** 2, 1e-6))
        Z5_wave = ((a * G * wn ** 2 / (Z5 * w_e)) * np.sin(w_e * t + eps5))

        # 横滚
        Z4 = np.sqrt(max((2 * w4 * zeta4) ** 2 + max(w4 ** 2 - w_e ** 2, 0) ** 2 / max(w_e ** 2, 1e-6), 0))
        eps4 = np.arctan(2 * w_e * w4 * zeta4 / max(w4 ** 2 - w_e ** 2, 1e-6))
        Z4_wave = (180 / np.pi) * ((M / C44) * w4 ** 2 / (Z4 * w_e)) * np.cos(w_e * t + eps4)

        return Z3_wave / 160.93, Z4_wave / 160.93, Z5_wave / 160.93


    def get_full_state(self):
        """
        获取完整的船舶状态，包括航向误差、横向误差、舵角等信息。
        :return: 当前船舶状态（包含航向误差、横向误差、舵角等）
        """
        speed = np.sqrt((self.U0 + self.x[0]) ** 2 + self.x[1] ** 2)  # 包含静态速度和扰动速度
        return np.array(
            [self.heading_error, self.cross_track_error, self.x[6], self.x[3], self.x[4], self.x[5], speed, self.x[0],
             self.x[1]])

    def calculate_heading_error(self):
        self.path_angle = np.arctan2(self.current_waypoint[1] - self.prev_waypoint[1],
                                     self.current_waypoint[0] - self.prev_waypoint[0])
        heading_error = self.path_angle - self.x[5]
        # 将误差归一化到 [-π, π]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        return heading_error

    def calculate_cross_track_error(self):
        d_wk_minus_1 = np.sqrt((self.x[4] - self.prev_waypoint[1]) ** 2 + (self.x[3] - self.prev_waypoint[0]) ** 2)
        chi_wk_minus_1 = np.arctan2(self.x[4] - self.prev_waypoint[1], self.x[3] - self.prev_waypoint[0])

        cross_track_error = np.sin(self.path_angle - chi_wk_minus_1) * d_wk_minus_1
        return cross_track_error

    def calculate_path_parameters(self):
        path_angle = np.arctan2(self.current_waypoint[1] - self.prev_waypoint[1],
                                self.current_waypoint[0] - self.prev_waypoint[0])
        waypoint_angle = np.arctan2(self.x[4] - self.prev_waypoint[1],
                                    self.x[3] - self.prev_waypoint[0])
        distance_to_waypoint = np.sqrt(
            (self.x[4] - self.prev_waypoint[1]) ** 2 + (self.x[3] - self.prev_waypoint[0]) ** 2)
        return path_angle, waypoint_angle, distance_to_waypoint

    def step(self, ui):
        dt = 0.1
        self.ui = ui[0]

        # 更新洋流参数
        self.V_c = self.current_speed / self.U0
        self.V_angle = np.radians(self.current_direction)

        # 计算洋流干扰
        u_c, v_c = decompose_current(self.V_angle, self.V_c, self.x[5], self.U0)
        scaling_factor = 0.2
        self.x[0] -= u_c * scaling_factor * dt
        self.x[1] -= v_c * scaling_factor * dt

        if self.model_func.__name__ == 'marinerwind':
            xdot, wind_force = self.model_func(self.x, self.ui, wind_speed=self.wind_speed,
                                               wind_direction=self.wind_direction)
        else:
            xdot = self.model_func(self.x, self.ui)

        self.x += xdot * dt

        # 波浪干扰（假设波浪周期 Tp 和高度 Hs 已知）
        self.time += dt  # 时间递增
        Z3_wave, Z4_wave, Z5_wave = self.generate_wave_effect(self.time, self.wave_height, self.wave_period)

        # 将波浪扰动加到船舶状态中
        self.x[0] += Z3_wave * dt  # 升沉影响
        self.x[1] += Z4_wave * dt  # 横摇影响
        self.x[2] += Z5_wave * dt  # 纵摇影响

        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()

        self.delta_history.append(self.x[6])
        if len(self.delta_history) > 20:
            self.delta_history.pop(0)

        heading_error_scalar = self.heading_error
        cross_track_error_scalar = self.cross_track_error
        rudder_angle = self.x[6]

        self.state = np.array([heading_error_scalar, cross_track_error_scalar, rudder_angle], dtype=np.float32)
        reward = self.calculate_reward()
        done = self.check_done()

        if not done and self.distance_to_waypoint < 10:
            self.current_index += 1
            self.prev_waypoint = self.current_waypoint
            self.current_waypoint = self.waypoints[self.current_index]
            self.path_angle, self.waypoint_angle, self.distance_to_waypoint = self.calculate_path_parameters()

        next_state = np.array([heading_error_scalar, cross_track_error_scalar, rudder_angle])
        if np.isnan(xdot).any() or np.isinf(xdot).any() or (np.abs(xdot) > 1e6).any():
            print("⚠️ xdot contains NaN, Inf, or extreme values!", xdot)
            xdot = np.clip(xdot, -1e6, 1e6)  # 限制 `xdot` 绝对值最大为 1e6
        return next_state, reward, done, {}

    def calculate_wind_force(self):
        # 计算相对风角（gamma_r）和相对风速（V_r）
        gamma_r = np.radians(self.wind_direction - np.degrees(self.x[5]))  # 相对风角
        V_r = self.wind_speed  # 使用已定义的风速

        # 获取船舶参数
        Loa = env_params.Loa
        B = env_params.B
        ALw = env_params.ALw
        AFw = env_params.AFw
        A_SS = env_params.A_SS
        S = env_params.S
        C = env_params.C
        M = env_params.M

        # 调用 isherwood72 计算风力
        tauW, CX, CY, CN = isherwood72(gamma_r, V_r, Loa, B, ALw, AFw, A_SS, S, C, M)

        # 返回风力和力矩
        wind_force = np.zeros(7)
        wind_force[0] = tauW[0]  # 风力作用在 x 方向
        wind_force[1] = tauW[1]  # 风力作用在 y 方向
        wind_force[2] = tauW[2]  # 风力对偏航的影响

        return wind_force

    def calculate_reward(self):
        r_chi = -self.k1 * np.abs(self.heading_error)
        r_ey = -self.k2 * np.abs(self.cross_track_error)

        if len(self.delta_history) > 1:
            sigma_delta = np.std(self.delta_history)
        else:
            sigma_delta = 0
        r_sigma_delta = -self.k3 * sigma_delta

        reward = self.w_chi * r_chi + self.w_ey * r_ey + self.w_sigma_delta * r_sigma_delta
        return reward

    def check_done(self):
        if self.current_index >= len(self.waypoints) - 1:
            return True
        return False

    def reset(self):
        self.current_index = 1
        # # 随机化艏向角，范围为20到30度
        # initial_heading_angle = np.radians(self.randomize_value(0.0, 0.0, no_seed_random))  # 艏向角随机化
        # # 随机化初始位置 (x, y)，以控制初始距离
        # initial_x = self.randomize_value(300.0, 0.0, no_seed_random)  # 随机化初始 x 坐标
        # initial_y = self.randomize_value(0.0, 0.0, no_seed_random)  # 随机化初始 y 坐标
        # 初始化状态，艏向角使用随机化的角度
        self.x = np.array([0.0, 0.0, 0.0, 300.0, 0.0, np.radians(20), 0.0])
        self.ui = 0.0
        self.delta_history = []

        # 随机化其他参数，每次 reset 时都会触发
        if self.randomize_params:
            self.wind_speed = self.randomize_value(1.0, 5.0, no_seed_random)  # 风速范围 [5,7]knot
            self.wind_direction = self.randomize_value(0.0, 90.0, no_seed_random)
            self.current_speed = self.randomize_value(0.5, 2.0, no_seed_random)  # 洋流速度范围 [2,3] knot
            self.current_direction = self.randomize_value(0.0, 90.0, no_seed_random)
            self.wave_height = self.randomize_value(2.0, 5.0, no_seed_random)  # 波高范围 [0.5,3] m
            self.beta = np.radians(self.randomize_value(0.0, 90.0, no_seed_random) )  # 转换为弧度

        self.prev_waypoint = self.waypoints[self.current_index - 1]
        self.current_waypoint = self.waypoints[self.current_index]

        self.path_angle, self.waypoint_angle, self.distance_to_waypoint = self.calculate_path_parameters()

        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()
        rudder_angle = self.x[6]

        self.state = np.array([self.heading_error, self.cross_track_error, rudder_angle], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        heading_error, cross_track_error, rudder_angle = self.state
        return np.array([heading_error, cross_track_error, rudder_angle], dtype=np.float32)


def generate_path(angle_deg, length, interval):
    angle_rad = np.radians(angle_deg)
    num_points = int(length // interval) + 1

    waypoints = []
    for i in range(num_points):
        x = i * interval * np.cos(angle_rad)
        y = i * interval * np.sin(angle_rad)
        waypoints.append((x, y))

    waypoints.append((length * np.cos(angle_rad), length * np.sin(angle_rad)))

    return waypoints
