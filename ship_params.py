# ship_params.py

class ShipParams:
    def __init__(self):
        # === 船舶主尺度 ===
        self.Loa = 160.93  # 船长 (m)
        self.B = 30.0  # 船宽 (m)
        self.T = 6.0  # 吃水 (m)

        # === 船体投影面积 ===
        self.ALw = 900.0  # 船侧投影面积 (m²)
        self.AFw = 300.0  # 船首投影面积 (m²)
        self.A_SS = 100.0  # 上层建筑侧投影面积 (m²)

        # === 船体几何参数 ===
        self.S = 100.0  # 船体侧投影周长 (m)
        self.C = 50.0  # 船体侧投影重心到船首的距离 (m)
        self.M = 2  # 上层建筑桅杆数量

        # === 船舶运动参数 ===
        self.GMT = 1.0  # 横稳心高度 (m)
        self.Cb = 0.65  # 方形系数
        self.U = 7.7175  # 船速 (m/s)

        # === 物理常数 ===
        self.rho_air = 1.224  # 空气密度 kg/m³
        self.rho_water = 1025  # 水密度 kg/m³
        self.g = 9.81  # 重力加速度 m/s²

    def show(self):
        print("==== Ship Parameters ====")
        print(f"Length overall (Loa): {self.Loa} m")
        print(f"Breadth (B): {self.B} m")
        print(f"Draught (T): {self.T} m")
        print(f"Lateral projected area (ALw): {self.ALw} m²")
        print(f"Frontal projected area (AFw): {self.AFw} m²")
        print(f"Superstructure lateral area (A_SS): {self.A_SS} m²")
        print(f"Perimeter length (S): {self.S} m")
        print(f"Centroid distance (C): {self.C} m")
        print(f"Number of masts (M): {self.M}")
        print(f"Metacentric height (GMT): {self.GMT} m")
        print(f"Block coefficient (Cb): {self.Cb}")
        print(f"Ship speed (U): {self.U} m/s")
        print(f"Air density: {self.rho_air} kg/m³")
        print(f"Water density: {self.rho_water} kg/m³")
        print(f"Gravity: {self.g} m/s²")


# Initialize the global ship parameters
ship_params = ShipParams()
