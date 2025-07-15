# env_params.py

class EnvParams:
    """
    A class that holds all the parameters related to the ship's geometry, wind, and environment.
    These parameters are used in functions like marinerwind and isherwood72.
    """

    def __init__(self):
        # Ship geometry parameters
        self.Loa = 160.93 # Length overall (m)
        self.B = 30  # Beam (width) (m)
        self.ALw = 900  # Lateral projected area (m^2)
        self.AFw = 300  # Transverse projected area (m^2)
        self.A_SS = 100  # Lateral projected area of superstructure (m^2)
        self.S = 100  # Length of perimeter of lateral projection (m)
        self.C = 50  # Distance from bow of centroid of lateral projected area (m)
        self.M = 2  # Number of masts or king posts (distinct groups seen in lateral projection)

        # Default wind parameters
        self.wind_speed = 10  # Default wind speed (m/s)
        self.wind_direction = 0  # Default wind direction (degrees)

        # Environment parameters
        self.rho_air = 1.224  # Air density (kg/m^3)

    def update_wind_conditions(self, wind_speed, wind_direction):
        """
        Update wind speed and direction during the simulation.
        :param wind_speed: New wind speed (m/s)
        :param wind_direction: New wind direction (degrees)
        """
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction


# Initialize the global ship parameters
env_params = EnvParams()
