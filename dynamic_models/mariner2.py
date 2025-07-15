import numpy as np

def mariner2(x, ui, U0=7.7175):
    """
    mariner2(x, ui, U0) returns the speed U in m/s and the time derivative of the state vector:
    x = [ u v r x y psi delta ] for the Mariner class vessel L = 160.93 m, where

    u     = perturbed surge velocity about U0 (m/s)
    v     = perturbed sway velocity about zero (m/s)
    r     = perturbed yaw velocity about zero (rad/s)
    x     = position in x-direction (m)
    y     = position in y-direction (m)
    psi   = perturbed yaw angle about zero (rad)
    delta = actual rudder angle (rad)

    The inputs are:

    ui    = commanded rudder angle (rad)
    U0    = nominal speed. Default value is U0 = 7.7175 m/s (15 knots).

    Reference:
    M.S. Chislett and J. Stroem-Tejsen (1965). Planar Motion Mechanism Tests and Full-Scale Steering and Maneuvering Predictions for a Mariner Class Vessel,
    Technical Report Hy-5, Hydro- and Aerodynamics Laboratory, Lyngby, Denmark.
    """
    if len(x) != 7:
        raise ValueError("x-vector must have dimension 7!")
    if not np.isscalar(ui):
        raise ValueError("ui must be a scalar input!")

    # Normalization variables
    L = 160.93
    U = np.sqrt((U0 + x[0])**2 + x[1]**2)

    # Non-dimensional states and inputs
    delta_c = -ui  # delta_c = -ui such that positive delta_c -> positive r

    u = x[0] / U
    v = x[1] / U
    r = x[2] * L / U
    psi = x[5]
    delta = x[6]

    # Parameters, hydrodynamic derivatives and main dimensions
    delta_max = 35  # max rudder angle (deg)
    Ddelta_max = 5  # max rudder derivative (deg/s)

    m = 798e-5
    Izz = 39.2e-5
    xG = -0.023

    Xudot = -42e-5
    Yvdot = -748e-5
    Nvdot = 4.646e-5
    Xu = -120e-5
    Yrdot = -9.354e-5
    Nrdot = -43.8e-5
    Xuu = 45e-5
    Yv = -1160.4e-5
    Nv = -263.5e-5
    Xuuu = -10.3e-5
    Yr = -499e-5
    Nr = -166e-5
    Xvv = -898.8e-5
    Yvvv = -8078.2e-5
    Nvvv = 1636.1e-5
    Xrr = 18e-5
    Yvvr = 15356e-5
    Nvvr = -5483e-5
    Xdd = -94.8e-5
    Yd = 277.9e-5
    Nd = -138.8e-5
    Xrv = 798e-5
    Yddd = -90e-5
    Nddd = 45e-5
    Xvd = 93.2e-5
    Yvdd = -3.8e-5
    Nvdd = 12.5e-5
    Yvvd = 1189.6e-5
    Nvvd = -489e-5
    Y0 = -3.6e-5
    N0 = 2.8e-5

    # Masses and moments of inertia
    m11 = m - Xudot
    m22 = m - Yvdot
    m23 = m * xG - Yrdot
    m32 = m * xG - Nvdot
    m33 = Izz - Nrdot

    # Rudder saturation and dynamics
    if abs(delta_c) >= delta_max * np.pi / 180:
        delta_c = np.sign(delta_c) * delta_max * np.pi / 180
    delta_dot = delta_c - delta
    if abs(delta_dot) >= Ddelta_max * np.pi / 180:
        delta_dot = np.sign(delta_dot) * Ddelta_max * np.pi / 180

    # Forces and moments
    X = (Xu * u + Xuu * u**2 + Xuuu * u**3 + Xvv * v**2 + Xrr * r**2 + Xrv * r * v +
         Xdd * delta**2 + Xvd * v * delta)
    Y = (Yv * v + Yr * r + Yvvv * v**3 + Yvvr * v**2 * r + Yd * delta + Yddd * delta**3 +
         Yvdd * v * delta**2 + Yvvd * v**2 * delta + Y0)
    N = (Nv * v + Nr * r + Nvvv * v**3 + Nvvr * v**2 * r + Nd * delta + Nddd * delta**3 +
         Nvdd * v * delta**2 + Nvvd * v**2 * delta + N0)

    # Dimensional state derivative
    detM22 = m22 * m33 - m23 * m32

    xdot = np.zeros(7)
    xdot[0] = X * (U**2 / L) / m11
    xdot[1] = -(-m33 * Y + m23 * N) * (U**2 / L) / detM22
    xdot[2] = (-m32 * Y + m22 * N) * (U**2 / L**2) / detM22
    xdot[3] = (np.cos(psi) * (U0 / U + u) - np.sin(psi) * v) * U
    xdot[4] = (np.sin(psi) * (U0 / U + u) + np.cos(psi) * v) * U
    xdot[5] = r * (U / L)
    xdot[6] = delta_dot

    return np.array(xdot)
