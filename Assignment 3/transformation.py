import numpy as np
import spacetools.constants as sc
import spacetools
import matplotlib.pyplot as plt
import scipy.integrate as spi
import math as m

def orbital_period(a: float, mu: float) -> float:
    """ Calculates orbital period of satellite

    Args:
        a (float): Semimajor axis of satellite orbit
        mu (float): gravitational parameter of planet

    Returns:
        float: Orbital period 
    """
    return ((2 * np.pi) / mu**(1 / 2)) * a**(3 / 2)

def angular_mv_calculations(r_a: float, r_p: float, mu: float) -> float:
    """ Calculates angular momentum

    Args:
        r_a (float): radius of apogee
        r_p (float): radius of perigee
        mu (float): gravitational parameter

    Returns:
        float: angula momentum
    """
    numerator = r_a * r_p
    denominator = r_a + r_p
    h = np.sqrt((2 * mu)) * np.sqrt(numerator / denominator)
    return h

def system_dynamics(t: np.ndarray, y: np.ndarray, mu: float) -> np.ndarray:
    """System dynamics for the two-body system.

    Args:
        t (np.ndarray): Time steps.
        y (np.ndarray): Initial state vector.
        mu (float): Gravitational parameter.

    Returns:
        np.ndarray: Derivative vector.
    """
    x, y, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y**2 + z**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3

    return [vx, vy, vz, ax, ay, az]

import numpy as np

def system_dynamics_J2(t: float, y: np.ndarray, mu: float, J2: float, R: float) -> np.ndarray:
    """
    System dynamics for a two-body system including J2 perturbation (directly from Eqs. 6.1–6.3).

    Args:
        t (float): Time (s)
        y (np.ndarray): State vector [x, y, z, vx, vy, vz]
        mu (float): Gravitational parameter (m^3/s^2)
        J2 (float): Second zonal harmonic coefficient
        R (float): Planetary mean radius (m)

    Returns:
        np.ndarray: Derivative vector [vx, vy, vz, ax, ay, az]
    """
    # Unpack state vector
    x, y, z, vx, vy, vz = y

    # Distance
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)

    # Define constant c = (J2 * mu * R^2) / 2
    c = 0.5 * J2 * mu * R**2

    # Accelerations per Eqs. (6.1)–(6.3)
    ax = -mu * x / r**3 + 3 * c * (x / r**5) * (1 - 5 * z**2 / r2)
    ay = -mu * y / r**3 + 3 * c * (y / r**5) * (1 - 5 * z**2 / r2)
    az = -mu * z / r**3 + 3 * c * (z / r**5) * (3 - 5 * z**2 / r2)

    return np.array([vx, vy, vz, ax, ay, az])


def perifocal_coordinates(h: float, mu: float, theta: float, e: float) -> np.ndarray:
    """ Computes original perifocal coordinates

    Args:
        h (float): angular momentum
        mu (float): gravitational parameter
        theta (float): true anomaly
        e (float): ecentricity

    Returns:
        np.ndarray: vectors of position and velocity 
    """
    x_perifocal = (h**2 / mu) * (1 / (1 + e * np.cos(theta))) * np.cos(theta)
    y_perifocal = (h**2 / mu) * (1 / (1 + e * np.cos(theta))) * np.sin(theta)

    vx_perifocal = (mu / h) * (-np.sin(theta))
    vy_perifocal = (mu / h) * (e + np.cos(theta))

    r = np.array([x_perifocal, y_perifocal, np.zeros_like(theta)])
    v = np.array([vx_perifocal, vy_perifocal, np.zeros_like(theta)])

    return r, v

def rot_x(angle: float) -> np.ndarray:
    """ DCM for x axis

    Args:
        angle (float): angle of rotation

    Returns:
        np.ndarray: DCM Matrix
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), np.sin(angle)],
                     [0, -np.sin(angle), np.cos(angle)]])


def rot_y(angle: float) -> np.ndarray:
    """ DCM for y axis

    Args:
        angle (float): angle of rotation

    Returns:
        np.ndarray: DCM Matrix
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rot_z(angle: float) -> np.ndarray:
    """ DCM for x axis

    Args:
        angle (float): angle of rotation

    Returns:
        np.ndarray: DCM Matrix
    """
    return np.array([[np.cos(angle), np.sin(angle), 0],
                     [-np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


def rotation_matrix_ECEF(dt: float, omega_E: float) -> np.ndarray:
    """ Rotation matrix for ECI TO ECEF frame

    Args:
        dt (float): time passed
        omega_E (float): rotational vecloity of Earth

    Returns:
        np.ndarray: rotation matrix
    """
    theta = omega_E * dt
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def euler_angle_sequence(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ Rotation matrix for perifocal to ECI

    Args:
        alpha (float): RAAN
        beta (float): Inclination
        gamma (float): argument of perigee

    Returns:
        np.ndarray: rotation matrix
    """
    R_z_alpha = rot_z(alpha)
    R_x_beta = rot_x(beta)
    R_z_gamma = rot_z(gamma)

    Q = R_z_gamma @ R_x_beta @ R_z_alpha
    return np.transpose(Q)


def perifocal_to_ECI(perifocal: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transforms Perifocal coordinates to ECI coordinates

    Args:
        perifocal (np.ndarray): Perifocal coordinates
        rotation (np.ndarray): rotation matrix

    Returns:
        np.ndarray: ECI cooridnates
    """
    return rotation @ perifocal

def ECI_to_ECEF(r_eci: np.ndarray, t_eval: np.ndarray, omega_E: float) -> np.ndarray:
    """Rotate each ECI state into ECEF using your rotation_matrix_ECEF"""
    N = r_eci.shape[1]
    r_ecef = np.zeros_like(r_eci)
    for j in range(N):
        R = rotation_matrix_ECEF(t_eval[j], omega_E)  # new rotation each time
        r_ecef[:, j] = R @ r_eci[:, j]
    return r_ecef
   

def declination_ascension(r_ECI: np.ndarray) -> float:
    """ Calculates the angle of ascension and declination angle of ECEF coordinates

    Args:
        r_ECEF (np.ndarray): ECI coordinates

    Returns:
        float: float containing ascension and declination angles
    """
    x, y, z = r_ECI
    lon = np.degrees(np.arctan2(y, x))
    lon = ((lon + 180) % 360) - 180  # normalise to [-180,180]
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))  # geocentric latitude
    return np.vstack([lat, lon])

def ECEF_to_LLA(r_ecef: np.ndarray, a: float, tol=1e-6, maxit=20) -> np.ndarray:
    """
    ECEF -> LLA (geodetic) using Bowring/Newton–Raphson.
    Accepts (N,3), (3,N), or (3,) and returns (N,3) [lat_deg, lon_deg, h_m].
    """
    A = np.asarray(r_ecef, float)

    # Normalise orientation to (N,3): rows = samples, cols = [X,Y,Z]
    if A.ndim == 1:
        A = A.reshape(1, 3)
    elif A.shape[1] == 3:
        pass  # already (N,3)
    elif A.shape[0] == 3:
        A = A.T  # was (3,N) -> (N,3)
    else:
        raise ValueError(f"r_ecef must be (N,3), (3,N), or (3,), got {A.shape}")

    X, Y, Z = A[:, 0], A[:, 1], A[:, 2]

    # WGS-84 constants
    f  = 1/298.257223563
    e2 = f*(2 - f)
    p  = np.hypot(X, Y)

    # Longitude, wrapped to [-180, 180]
    lon = np.degrees(np.arctan2(Y, X))
    lon = ((lon + 180.0) % 360.0) - 180.0
    lon = np.where(p == 0.0, 0.0, lon)  # define lon at poles

    # Newton–Raphson on Bowring's kappa
    kappa = 1.0/(1.0 - e2)  # k0 starter (h≈0)
    for _ in range(maxit):
        ci = ((p**2 + (1.0 - e2)*(Z**2)*(kappa**2))**1.5) / (a*e2)
        k_next = 1.0 + (p**2 + (1.0 - e2)*(Z**2)*(kappa**3)) / (ci - p**2)
        if np.all(np.abs(k_next - kappa) < tol):
            kappa = k_next
            break
        kappa = k_next

    # Latitude & height
    phi = np.arctan((Z / p) * kappa)                 # radians
    N   = a / np.sqrt(1.0 - e2 * np.sin(phi)**2)
    h   = p / np.cos(phi) - N

    lat = np.degrees(phi)

    return np.column_stack([lat, lon, h])

def plot_height_decay(sol, R_earth=6378.137):
    """
    Propagates a decaying orbit under drag and plots altitude vs time.

    Args:
        y0 (list): Initial state [x, y, z, vx, vy, vz] in km and km/s
        t_span (tuple): (start_time, end_time) in seconds
        mu (float): Gravitational parameter (km^3/s^2)
        Cd (float): Drag coefficient
        A (float): Cross-sectional area (km^2)
        m (float): Mass (kg)
        R_earth (float): Earth's radius (km)
    """
    # --- Integrate using solve_ivp ---

    # --- Extract results ---
    t = sol.t / 3600.0  # convert time to hours
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    h = r - R_earth  # altitude (km)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(t, h, color='royalblue', linewidth=1.5)
    plt.title("Orbital Altitude Decay under Atmospheric Drag", fontsize=13)
    plt.xlabel("Time (hours)")
    plt.ylabel("Altitude (km)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def Transformations(a: float, e: float, i: float,
                    RAAN: float, theta: float, arg_perigee: float, mode: int) -> None:
    """ Runs the scripts on this page

    Args:
        a (float): semimajor axis
        e (float): eccentricity
        i (float): _description_
        RAAN (float): Right Ascension of Ascending node
        theta (float): True anomaly
        arg_perigee (float): Argument of perigee
        mode (int): mode determining pertubation or not
    """
    ## -------- Conversion
    mu = sc.MU_EARTH / 1e9 ## in km^3/s^2
    r_p = a * (1 - e) # radius of perigee
    r_a = 2*a - r_p # radius of apogee
    R_earth = 6378.137          # km
    J2_earth = 1.08262668e-3


## -------- Orbital Period 
    T = orbital_period(a, 398600) ## Orbital Period
    display_orbital_period(T)

    h = angular_mv_calculations(r_a, r_p, mu) ## Angular momentum of satellite


## ---------- ECI plot
    r_peri, v_peri = perifocal_coordinates(h, mu, theta, e)
    Q = euler_angle_sequence(RAAN, i, arg_perigee)
    r_eci = perifocal_to_ECI(r_peri, Q)
    v_eci = perifocal_to_ECI(v_peri, Q)
    ECI = np.concatenate((r_eci, v_eci))
    
    t = (0, 240 * 3600)
    t_eval = np.linspace(0, 240 * 3600, 100)
    if (mode == 0):
        orbit = spi.solve_ivp(
            system_dynamics, t, ECI, t_eval=t_eval, args=(mu,), max_step=100)

        orbit_eci = np.array([orbit.y[0], orbit.y[1], orbit.y[2]])
        print("Propogation completed")
    else:
        orbit = spi.solve_ivp(
            system_dynamics_J2, t, ECI, t_eval=t_eval, args=(mu, J2_earth, R_earth), max_step=1)

        orbit_eci = np.array([orbit.y[0], orbit.y[1], orbit.y[2]])

    
    plot_orbit_ECI(orbit_eci)  ## Plot for 24 hours


    ## ----- ECI -> ECEF
    r_ecef = ECI_to_ECEF(orbit_eci, t_eval,  7.2921159e-5)
    
    latlon = declination_ascension(r_ecef)    
    
    plot_orbit_ECEF(latlon)

    ## ---- ECEF -> LLA

    LLA = ECEF_to_LLA(r_ecef, a)
    latlon_LLA = np.vstack([
    LLA[:, 0],                                # all latitudes
    ((LLA[:, 1] + 180.0) % 360.0) - 180.0 ])    # all longitudes, wrapped [-180,180]

    plot_orbit_LLA(latlon_LLA)
    

    return 


def plot_orbit_ECI(r_eci: np.ndarray) -> None:
    """ Plot orbit in ECI frame

    Args:
        r_eci (np.ndarray): ECI plot
    """
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    spacetools.plot_orbit_3d(ax_3d, r_eci, color='red', label='Orbital Trajectory', zorder=20)
    ax_3d.set_title('Satellite orbit in ECI frame')
    ax_3d.set_xlabel('X Position (km)')
    ax_3d.set_ylabel('Y Position (km)')
    ax_3d.set_zlabel('Z Position (km)')
    ax_3d.set_aspect('equal')   # Forces all axes to have the same scaling

# Plot earth as a sphere
    earth_radius = sc.R_EARTH * 1e-3
    U, V = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = earth_radius * np.outer(np.cos(U), np.sin(V))
    y = earth_radius * np.outer(np.sin(U), np.sin(V))
    z = earth_radius * np.outer(np.ones(np.size(U)), np.cos(V))
    ax_3d.plot_surface(x, y, z, color='blue', alpha=1, label='Earth', zorder=10)

    ax_3d.view_init(elev=20, azim=50)  # Adjust the view angle for better visualization
    ax_3d.grid(True)
    plt.show()

def plot_orbit_ECEF(r_ecef: np.ndarray) -> None:
    fig_groundtrack = plt.figure()
    ax_groundtrack = fig_groundtrack.add_subplot(111)

    spacetools.groundtrack(
        ax_groundtrack,
        r_ecef,
        arrows=True,
        arrow_interval=3000,
        arrow_kwargs={
            'linewidth': 0.5,
            'arrowstyle': '-|>',
            'mutation_scale': 10,
        },
        color='red',
        label='Ground Track',
    )

    # Blue Mountains (approx): 33.61° S, 150.46° E
    lon_bm = 150.46
    lat_bm = -33.61

    ax_groundtrack.scatter(
        lon_bm, lat_bm,
        s=30, marker='o', color='green',
        label='Blue Mountains', zorder=5
    )

    ax_groundtrack.set_title('Ground Track of satellite over 24 hours - ECEF')
    ax_groundtrack.set_xlabel('Longitude [deg]')
    ax_groundtrack.set_ylabel('Latitude [deg]')
    ax_groundtrack.legend(loc='upper right')
    plt.show()

def plot_orbit_LLA(r_LLA: np.ndarray) -> None:
    fig_groundtrack = plt.figure()
    ax_groundtrack = fig_groundtrack.add_subplot(111)

    spacetools.groundtrack(
        ax_groundtrack,
        r_LLA,
        arrows=True,
        arrow_interval=3000,
        arrow_kwargs={
            'linewidth': 0.5,
            'arrowstyle': '-|>',
            'mutation_scale': 10,
        },
        color='red',
        label='Ground Track',
    )

    # Blue Mountains (approx): 33.61° S, 150.46° E
    lon_bm = 150.46
    lat_bm = -33.61

    ax_groundtrack.scatter(
        lon_bm, lat_bm,
        s=30, marker='o', color='green',
        label='Blue Mountains', zorder=5
    )

    ax_groundtrack.set_title('Ground Track of satellite over 24 hours - LLA')
    ax_groundtrack.set_xlabel('Longitude [deg]')
    ax_groundtrack.set_ylabel('Latitude [deg]')
    ax_groundtrack.legend(loc='upper right')
    plt.show()

def system_dynamics_drag_km(t: float, y: np.ndarray, mu: float,
                            Cd: float, A: float, m: float) -> np.ndarray:
    """
    Two-body orbital dynamics including atmospheric drag.
    Units: km, km/s, s

    Args:
        t (float): Time (s)
        y (np.ndarray): State vector [x, y, z, vx, vy, vz] (km, km/s)
        mu (float): Gravitational parameter (km^3/s^2)
        Cd (float): Drag coefficient (dimensionless)
        A (float): Cross-sectional area (km^2)
        m (float): Spacecraft mass (kg)

    Returns:
        np.ndarray: Derivative [vx, vy, vz, ax, ay, az]
    """
    # Unpack state vector
    x, y, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y**2 + z**2)

    # --- Two-body gravity ---
    ax_grav = -mu * x / r**3
    ay_grav = -mu * y / r**3
    az_grav = -mu * z / r**3

    # --- Simple exponential atmosphere (valid 200–800 km) ---
    R_earth = 6378.137  # km
    h = r - R_earth     # altitude (km)

    rho0 = 3.614e-13    # kg/km^3 at 700 km
    h0 = 700.0          # reference altitude (km)
    H = 88.667          # scale height (km)
    rho = rho0 * np.exp(-(h - h0) / H)

    # --- Drag acceleration ---
    v = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v)

    if v_mag > 0:
        a_drag = -0.5 * Cd * A / m * rho * v_mag * v
    else:
        a_drag = np.zeros(3)

    # --- Total acceleration ---
    ax = ax_grav + a_drag[0]
    ay = ay_grav + a_drag[1]
    az = az_grav + a_drag[2]

    return np.array([vx, vy, vz, ax, ay, az])

def display_orbital_period(T: float) -> None:
    """ Displays orbital period in hours

    Args:
        T (float): Orbital Period
    """

    print(f"Orbital period of satellite is {T / 3600:.2f} hours")
    
