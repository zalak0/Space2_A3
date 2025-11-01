## NOTE: file extractats TLE components and plot orbit around the Earth

# ==================== gsat_multi_orbit_spacetools.py ====================
"""
Propagate multiple GSAT satellites, align all to a common epoch,
and plot ECI orbits, sampled ground points, and full ground tracks.

This module:
- Loads TLEs and extracts elements + epoch.
- Computes state at a specified (common) epoch.
- Propagates orbits in ECI using a simple two-body model.
- Converts ECI to ECEF and to latitude/longitude for ground tracks.
- Plots orbits and ground tracks.
- Runs an end-to-end estimation loop with a simple NLLS position estimator.

Notes
-----
Depends on local modules:
- `transformation` (coordinate transforms, dynamics)
- `spacetools` (plotting helpers; e.g., `groundtrack`)
- `spacetools.constants` (e.g., `R_EARTH`)
- `NLLS` (nonlinear least squares GNSS-like estimator)

Author: Cleaned for PEP 8, full docstrings, and unused code removed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math as m
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

import spacetools
import spacetools.constants as const
import transformation as transform


# ---------------------------- TLE utilities ----------------------------------


def _read_tle_lines(filename: str) -> List[str]:
    """
    Read a TLE file into a list of non-empty lines (stripped).

    Parameters
    ----------
    filename : str
        Path to a text file containing a standard 2-line TLE.

    Returns
    -------
    list[str]
        Non-empty, stripped lines from the file.
    """
    with open(filename, "rt", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _parse_standard_tle(line2: str) -> Tuple[float, float, float, float, float, float]:
    """
    Parse standard TLE line 2 fields used for Keplerian elements.

    Parameters
    ----------
    line2 : str
        The TLE line 2.

    Returns
    -------
    tuple
        (inclination_deg, raan_deg, eccentricity, arg_perigee_deg,
         mean_anomaly_deg, mean_motion_rev_per_day)
    """
    inc_deg = float(line2[8:16])
    raan_deg = float(line2[17:25])
    e_str = line2[26:33].strip()
    ecc = float(f"0.{e_str}") if e_str else 0.0
    argp_deg = float(line2[34:42])
    M_deg = float(line2[43:51])
    n_revday = float(line2[52:63])
    return inc_deg, raan_deg, ecc, argp_deg, M_deg, n_revday


def _parse_epoch_from_line1(line1: str) -> datetime:
    """
    Parse the epoch from TLE line 1 (YYDDD.DDDDDDDD, UTC).

    Parameters
    ----------
    line1 : str
        The TLE line 1.

    Returns
    -------
    datetime
        Epoch as an aware UTC datetime.
    """
    field = line1[18:32].strip()
    yy = int(field[:2])
    doy = float(field[2:])
    year = 2000 + yy if yy < 57 else 1900 + yy

    day_int = int(doy)
    frac = doy - day_int
    return (
        datetime(year, 1, 1, tzinfo=timezone.utc)
        + timedelta(days=day_int - 1)
        + timedelta(seconds=frac * 86400.0)
    )


def load_tle_elements_and_epoch(filename: str) -> Tuple[float, float, float, float, float, float, datetime]:
    """
    Load TLE elements and epoch from a file containing a standard 2-line TLE.

    Parameters
    ----------
    filename : str
        Path to the TLE file.

    Returns
    -------
    tuple
        (inclination_deg, raan_deg, eccentricity, arg_perigee_deg,
         mean_anomaly_deg, mean_motion_rev_per_day, epoch_utc)
    """
    lines = _read_tle_lines(filename)
    l1 = next((ln for ln in lines if ln.startswith("1 ")), None)
    l2 = next((ln for ln in lines if ln.startswith("2 ")), None)
    if l1 is None or l2 is None:
        raise ValueError("File must contain a standard 2-line TLE (lines starting with '1 ' and '2 ').")

    i_deg, raan_deg, e, argp_deg, M_deg, n_revday = _parse_standard_tle(l2)
    epoch = _parse_epoch_from_line1(l1)
    return i_deg, raan_deg, e, argp_deg, M_deg, n_revday, epoch


# ---------------------- Orbital math -----------------------------------------


def orbital_period_from_mean_motion(n_rev_day: float) -> float:
    """
    Convert mean motion [revolutions/day] to orbital period [seconds].

    Parameters
    ----------
    n_rev_day : float
        Mean motion in revolutions per day.

    Returns
    -------
    float
        Orbital period in seconds.
    """
    return 86400.0 / n_rev_day


def semimajor_axis_from_period(T: float, mu: float) -> float:
    """
    Compute semi-major axis from period and gravitational parameter.

    Parameters
    ----------
    T : float
        Orbital period [s].
    mu : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    float
        Semi-major axis [km].
    """
    return ((mu * T**2) / (4.0 * np.pi**2)) ** (1.0 / 3.0)


def eccentric_anomaly_from_mean_anomaly(M_rad: float, e: float, tol: float = 1e-10) -> float:
    """
    Solve Kepler's equation for eccentric anomaly using Newton iterations.

    Parameters
    ----------
    M_rad : float
        Mean anomaly [rad].
    e : float
        Eccentricity [-].
    tol : float, optional
        Absolute tolerance on update, by default 1e-10.

    Returns
    -------
    float
        Eccentric anomaly [rad].
    """
    E = M_rad + (e if M_rad < np.pi else -e) * 0.5
    while True:
        f = E - e * m.sin(E) - M_rad
        fp = 1 - e * m.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            return E


def true_anomaly_from_E(E_rad: float, e: float) -> float:
    """
    Convert eccentric anomaly to true anomaly.

    Parameters
    ----------
    E_rad : float
        Eccentric anomaly [rad].
    e : float
        Eccentricity [-].

    Returns
    -------
    float
        True anomaly [deg] wrapped to [0, 360).
    """
    tan_half = m.tan(E_rad / 2.0) * m.sqrt((1 + e) / (1 - e))
    th = 2.0 * m.atan(tan_half)
    th = (th + 2.0 * np.pi) % (2.0 * np.pi)
    return np.degrees(th)


def plot_height_decay(sol, R_earth=6378.137):
    """
    Plot orbital altitude decay vs time, showing one point per day.
    Uses data sampled every 12 hours from solve_ivp.
    
    Args:
        sol: solve_ivp solution with .t (s) and .y (state vectors)
        R_earth (float): Earth's radius (km)
    """
    # --- Extract data from solution ---
    t = sol.t              # time in seconds
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    h = r - R_earth        # altitude (km)

    # --- Sample every 24 hours (86400 s) ---
    day_step = 24 * 3600
    day_indices = np.arange(0, len(t), int(day_step / (t[1] - t[0])))  # step in indices
    t_days = t[day_indices] / (24 * 3600)  # convert to days
    h_days = h[day_indices]

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(t_days, h_days, color='royalblue', linewidth=1.5)
    plt.title("Orbital Altitude Decay (1-day sampling)", fontsize=13)
    plt.xlabel("Time (days)")
    plt.ylabel("Altitude (km)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def propagate_orbit_eci(
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    h: float,
    mu: float,
    theta0_deg: float,
    t_eval: np.ndarray,
    mode: int
) -> np.ndarray:
    """
    Propagate a Keplerian orbit in ECI using a two-body ODE integrator.

    Parameters
    ----------
    e : float
        Eccentricity [-].
    i_deg : float
        Inclination [deg].
    raan_deg : float
        Right ascension of ascending node [deg].
    argp_deg : float
        Argument of perigee [deg].
    h : float
        Specific angular momentum [km^2/s].
    mu : float
        Gravitational parameter [km^3/s^2].
    theta0_deg : float
        Initial true anomaly [deg].
    t_eval : np.ndarray
        Monotonic array of times [s] over which to integrate.
    mode: int
        Decides which model to run

    Returns
    -------
    np.ndarray
        ECI position history, shape (3, N) [km].
    """
    r_peri, v_peri = transform.perifocal_coordinates(h, mu, np.deg2rad(theta0_deg), e)
    Q = transform.euler_angle_sequence(np.deg2rad(raan_deg), np.deg2rad(i_deg), np.deg2rad(argp_deg))
    r_eci0 = transform.perifocal_to_ECI(r_peri, Q)
    v_eci0 = transform.perifocal_to_ECI(v_peri, Q)

    state0 = np.concatenate((r_eci0, v_eci0))
    t_span = (float(t_eval[0]), float(t_eval[-1]))
    R_earth = 6378.137          # km
    J2_earth = 1.08262668e-3
    Cd = 2.2
    A = 1e-8
    m = 8
    if (mode == 0):
        sol = spi.solve_ivp(
            transform.system_dynamics,
            t_span,
            state0,
            t_eval=t_eval,
            args=(mu,),
            max_step=1,
        )
    elif (mode == 1):
        sol = spi.solve_ivp(
            transform.system_dynamics_drag_km,
            t_span,
            state0,
            t_eval=t_eval,
            args=(mu, Cd, A, m),
            max_step=24 * 60 * 60,
        )
        plot_height_decay(sol)
    elif (mode == 2):
        sol = spi.solve_ivp(
            transform.system_dynamics_J2, 
            t_span, 
            state0,
            t_eval=t_eval, 
            args=(mu, J2_earth, R_earth),
            max_step=10)

    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")

    return np.vstack((sol.y[0], sol.y[1], sol.y[2])), np.vstack((sol.y[3], sol.y[4], sol.y[5]))


def eci_to_latlon_2xN(orbit_eci: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """
    Convert an ECI position history to latitude/longitude (2 x N).

    Parameters
    ----------
    orbit_eci : np.ndarray
        ECI position history, shape (3, N) [km].
    t_eval : np.ndarray
        Times corresponding to `orbit_eci` [s].

    Returns
    -------
    np.ndarray
        Latitude and longitude array, shape (2, N) [deg].
    """
    omega_earth = 7.2921159e-5  # [rad/s]
    r_ecef = transform.ECI_to_ECEF(orbit_eci, t_eval, omega_earth)
    latlon = transform.declination_ascension(r_ecef)

    if latlon.shape[0] == 2 and latlon.shape[1] != 2:
        lat_deg, lon_deg = latlon[0], latlon[1]
    elif latlon.shape[1] == 2:
        lat_deg, lon_deg = latlon[:, 0], latlon[:, 1]
    else:
        raise ValueError("declination_ascension must return (2 x N) or (N x 2).")

    lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
    return np.vstack((lat_deg, lon_wrapped))


# ------------------------------- Plotting ------------------------------------


def _set_equal_3d(ax, data_3xN_list: Iterable[np.ndarray]) -> None:
    """
    Set equal aspect limits for a 3D axis given several (3 x N) arrays.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Target 3D axis.
    data_3xN_list : iterable of np.ndarray
        Arrays with shape (3, N) to bound the axes.
    """
    mins = np.array([+np.inf, +np.inf, +np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])

    for arr in data_3xN_list:
        mins = np.minimum(mins, arr.min(axis=1))
        maxs = np.maximum(maxs, arr.max(axis=1))

    centers = (mins + maxs) / 2.0
    ranges = (maxs - mins) / 2.0
    max_r = float(np.max(ranges))

    ax.set_xlim(centers[0] - max_r, centers[0] + max_r)
    ax.set_ylim(centers[1] - max_r, centers[1] + max_r)
    ax.set_zlim(centers[2] - max_r, centers[2] + max_r)
    ax.set_box_aspect([1, 1, 1])


def plot_multiple_orbits_eci(orbits_eci: List[np.ndarray], labels: List[str]) -> None:
    """
    Plot several ECI orbits in 3D with a wireframe Earth.

    Parameters
    ----------
    orbits_eci : list[np.ndarray]
        List of (3, N) ECI position arrays.
    labels : list[str]
        Legend labels for each orbit.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    rE_km = const.R_EARTH / 1e3
    u, v = np.mgrid[0 : 2 * np.pi : 120j, 0 : np.pi : 60j]
    x = rE_km * np.cos(u) * np.sin(v)
    y = rE_km * np.sin(u) * np.sin(v)
    z = rE_km * np.cos(v)
    ax.plot_surface(x, y, z, color="tab:blue", alpha=0.15, linewidth=0)

    for arr, lab in zip(orbits_eci, labels):
        ax.plot(arr[0], arr[1], arr[2], linewidth=1.2, label=lab)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    _set_equal_3d(ax, orbits_eci)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.title("Buccaneer Orbit")
    plt.tight_layout()
    plt.show()

def plot_groundtracks_spacetools_full(latlons_2xN: List[np.ndarray], labels: List[str]) -> None:
    """
    Plot full ground tracks for many satellites.

    Parameters
    ----------
    latlons_2xN : list[np.ndarray]
        Each item is a (2, N) array of [lat; lon] degrees.
    labels : list[str]
        Labels for each satellite.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for latlon, lab in zip(latlons_2xN, labels):
        spacetools.groundtrack(
            ax=ax,
            latlon=latlon,
            arrows=True,
            arrow_interval=3000,
            arrow_kwargs=dict(linewidth=0.5, arrowstyle="-|>", mutation_scale=10),
            label=lab,
            linewidth=1.2,
        )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()



# -------------------------- Display -----------------------------


def display_orbital_parameters(a: float, i: float, RAAN: float, 
                               arg_perigee: float, e_mag: float,
                                 h_mag: float, theta: float) -> None:
    """ Displays the orbital parameters of the Bucanneer orbit

    Args:
        a (float): semimajor axis (km)
        i (float):inclination angle (deg)
        RAAN (float): RAAn (deg)
        arg_perigee (float): argument of perigee (deg)
        e_mag (float): eccentircty (unitless)
        h_mag (float): agular momentum (km^2/s)
        theta (float): true anomaly (theta)
    """
    print(f"Semimajor axis: {a}km")
    print(f"Inclination: {i} degrees")
    print(f"RAAN: {RAAN} degrees")
    print(f"Argument of perigee: {arg_perigee} degrees")
    print(f"Eccentricity {e_mag}")
    print(f"Specific Angular Momentum: {h_mag} km^2/s")
    print(f"True anomaly: {theta} degrees")
    

# ----------------------------- End-to-end run --------------------------------


def run_all_gsat() -> Tuple[np.ndarray, np.ndarray]:
    """
    Run propagation and plotting for a single TLE satellite (e.g. Buccaneer).
    
    Steps:
    - Load TLE and extract elements
    - Propagate 24-hour orbit from its own epoch
    - Convert to ECEF and latitude/longitude
    - Plot 3D orbit and ground track
    - Return (orbit_eci, latlon_full)
    """
    mu = 398600.0  # km^3/s^2 (Earth)
    tle_file = "TLE.txt"  # path to your single TLE file
    mode = 0 # which simulator you want

    # --- Load TLE elements and epoch ---
    i_deg, raan_deg, e, argp_deg, M_deg, n_rev_day, tle_epoch = load_tle_elements_and_epoch(tle_file)

    # --- Compute orbit parameters directly from TLE ---
    T_sec = orbital_period_from_mean_motion(n_rev_day)
    print(T_sec)
    a_km = semimajor_axis_from_period(T_sec, mu)
    p_km = a_km * (1.0 - e * e)
    h = np.sqrt(mu * p_km)

    # Compute initial true anomaly from mean anomaly
    M_rad = np.deg2rad(M_deg)
    E_rad = eccentric_anomaly_from_mean_anomaly(M_rad, e)
    theta0_deg = true_anomaly_from_E(E_rad, e)
    display_orbital_parameters(a_km, i_deg, raan_deg, argp_deg, e, h, theta0_deg)
    # --- Propagate for 24 hours ---
    if (mode == 0):
        t_eval = np.linspace(0.0,  24 * 3600, 24 * 60) 
    elif (mode == 1):
       t_eval = np.linspace(0.0,  31 * 24 * 3600, 24 * 60 * 60)
    elif (mode == 2):
        t_eval = np.linspace(0.0,  10 * 24 * 3600,  60 * 60) 
    r_eci, v_eci = propagate_orbit_eci(
        e, i_deg, raan_deg, argp_deg, h, mu, theta0_deg, t_eval, mode
    )
    
    # --- Convert to ECEF and Lat/Lon ---
    orbit_ecef = transform.ECI_to_ECEF(r_eci, t_eval, 7.2921159e-5)
    latlon_full = eci_to_latlon_2xN(r_eci, t_eval)

    # --- Plot results ---
    plot_multiple_orbits_eci([r_eci], ["Buccaneer"])
    plot_groundtracks_spacetools_full([latlon_full], ["Buccaneer"])

    return r_eci, v_eci, t_eval