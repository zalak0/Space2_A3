import numpy as np
import vec_transforms as vt
import short_formulae as form

def sun_vector_eci(julian_date: float) -> np.ndarray:
    """
    Compute approximate Sun position vector in ECI frame.
    Uses simplified algorithm (accurate to ~0.01 degrees).
    
    Args:
        julian_date: Julian date (days since J2000 epoch)
    
    Returns:
        Unit vector pointing from Earth to Sun in ECI frame, shape (3,)
    """
    # Days since J2000.0 epoch (Jan 1, 2000, 12:00 TT)
    T = julian_date / 36525.0  # Julian centuries since J2000
    
    # Mean longitude of the Sun (degrees)
    L = 280.460 + 36000.771 * T
    
    # Mean anomaly of the Sun (degrees)
    g = 357.528 + 35999.050 * T
    g_rad = np.deg2rad(g)
    
    # Ecliptic longitude (degrees)
    lambda_sun = L + 1.915 * np.sin(g_rad) + 0.020 * np.sin(2 * g_rad)
    lambda_sun_rad = np.deg2rad(lambda_sun)
    
    # Obliquity of ecliptic (degrees)
    epsilon = 23.439 - 0.013 * T
    epsilon_rad = np.deg2rad(epsilon)
    
    # Sun position in ECI (unit vector)
    x = np.cos(lambda_sun_rad)
    y = np.sin(lambda_sun_rad) * np.cos(epsilon_rad)
    z = np.sin(lambda_sun_rad) * np.sin(epsilon_rad)
    
    sun_vec = np.array([x, y, z])
    
    # Normalize (should already be ~1, but ensure it)
    return sun_vec / np.linalg.norm(sun_vec)

def is_in_eclipse(r_eci: np.ndarray, sun_eci: np.ndarray) -> bool:
    """
    Check if satellite is in Earth's shadow (umbra).
    
    Args:
        r_eci: Satellite position in ECI [km]
        sun_eci: Unit vector to Sun in ECI
    
    Returns:
        True if satellite is in eclipse (no sun sensor measurements possible)
    """
    # Simple cylindrical shadow model
    R_earth = 6378.137  # km
    
    # Satellite position relative to Sun-Earth line
    sat_along_sun = np.dot(r_eci, sun_eci)
    
    # If satellite is on day side, not in eclipse
    if sat_along_sun > 0:
        return False
    
    # Distance from Sun-Earth line
    r_perp = r_eci - sat_along_sun * sun_eci
    dist_from_shadow_axis = np.linalg.norm(r_perp)
    
    # In shadow if closer than Earth radius to shadow axis
    return dist_from_shadow_axis < R_earth

def get_sun_reference_vectors_lvlh(r_eci: np.ndarray, 
                                    v_eci: np.ndarray,
                                    times: np.ndarray,
                                    epoch_jd: float) -> np.ndarray:
    """
    Compute sun direction vectors in LVLH frame for all time steps.
    
    Args:
        r_eci: Position vectors in ECI, shape (3, N) [km]
        v_eci: Velocity vectors in ECI, shape (3, N) [km/s]
        times: Time array [seconds since epoch]
        epoch_jd: Julian date of epoch (days since J2000)
    
    Returns:
        Sun vectors in LVLH frame, shape (3, N) [unit vectors]
    """
    N = r_eci.shape[1]
    sun_lvlh = np.zeros((3, N))
    
    for k in range(N):
        # 1. Get current Julian date
        jd = form.seconds_to_julian_date(times[k], epoch_jd)
        
        # 2. Compute sun direction in ECI
        sun_eci = sun_vector_eci(jd)
        
        # 3. Build LVLH frame for this instant
        r_k = r_eci[:, k]
        v_k = v_eci[:, k]
        
        Z_lvlh = -r_k / np.linalg.norm(r_k)
        h_vec = np.cross(r_k, v_k)
        Y_lvlh = -h_vec / np.linalg.norm(h_vec)
        X_lvlh = np.cross(Y_lvlh, Z_lvlh)
        
        R_ECI_to_LVLH = np.vstack([X_lvlh, Y_lvlh, Z_lvlh])
        
        # 4. Transform sun vector to LVLH
        sun_lvlh[:, k] = R_ECI_to_LVLH @ sun_eci
    
    return sun_lvlh

def simulate_fine_sun_sensor(sun_lvlh: np.ndarray,
                              true_attitude_deg: np.ndarray,
                              noise_std_deg: float = 0.01) -> np.ndarray:
    """
    Simulate fine sun sensor measurements in body frame.
    
    Fine sun sensors typically have ~0.01° to 0.1° accuracy.
    
    Args:
        sun_lvlh: Sun direction in LVLH, shape (3, N) or (3,)
        true_attitude_deg: True Euler angles [roll, pitch, yaw] in degrees
        noise_std_deg: Sensor noise standard deviation [degrees]
    
    Returns:
        Sun measurements in body frame, shape (3, N) or (3,)
    """
    # Convert Euler angles to rotation matrix
    phi, theta, psi = np.deg2rad(true_attitude_deg)
    
    noise = np.random.normal(0, np.deg2rad(noise_std_deg))
    
    print(f"Sun sensor deviation: {np.deg2rad(noise_std_deg):.4f}")
    print(f"Sun sensor noise: {noise:.4f}")
    
    # Handle single vector or array
    if sun_lvlh.ndim == 1:
        # Single measurement
        sun_body = vt.lvlh_to_body(phi, theta, psi, sun_lvlh)
        
        # Add noise
        sun_body_noisy = sun_body + noise
        
        # Renormalize
        return sun_body_noisy / np.linalg.norm(sun_body_noisy)
    
    else:
        # Multiple measurements
        N = sun_lvlh.shape[1]
        sun_body = np.zeros((3, N))
        
        for k in range(N):
            sun_body_k = vt.lvlh_to_body(phi, theta, psi, sun_lvlh[:, k])
            
            # Add noise
            sun_body_noisy = sun_body_k + noise
            
            # Renormalize
            sun_body[:, k] = sun_body_noisy / np.linalg.norm(sun_body_noisy)
        
        return sun_body
