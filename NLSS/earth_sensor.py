import numpy as np
import spacetools.constants as const
import vec_transforms as vt

def get_earth_reference_vectors_lvlh(r_eci: np.ndarray,
                                     v_eci: np.ndarray) -> np.ndarray:
    """
    Compute Earth horizon sensor reference (boresight) direction vectors in LVLH frame.
    Accounts for changing altitude over time.

    Args:
        r_eci: Position vectors in ECI, shape (3, N) [km]
        v_eci: Velocity vectors in ECI, shape (3, N) [km/s]

    Returns:
        earth_lvlh: Horizon boresight vectors in LVLH, shape (3, N)
    """
    R_earth = const.R_EARTH/1e3  # [km]
    N = r_eci.shape[1]
    earth_lvlh = np.zeros((3, N))

    for k in range(N):
        # --- Compute instantaneous LVLH frame ---
        r_k, v_k = r_eci[:, k], v_eci[:, k]
        r_norm = np.linalg.norm(r_k)

        Z_lvlh = -r_k / r_norm
        h_vec = np.cross(r_k, v_k)
        Y_lvlh = -h_vec / np.linalg.norm(h_vec)
        X_lvlh = np.cross(Y_lvlh, Z_lvlh)
        R_ECI_to_LVLH = np.vstack([X_lvlh, Y_lvlh, Z_lvlh])

        # --- Instantaneous horizon tilt ---
        theta_hor = np.arcsin(R_earth / r_norm)
        
        if R_earth / r_norm > 1:
            print(f"Warning: R_earth / r_norm = {R_earth / r_norm:.6f} (>1)")

        # Horizon boresight vector (tilted from nadir around +X)
        R_tilt = vt.rotation_matrix_x(theta_hor)
        horizon_vec_lvlh = R_tilt @ np.array([0, 0, -1])  # in LVLH

        # --- Store LVLH vector ---
        earth_lvlh[:, k] = horizon_vec_lvlh

    return earth_lvlh / np.linalg.norm(earth_lvlh, axis=0)


def simulate_ir_earth_sensor(earth_lvlh: np.ndarray,
                             true_attitude_deg: np.ndarray,
                             noise_std_deg: float = 1.0) -> np.ndarray:
    """
    Simulate IR Earth Horizon Sensor (CubeSense Earth Gen2).

    Args:
        earth_lvlh: Earth (boresight) reference in LVLH frame, shape (3, N)
        true_attitude_deg: [roll, pitch, yaw] in degrees
        noise_std_deg: 3*sigma accuracy ~1°, so sigma ≈ 0.33°

    Returns:
        Earth measurement vectors in body frame, shape (3, N)
    """
    # Convert accuracy (3σ) to 1σ
    noise_std_rad = np.deg2rad(noise_std_deg / 3)
    phi, theta, psi = np.deg2rad(true_attitude_deg)
    
    N = earth_lvlh.shape[1]
    earth_body = np.zeros((3, N))
    
    for k in range(N):
        v_b = vt.lvlh_to_body(phi, theta, psi, earth_lvlh[:, k])
        noise = np.random.randn(3) * noise_std_rad
        v_b_noisy = v_b + noise
        earth_body[:, k] = v_b_noisy / np.linalg.norm(v_b_noisy)
    
    return earth_body
