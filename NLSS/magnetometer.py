# magnetometer.py
import numpy as np
import vec_transforms as vt

def get_mag_reference_vectors_lvlh(r_eci: np.ndarray, 
                                    v_eci: np.ndarray,
                                    times: np.ndarray,
                                    epoch_jd: float) -> np.ndarray:
    """Simple dipole magnetic field model."""
    N = r_eci.shape[1]
    mag_lvlh = np.zeros((3, N))
    
    # Earth's magnetic dipole (simplified)
    tilt_angle = np.deg2rad(11.0)  # Magnetic axis tilt
    
    for k in range(N):
        r_k = r_eci[:, k]
        r_mag = np.linalg.norm(r_k)
        
        # Dipole axis in ECI (tilted from Z-axis)
        m_axis = np.array([np.sin(tilt_angle), 0, np.cos(tilt_angle)])
        
        # Dipole field: B ∝ 3(m·r̂)r̂ - m
        r_hat = r_k / r_mag
        m_dot_r = np.dot(m_axis, r_hat)
        B_eci = 3 * m_dot_r * r_hat - m_axis
        B_eci = B_eci / np.linalg.norm(B_eci)
        
        # Transform to LVLH
        v_k = v_eci[:, k]
        
        # Build LVLH frame
        Z_lvlh = -r_k / r_mag
        h = np.cross(r_k, v_k)
        Y_lvlh = -h / np.linalg.norm(h)
        X_lvlh = np.cross(Y_lvlh, Z_lvlh)
        R_LVLH_ECI = np.vstack([X_lvlh, Y_lvlh, Z_lvlh])
        
        mag_lvlh[:, k] = R_LVLH_ECI @ B_eci
    
    return mag_lvlh


def simulate_magnetometer(mag_lvlh: np.ndarray,
                          true_attitude_deg: np.ndarray,
                          noise_std_deg: float = 0.02) -> np.ndarray:
    """Simulate magnetometer measurements."""
    phi, theta, psi = np.deg2rad(true_attitude_deg)
    
    noise = np.random.normal(0, np.deg2rad(noise_std_deg))
    
    print(f"Magnetometer deviation: {np.deg2rad(noise_std_deg):.4f}")
    print(f"Magnetometer noise: {noise:4f}")
    
    if mag_lvlh.ndim == 1:
        mag_body = vt.lvlh_to_body(phi, theta, psi, mag_lvlh)
        mag_body_noisy = mag_body + noise
        return mag_body_noisy / np.linalg.norm(mag_body_noisy)
    else:
        N = mag_lvlh.shape[1]
        mag_body = np.zeros((3, N))
        for k in range(N):
            mag_body_k = vt.lvlh_to_body(phi, theta, psi, mag_lvlh[:, k])
            mag_body_noisy = mag_body_k + noise
            mag_body[:, k] = mag_body_noisy / np.linalg.norm(mag_body_noisy)
        return mag_body