import numpy as np

def eci_to_lvlh(r_eci: np.ndarray, v_eci: np.ndarray):
    """
    Return r_lvlh, v_lvlh and also the rotation matrix R_ECI_to_LVLH (optional).
    r_eci, v_eci: shape (3,)
    returns: r_lvlh (3,), v_lvlh (3,), R_ECI_to_LVLH (3x3)
    """
    r_norm = np.linalg.norm(r_eci)
    if r_norm == 0:
        raise ValueError("r_eci has zero norm")

    # Z axis: points toward Earth center (i.e. -r direction)
    Z_lvlh = -r_eci / r_norm

    # Orbit normal (angular momentum)
    h_vec = np.cross(r_eci, v_eci)
    h_norm = np.linalg.norm(h_vec)
    if h_norm == 0:
        raise ValueError("h (orbit normal) has zero norm")

    # Y axis: opposite orbit normal
    Y_lvlh = -h_vec / h_norm

    # X axis: completes right-hand system
    X_lvlh = np.cross(Y_lvlh, Z_lvlh)
    X_lvlh = X_lvlh / np.linalg.norm(X_lvlh)

    # Rotation matrix ECI -> LVLH: rows are LVLH basis vectors in ECI coords
    R_ECI_to_LVLH = np.vstack([X_lvlh, Y_lvlh, Z_lvlh])  # shape (3,3)

    # Transform
    r_lvlh = R_ECI_to_LVLH @ r_eci
    v_lvlh = R_ECI_to_LVLH @ v_eci

    return r_lvlh, v_lvlh

def lvlh_to_body(phi: float, theta: float, psi: float,
                 mag_lvlh : np.ndarray) -> np.ndarray:
    """
    Z-Y-X Euler angle sequence to rotation matrix (slide 86).
    
    Args:
        phi: Roll angle [rad]
        theta: Pitch angle [rad]
        psi: Yaw angle [rad]
    """
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    R = np.array([
        [c_theta*c_psi, 
         c_theta*s_psi, 
         -s_theta],
        [s_phi*s_theta*c_psi - c_phi*s_psi, 
         s_phi*s_theta*s_psi + c_phi*c_psi, 
         s_phi*c_theta],
        [c_phi*s_theta*c_psi + s_phi*s_psi, 
         c_phi*s_theta*s_psi - s_phi*c_psi, 
         c_phi*c_theta]
    ])
    
    
    return R @ mag_lvlh
