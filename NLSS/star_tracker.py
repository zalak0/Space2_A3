# star_tracker.py
import numpy as np
from scipy.spatial.transform import Rotation as R
import spacetools

# ========== 1. Tiny internal catalog (can expand or import real one later) ==========
CATALOG = [
    ("Sirius",   101.28715533, -16.71611586, -1.46),
    ("Canopus",   95.98787783, -52.69566150, -0.74),
    ("Arcturus", 213.91530001,  19.18240922, -0.05),
    ("Vega",     279.23473479,  38.78368896,  0.03),
    ("Capella",   79.17232794,  45.99799147,  0.08),
    ("Rigel",     78.63446707,  -8.20163837,  0.12),
    ("Procyon",  114.825493,    5.224993,     0.38),
    ("Betelgeuse", 88.792939,    7.407064,     0.42),
    ("Achernar",  24.4286,     -57.23675,     0.46),
    ("Altair",   297.6958273,   8.8683212,    0.77),
]

def ra_dec_to_unit_vector(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)])

CAT_VECS = np.array([ra_dec_to_unit_vector(ra, dec) for (_, ra, dec, _) in CATALOG])
CAT_NAMES = [c[0] for c in CATALOG]

def simulate_star_tracker(star_vectors_eci, true_attitude_deg,
                          centroid_sigma_arcsec=15.0,
                          fov_half_angle_deg=45.0):
    """
    Simulate star tracker measurements with detailed diagnostics.
    """
    print(f"\n[DEBUG] Star Tracker Simulation:")
    print(f"  Input shape: {star_vectors_eci.shape}")
    print(f"  True attitude: {true_attitude_deg}")
    print(f"  FOV half-angle: {fov_half_angle_deg}°")
    
    sigma_rad = np.deg2rad(centroid_sigma_arcsec / 3600.0)
    
    # Build rotation matrix LVLH -> Body
    phi, theta, psi = np.deg2rad(true_attitude_deg)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    R_body_lvlh = np.array([
        [c_theta*c_psi, c_theta*s_psi, -s_theta],
        [s_phi*s_theta*c_psi - c_phi*s_psi, 
         s_phi*s_theta*s_psi + c_phi*c_psi, 
         s_phi*c_theta],
        [c_phi*s_theta*c_psi + s_phi*s_psi, 
         c_phi*s_theta*s_psi - s_phi*c_psi, 
         c_phi*c_theta]
    ])
    
    print(f"\n  Rotation matrix R_body_lvlh:")
    print(f"    {R_body_lvlh[0]}")
    print(f"    {R_body_lvlh[1]}")
    print(f"    {R_body_lvlh[2]}")
    
    # Handle different input shapes
    if star_vectors_eci.ndim == 1:
        # Single star: shape (3,)
        star_vectors_eci = star_vectors_eci.reshape(1, 3)
    elif star_vectors_eci.ndim == 2:
        if star_vectors_eci.shape[0] == 3:
            # Shape (3, N) - transpose to (N, 3)
            star_vectors_eci = star_vectors_eci.T
        # else already (N, 3)
    
    n_stars = star_vectors_eci.shape[0]
    print(f"\n  Number of stars: {n_stars}")
    
    # Transform stars to body frame
    stars_body = np.zeros((n_stars, 3))
    for i in range(n_stars):
        stars_body[i] = R_body_lvlh @ star_vectors_eci[i]
        print(f"  Star {i}:")
        print(f"    ECI:  {star_vectors_eci[i]}")
        print(f"    Body: {stars_body[i]}")
    
    # Check visibility
    boresight_body = np.array([1, 0, 0])
    print(f"\n  Boresight direction (body): {boresight_body}")
    
    cos_angles = stars_body @ boresight_body
    angles_deg = np.rad2deg(np.arccos(np.clip(cos_angles, -1, 1)))
    
    print(f"\n  Star angles from boresight:")
    for i in range(n_stars):
        visible_str = "✓ VISIBLE" if angles_deg[i] <= fov_half_angle_deg else "✗ NOT VISIBLE"
        print(f"    Star {i}: {angles_deg[i]:6.2f}° - {visible_str}")
    
    visible_mask = angles_deg <= fov_half_angle_deg
    visible_indices = np.where(visible_mask)[0]
    
    print(f"\n  Result: {len(visible_indices)} / {n_stars} stars visible")
    
    if len(visible_indices) == 0:
        print(f"\n  ❌ NO STARS VISIBLE!")
        print(f"  Suggestions:")
        print(f"    - Increase FOV (currently {fov_half_angle_deg}°)")
        print(f"    - Check if stars are in correct frame")
        print(f"    - Verify rotation matrix is correct")
        raise ValueError(f"Only {len(visible_indices)} stars visible - need at least 2!")
    
    # Add centroid noise
    body_vectors = []
    for idx in visible_indices:
        v = stars_body[idx]
        
        # Perpendicular noise axis
        if abs(v[0]) < 0.9:
            noise_axis = np.cross(v, [1, 0, 0])
        else:
            noise_axis = np.cross(v, [0, 1, 0])
        
        noise_axis /= np.linalg.norm(noise_axis)
        noise_angle = np.random.normal(0, sigma_rad)
        
        # Rodrigues rotation
        v_noisy = (v * np.cos(noise_angle) +
                   np.cross(noise_axis, v) * np.sin(noise_angle) +
                   noise_axis * np.dot(noise_axis, v) * (1 - np.cos(noise_angle)))
        
        body_vectors.append(v_noisy / np.linalg.norm(v_noisy))
    
    return np.array(body_vectors).T, visible_indices

def get_star_reference_vectors_lvlh(r_eci, v_eci, n_stars=3):
    """
    Return star vectors transformed to LVLH frame at each time step.
    Stars are fixed in ECI, but we need them in LVLH for body frame comparison.
    """
    N_times = r_eci.shape[1]
    
    # Sort by magnitude (brightest first)
    magnitudes = [c[3] for c in CATALOG]
    brightest_indices = np.argsort(magnitudes)[:n_stars]
    
    # Get star unit vectors in ECI (inertial frame)
    star_vectors_eci = CAT_VECS[brightest_indices]  # shape (n_stars, 3)
    
    print(f"    Selected {n_stars} brightest stars:")
    for idx in brightest_indices:
        print(f"      {CAT_NAMES[idx]} (mag {CATALOG[idx][3]:.2f})")
    
    # Transform each star to LVLH at each time step
    star_lvlh = np.zeros((3, n_stars, N_times))
    
    for k in range(N_times):
        # Build LVLH frame at this time
        r_k = r_eci[:, k]
        v_k = v_eci[:, k]
        
        r_norm = np.linalg.norm(r_k)
        Z_lvlh = -r_k / r_norm  # Nadir (toward Earth)
        
        h = np.cross(r_k, v_k)
        h_norm = np.linalg.norm(h)
        Y_lvlh = -h / h_norm  # Opposite orbit normal
        
        X_lvlh = np.cross(Y_lvlh, Z_lvlh)  # Forward
        
        # Rotation matrix ECI -> LVLH
        R_LVLH_ECI = np.vstack([X_lvlh, Y_lvlh, Z_lvlh])  # (3, 3)
        
        # Transform each star from ECI to LVLH
        for i in range(n_stars):
            star_lvlh[:, i, k] = R_LVLH_ECI @ star_vectors_eci[i]
    
    return star_lvlh  # shape (3, n_stars, N_times) - NOW in LVLH!