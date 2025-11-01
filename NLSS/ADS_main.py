import numpy as np

def solve_attitude_determination(
    body_measurements_all: np.ndarray,      # shape (3, N_vectors, N_times)
    reference_vectors_lvlh: np.ndarray,     # shape (3, N_vectors, N_times)
    selected_epochs: np.ndarray,            # which time indices to solve for
    x_init: np.ndarray = np.zeros(3),       # [phi, theta, psi] initial guess
    weight: np.ndarray = None,
    max_iter: int = 50,
    tol: float = 1e-10
) -> dict:
    """
    Solves for spacecraft attitude using NLLS on vector measurements.
    
    Args:
        body_measurements_all: Vector measurements in body frame, shape (3, N_vectors, N_times)
        reference_vectors_lvlh: Known vectors in LVLH frame, shape (3, N_vectors, N_times)
        selected_epochs: Array of time indices to solve attitude for
        x_init: Initial Euler angle guess [phi, theta, psi] in radians
        weight: Weighting matrix for measurements (N_vectors*3, N_vectors*3)
        max_iter: Maximum iterations for NLLS
        tol: Convergence tolerance in radians
    
    Returns:
        dict: Statistics including estimated attitudes, errors, DOPs, etc.
    """
    
    # Storage for results
    estimated_attitudes_deg = []
    estimated_attitudes_rad = []
    iterations_per_epoch = []
    adop_values = []
    residual_norms = []
    
    print("\n" + "="*70)
    print("ATTITUDE DETERMINATION - NLLS")
    print("="*70)
    
    # Solve attitude at each selected epoch
    for epoch_idx in selected_epochs:
        print(f"\n{'─'*70}")
        print(f"Solving attitude at epoch {epoch_idx}")
        print(f"{'─'*70}")
        
        # Extract measurements for this epoch
        # body_measurements: shape (3, N_vectors)
        # reference_vectors: shape (3, N_vectors)
        body_measurements = body_measurements_all[:, :, epoch_idx]
        reference_vectors = reference_vectors_lvlh[:, :, epoch_idx]
        
        N_vectors = body_measurements.shape[1]
        print(f"Number of vector measurements: {N_vectors}")
        
        if N_vectors < 2:
            raise ValueError("At least 2 non-parallel vector measurements required for attitude determination.")
        
        # Stack into single vectors for NLLS
        y_meas = body_measurements.T.flatten()  # shape (N_vectors*3,)
        m_ref = reference_vectors              # keep as (3, N_vectors) for model
        
        # Weight matrix (if not provided, use identity)
        if weight is None:
            W = np.eye(N_vectors * 3)
        else:
            W = weight
        
        # Initial guess
        x_est = np.array(x_init, dtype=float).copy()
        
        # NLLS iteration
        converged = False
        for iter in range(max_iter):
            # Compute Jacobian
            H = attitude_jacobian(m_ref, x_est)  # shape (N_vectors*3, 3)
            
            # Predict measurements
            y_pred = attitude_measurement_model(m_ref, x_est)  # shape (N_vectors*3,)
            
            # Residuals
            residual = y_meas - y_pred  # shape (N_vectors*3,)
            
            # Gauss-Newton update
            A = H.T @ W @ H  # (3, 3)
            B = H.T @ W @ residual  # (3,)
            delta_x = np.linalg.solve(A, B)
            
            # Update estimate
            x_est = x_est + delta_x
            
            # Wrap angles to [-π, π]
            x_est = np.arctan2(np.sin(x_est), np.cos(x_est))
            
            # Check convergence
            if np.linalg.norm(delta_x) < tol:
                converged = True
                break
        
        # Results for this epoch
        if converged:
            print(f"✓ Converged in {iter + 1} iterations")
        else:
            print(f"✗ Did not converge after {max_iter} iterations")
        
        # Calculate DOP
        ADOP = attitude_DOP(H)
        
        # Store results
        estimated_attitudes_rad.append(x_est.copy())
        estimated_attitudes_deg.append(np.rad2deg(x_est))
        iterations_per_epoch.append(iter + 1)
        adop_values.append(ADOP)
        residual_norms.append(np.linalg.norm(residual))
        
        print(f"Estimated attitude: φ={np.rad2deg(x_est[0]):.4f}°, "
              f"θ={np.rad2deg(x_est[1]):.4f}°, ψ={np.rad2deg(x_est[2]):.4f}°")
        print(f"ADOP: {ADOP:.4f}")
        print(f"Residual norm: {np.linalg.norm(residual):.6f}")
    
    # Compile statistics
    attitude_stats = {
        "estimated_attitudes_deg": np.array(estimated_attitudes_deg),  # (N_epochs, 3)
        "estimated_attitudes_rad": np.array(estimated_attitudes_rad),
        "measured_epochs": selected_epochs,
        "iterations": np.array(iterations_per_epoch),
        "adops": np.array(adop_values),
        "residual_norms": np.array(residual_norms)
    }
    
    return attitude_stats


def attitude_measurement_model(reference_vectors_lvlh: np.ndarray, 
                                x_est: np.ndarray) -> np.ndarray:
    """
    Predict body-frame measurements from Euler angles.
    
    Args:
        reference_vectors_lvlh: Known vectors in LVLH, shape (3, N_vectors)
        x_est: Current Euler angle estimate [phi, theta, psi] in radians
    
    Returns:
        Predicted measurements in body frame, shape (N_vectors*3,)
    """
    phi, theta, psi = x_est
    
    # Build rotation matrix R_body_LVLH (slide 86)
    R = euler_to_rotation_matrix(phi, theta, psi)
    
    # Predict measurements: y_body = R @ m_LVLH
    N_vectors = reference_vectors_lvlh.shape[1]
    y_pred = np.empty(N_vectors * 3)
    
    for i in range(N_vectors):
        y_pred_i = R @ reference_vectors_lvlh[:, i]
        y_pred[i*3:(i+1)*3] = y_pred_i
    
    return y_pred


def attitude_jacobian(reference_vectors_lvlh: np.ndarray,
                      x_est: np.ndarray,
                      epsilon: float = 1e-6) -> np.ndarray:
    """
    Numerical Jacobian for attitude determination (slide 126).
    
    Args:
        reference_vectors_lvlh: Reference vectors in LVLH, shape (3, N_vectors)
        x_est: Current Euler angles [phi, theta, psi] in radians
        epsilon: Perturbation size for numerical derivatives
    
    Returns:
        Jacobian matrix, shape (N_vectors*3, 3)
    """
    N_vectors = reference_vectors_lvlh.shape[1]
    J = np.empty((N_vectors * 3, 3))
    
    # Central difference for each Euler angle
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = epsilon
        
        y_plus = attitude_measurement_model(reference_vectors_lvlh, x_est + e_j)
        y_minus = attitude_measurement_model(reference_vectors_lvlh, x_est - e_j)
        
        J[:, j] = (y_plus - y_minus) / (2 * epsilon)
    
    return J


def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Z-Y-X Euler angle sequence to rotation matrix (slide 86).
    
    Args:
        phi: Roll angle [rad]
        theta: Pitch angle [rad]
        psi: Yaw angle [rad]
    
    Returns:
        3x3 rotation matrix R_body_LVLH
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
    
    return R


def attitude_DOP(H: np.ndarray) -> float:
    """
    Calculate Attitude Dilution of Precision (slide 138).
    
    Args:
        H: Jacobian matrix, shape (N_measurements, 3)
    
    Returns:
        ADOP: Scalar dilution of precision value
    """
    try:
        V = np.linalg.inv(H.T @ H)
        ADOP = np.sqrt(np.trace(V))
        return ADOP
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix in DOP calculation")
        return np.inf