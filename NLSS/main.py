## SID: 530575836
import find_orbit 
import numpy as np
import short_formulae as form
import vec_transforms as vt
import sun_vector as sun
import magnetometer as mag
import ADS_main as ads  # Your new module

def main() -> None:
    """ Main for Assignment 3 - Attitude Determination """
    
    print("="*70)
    print("SPACECRAFT ATTITUDE DETERMINATION")
    print("="*70)
    
    # ========================================================
    # STEP 1: ORBIT PROPAGATION
    # ========================================================
    print("\n[1] Propagating orbit...")
    r_eci, v_eci, t_eval = find_orbit.run_all_gsat()
    N_times = r_eci.shape[1]
    print(f"    Orbit propagated: {N_times} time steps")
    
    # ========================================================
    # STEP 2: CONVERT TO LVLH
    # ========================================================
    print("\n[2] Converting to LVLH frame...")
    r_lvlh = np.zeros_like(r_eci)
    v_lvlh = np.zeros_like(v_eci)
    
    for k in range(N_times):
        r_lvlh[:, k], v_lvlh[:, k] = vt.eci_to_lvlh(r_eci[:, k], v_eci[:, k])
    print("    LVLH conversion complete")
    
    # ========================================================
    # STEP 3: COMPUTE REFERENCE VECTORS (LVLH)
    # ========================================================
    print("\n[3] Computing reference vectors in LVLH...")
    
    epoch_jd = form.utc_to_jd(2025, 10, 29, 0, 0, 0)
    
    # Sun vectors
    sun_lvlh = sun.get_sun_reference_vectors_lvlh(r_eci, v_eci, t_eval, epoch_jd)
    print(f"    Sun vectors computed")
    
    # Magnetometer vectors
    mag_lvlh = mag.get_mag_reference_vectors_lvlh(r_eci, v_eci, t_eval, epoch_jd)
    print(f"    Magnetic field vectors computed")
    
    # Stack reference vectors: shape (3, N_vectors, N_times)
    N_vectors = 2
    reference_vectors_lvlh = np.zeros((3, N_vectors, N_times))
    reference_vectors_lvlh[:, 0, :] = sun_lvlh   # ✓ Sun
    reference_vectors_lvlh[:, 1, :] = mag_lvlh   # ✓ Mag field
    
    # print(f"    Total reference vectors: {N_vectors}")
    
    # print("\n[3.5] Diagnostic - Vector Geometry...")

    # # Check several epochs
    # test_epochs = [0, N_times//2, N_times-1]
    # for epoch in test_epochs:
    #     sun_vec = sun_lvlh[:, epoch]
    #     mag_vec = mag_lvlh[:, epoch]
        
    #     # Normalize
    #     sun_vec = sun_vec / np.linalg.norm(sun_vec)
    #     mag_vec = mag_vec / np.linalg.norm(mag_vec)
        
    #     # Angle between vectors
    #     cos_angle = np.dot(sun_vec, mag_vec)
    #     angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
        
    #     print(f"  Epoch {epoch}: Sun-Mag angle = {angle_deg:.1f}°")
        
    #     if angle_deg < 30 or angle_deg > 150:
    #         print(f"    ⚠️  WARNING: Poor geometry (too parallel/anti-parallel)")
    #     elif 60 <= angle_deg <= 120:
    #         print(f"    ✓ Good geometry (near orthogonal)")

    # print("\n  Ideal: 60° < angle < 120° (orthogonal is best)")

    
    # ========================================================
    # STEP 4: SIMULATE BODY MEASUREMENTS
    # ========================================================
    print("\n[4] Simulating body frame measurements...")
    
    true_attitude_deg = np.array([11.0, 32.0, -45.0])
    print(f"    True attitude (for testing): Roll={true_attitude_deg[0]}°, "
          f"Pitch={true_attitude_deg[1]}°, Yaw={true_attitude_deg[2]}°")
    
    # Simulate sensor measurements
    sun_body = sun.simulate_fine_sun_sensor(sun_lvlh, true_attitude_deg, 
                                             noise_std_deg=0.1)
    mag_body = mag.simulate_magnetometer(mag_lvlh, true_attitude_deg,
                                         noise_std_deg=1.0)
    
    # Stack body measurements: shape (3, N_vectors, N_times)
    body_measurements = np.zeros((3, N_vectors, N_times))
    body_measurements[:, 0, :] = sun_body  # Sun measurements
    body_measurements[:, 1, :] = mag_body  # Mag measurements
    
    # ========================================================
    # STEP 5: SELECT EPOCHS FOR ATTITUDE DETERMINATION
    # ========================================================
    print("\n[5] Selecting measurement epochs...")
    
    # Select evenly spaced epochs (similar to your GNSS code)
    n_epochs = 10
    selected_epochs = np.linspace(0, N_times-1, n_epochs, dtype=int)
    print(f"    Selected {n_epochs} epochs")
    
    # ========================================================
    # STEP 6: RUN NLLS ATTITUDE DETERMINATION
    # ========================================================
    print("\n[6] Running NLLS attitude determination...")
    
    attitude_stats = ads.solve_attitude_determination(
        body_measurements_all=body_measurements,
        reference_vectors_lvlh=reference_vectors_lvlh,
        selected_epochs=selected_epochs,
        x_init=np.zeros(3),  # Start from zero guess
        weight=None,         # Equal weighting
        max_iter=50,
        tol=1e-6
    )
    
    # ========================================================
    # STEP 7: VALIDATION & ANALYSIS
    # ========================================================
    print("\n" + "="*70)
    print("VALIDATION - Compare to True Attitude")
    print("="*70)
    
    estimated = attitude_stats["estimated_attitudes_deg"]  # (n_epochs, 3)
    errors = estimated - true_attitude_deg  # Broadcasting
    
    print(f"\nTrue attitude:      [{true_attitude_deg[0]:.2f}°, "
          f"{true_attitude_deg[1]:.2f}°, {true_attitude_deg[2]:.2f}°]")
    print(f"Mean estimated:     [{np.mean(estimated[:, 0]):.2f}°, "
          f"{np.mean(estimated[:, 1]):.2f}°, {np.mean(estimated[:, 2]):.2f}°]")
    print(f"\nMean errors:")
    print(f"  Roll error:  {np.mean(errors[:, 0]):.4f}° ± {np.std(errors[:, 0]):.4f}°")
    print(f"  Pitch error: {np.mean(errors[:, 1]):.4f}° ± {np.std(errors[:, 1]):.4f}°")
    print(f"  Yaw error:   {np.mean(errors[:, 2]):.4f}° ± {np.std(errors[:, 2]):.4f}°")
    print(f"  RMS error:   {np.sqrt(np.mean(np.sum(errors**2, axis=1))):.4f}°")
    
    print(f"\nMean ADOP: {np.mean(attitude_stats['adops']):.4f}")
    print(f"Mean iterations: {np.mean(attitude_stats['iterations']):.1f}")
    
    if np.sqrt(np.mean(np.sum(errors**2, axis=1))) < 0.5:
        print("\n✓ SUCCESS: Algorithm accurately recovered true attitude!")
    else:
        print("\n✗ WARNING: Large attitude errors detected")
    
    return attitude_stats


if __name__ == "__main__":
    results = main()