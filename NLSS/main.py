## SID: 530575836
import find_orbit 
import numpy as np
import short_formulae as form
import vec_transforms as vt

import earth_sensor as earth
import sun_sensor as sun
import magnetometer as mag
import star_tracker as star

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
    # In STEP 3:
    print("\n[3] Computing reference vectors...")

    epoch_jd = form.utc_to_jd(2025, 10, 29, 0, 0, 0)

    sun_lvlh = sun.get_sun_reference_vectors_lvlh(r_eci, v_eci, t_eval, epoch_jd)
    earth_lvlh = earth.get_earth_reference_vectors_lvlh(r_eci, v_eci)
    mag_lvlh = mag.get_mag_reference_vectors_lvlh(r_eci, v_eci, t_eval, epoch_jd)

    # Stars are in ECI (inertial) - they don't change with time
    star_lvlh = star.get_star_reference_vectors_lvlh(r_eci, v_eci, n_stars=5)  # Get 5 stars

    print(f"    Sun vectors computed")
    print(f"    Earth vectors computed")
    print(f"    Star vectors: {len(star_lvlh)} stars in ECI")
    
    user_mode = input("\nSelect mode (1: Sun + Earth, 2: Sun + Mag, 3: Star (Don't use mode 3, underworks)): ")
    
    # ========================================================
    # STEP 4: SELECT AND SIMULATE SENSORS
    # ========================================================
    print("\n[4] Simulating body frame measurements...")
    
    true_attitude_deg = np.array([11.0, 32.0, -45.0])
    print(f"    True attitude (for testing): Roll={true_attitude_deg[0]}°, "
          f"Pitch={true_attitude_deg[1]}°, Yaw={true_attitude_deg[2]}°")
    
    if user_mode == '1':
        # ========================================================
        # MODE 1: Sun + Earth Sensors
        # ========================================================
        print("    Mode: Sun + Earth sensors")
        
        N_vectors = 2
        
        # Reference vectors
        reference_vectors_lvlh = np.zeros((3, N_vectors, N_times))
        reference_vectors_lvlh[:, 0, :] = sun_lvlh
        reference_vectors_lvlh[:, 1, :] = earth_lvlh
        
        # Simulate sensors
        sun_body = sun.simulate_fine_sun_sensor(sun_lvlh, true_attitude_deg)
        earth_body = earth.simulate_ir_earth_sensor(earth_lvlh, true_attitude_deg)
        
        # Body measurements
        body_measurements = np.zeros((3, N_vectors, N_times))
        body_measurements[:, 0, :] = sun_body
        body_measurements[:, 1, :] = earth_body

    elif user_mode == '2':
        # ========================================================
        # MODE 2: Sun + Magnetometer
        # ========================================================
        print("    Mode: Sun sensor + Magnetometer")
        
        N_vectors = 2
        
        # Reference vectors
        reference_vectors_lvlh = np.zeros((3, N_vectors, N_times))
        reference_vectors_lvlh[:, 0, :] = sun_lvlh
        reference_vectors_lvlh[:, 1, :] = mag_lvlh
        
        # Simulate sensors
        sun_body = sun.simulate_fine_sun_sensor(sun_lvlh, true_attitude_deg)
        mag_body = mag.simulate_magnetometer(mag_lvlh, true_attitude_deg)
        
        # Body measurements
        body_measurements = np.zeros((3, N_vectors, N_times))
        body_measurements[:, 0, :] = sun_body
        body_measurements[:, 1, :] = mag_body
    
    elif user_mode == '3':
        # ========================================================
        # MODE 2: Star Tracker Only
        # ========================================================
        print("    Mode: Star tracker")
        
        # Test visibility at first epoch
        test_epoch = 0
        star_body_test, visible_indices = star.simulate_star_tracker(
            star_lvlh[:, :, test_epoch],
            true_attitude_deg
        )
        
        N_visible = len(visible_indices)
        print(f"      Visible stars: {N_visible} out of {star_lvlh.shape[1]}")
        
        if N_visible < 2:
            raise ValueError(f"Only {N_visible} stars visible - need at least 2!")
        
        # Use only visible stars
        N_vectors = N_visible
        reference_vectors_lvlh = star_lvlh[:, visible_indices, :]  # (3, N_visible, N_times)
        
        # Simulate measurements for ALL epochs
        # (We need this for all N_times even though NLLS only uses selected epochs)
        body_measurements = np.zeros((3, N_vectors, N_times))
        
        print(f"      Simulating star tracker measurements...")
        for k in range(N_times):
            star_body_k, _ = star.simulate_star_tracker(
                reference_vectors_lvlh[:, :, k],
                true_attitude_deg
            )
            body_measurements[:, :, k] = star_body_k
        
        print(f"      Star tracker: 0.001-0.01° accuracy")
        print(f"      Using {N_vectors} visible stars")
    
    else:
        raise ValueError(f"Invalid mode '{user_mode}'. Choose '1' or '2'.")
    
    # ========================================================
    # STEP 5: SELECT EPOCHS FOR ATTITUDE DETERMINATION
    # ========================================================
    print("\n[5] Selecting measurement epochs...")
    
    n_epochs = 10
    selected_epochs = np.linspace(0, N_times-1, n_epochs, dtype=int)
    print(f"    Selected {n_epochs} epochs for NLLS")
    
    # ========================================================
    # STEP 6: RUN NLLS ATTITUDE DETERMINATION
    # ========================================================
    print("\n[6] Running NLLS attitude determination...")
    
    attitude_stats = ads.solve_attitude_determination(
        body_measurements_all=body_measurements,
        reference_vectors_lvlh=reference_vectors_lvlh,
        selected_epochs=selected_epochs,
        x_init=np.zeros(3),
        weight=None,
        max_iter=50,
        tol=1e-6
    )
    
    # ========================================================
    # STEP 7: VALIDATION & ANALYSIS
    # ========================================================
    print("\n" + "="*70)
    print("VALIDATION - Compare to True Attitude")
    print("="*70)
    
    estimated = attitude_stats["estimated_attitudes_deg"]
    errors = estimated - true_attitude_deg
    
    # Calculate RSS pointing error (as per your requirement)
    rss_mean_errors = np.sqrt(np.sum(np.mean(errors, axis=0)**2))
    rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    print(f"\nTrue attitude:      [{true_attitude_deg[0]:.2f}°, "
          f"{true_attitude_deg[1]:.2f}°, {true_attitude_deg[2]:.2f}°]")
    print(f"Mean estimated:     [{np.mean(estimated[:, 0]):.2f}°, "
          f"{np.mean(estimated[:, 1]):.2f}°, {np.mean(estimated[:, 2]):.2f}°]")
    
    print(f"\nIndividual axis errors:")
    print(f"  Roll error:  {np.mean(errors[:, 0]):.4f}° ± {np.std(errors[:, 0]):.4f}°")
    print(f"  Pitch error: {np.mean(errors[:, 1]):.4f}° ± {np.std(errors[:, 1]):.4f}°")
    print(f"  Yaw error:   {np.mean(errors[:, 2]):.4f}° ± {np.std(errors[:, 2]):.4f}°")
    
    print(f"\nCombined errors:")
    print(f"  RSS (mean errors):  {rss_mean_errors:.4f}°")
    print(f"  RMS error:          {rms_error:.4f}°")
    
    print(f"\nGeometric quality:")
    print(f"  Mean ADOP:          {np.mean(attitude_stats['adops']):.4f}")
    print(f"  Mean iterations:    {np.mean(attitude_stats['iterations']):.1f}")
    
    # Success criteria
    THRESHOLD = 0.05  # degrees
    print(f"\n" + "="*70)
    print(f"SUCCESS CRITERION: RSS < {THRESHOLD}°")
    print("="*70)
    print(f"Your result: {rss_mean_errors:.4f}°")
    
    if rss_mean_errors < THRESHOLD:
        print(f"✓ SUCCESS: RSS pointing error meets requirement!")
    else:
        print(f"✗ FAIL: RSS exceeds threshold by {rss_mean_errors - THRESHOLD:.4f}°")
        print(f"\nImprovement needed: {rss_mean_errors/THRESHOLD:.1f}× current threshold")
    
    return attitude_stats


if __name__ == "__main__":
    results = main()