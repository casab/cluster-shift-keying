/// Integration tests for the Chen attractor and ODE integrator.
use cluster_shift_keying::dynamics::{ChenSystem, DynamicalSystem, Rk4};

/// Verify the Chen attractor stays bounded over a long integration.
#[test]
fn chen_long_integration_bounded() {
    let chen = ChenSystem::default_paper();
    let mut rk4 = Rk4::new(chen.dimension());
    let state = rk4
        .integrate_to_end(&chen, &[1.0, 1.0, 1.0], 0.001, 100_000)
        .expect("100k steps should not diverge");

    // Chen attractor is bounded within roughly |x| < 50
    for (i, &val) in state.iter().enumerate() {
        assert!(
            val.abs() < 100.0,
            "component {i} = {val} exceeds bound after 100k steps"
        );
    }
}

/// Verify the largest Lyapunov exponent is positive (i.e. the system is chaotic).
///
/// Uses a simple finite-difference method: evolve two nearby initial conditions
/// and check that they diverge exponentially.
#[test]
fn chen_positive_lyapunov_exponent() {
    let chen = ChenSystem::default_paper();
    let dt = 0.001;

    // Transient: let the system settle onto the attractor
    let mut rk4 = Rk4::new(3);
    let base_state = rk4
        .integrate_to_end(&chen, &[1.0, 1.0, 1.0], dt, 10_000)
        .expect("transient");

    // Perturbed initial condition
    let epsilon = 1e-8;
    let perturbed: Vec<f64> = base_state.iter().map(|x| x + epsilon).collect();

    // Evolve both for a measurement window
    let measure_steps = 5000;
    let mut rk4_base = Rk4::new(3);
    let mut rk4_pert = Rk4::new(3);
    let final_base = rk4_base
        .integrate_to_end(&chen, &base_state, dt, measure_steps)
        .expect("base");
    let final_pert = rk4_pert
        .integrate_to_end(&chen, &perturbed, dt, measure_steps)
        .expect("perturbed");

    // Compute separation distance
    let separation: f64 = final_base
        .iter()
        .zip(final_pert.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    // For a chaotic system, separation >> initial perturbation
    // The Lyapunov time for Chen is short, so after 5 time units
    // the separation should have grown enormously.
    assert!(
        separation > epsilon * 100.0,
        "separation {separation} is not much larger than initial perturbation {epsilon} — \
         system may not be chaotic"
    );
}

/// Verify that the RK4 integrator is consistent: same IC → same result.
#[test]
fn chen_deterministic_integration() {
    let chen = ChenSystem::default_paper();
    let ic = [5.0, -3.0, 10.0];

    let mut rk4a = Rk4::new(3);
    let mut rk4b = Rk4::new(3);

    let result_a = rk4a
        .integrate_to_end(&chen, &ic, 0.001, 1000)
        .expect("run a");
    let result_b = rk4b
        .integrate_to_end(&chen, &ic, 0.001, 1000)
        .expect("run b");

    for i in 0..3 {
        assert!(
            (result_a[i] - result_b[i]).abs() < 1e-15,
            "non-deterministic at component {i}"
        );
    }
}

/// Verify the Jacobian is consistent with a numerical finite-difference approximation.
#[test]
fn chen_jacobian_vs_finite_difference() {
    let chen = ChenSystem::default_paper();
    let state = [5.0, -3.0, 15.0];
    let mut jac_analytic = [0.0; 9];
    chen.jacobian(&state, &mut jac_analytic)
        .expect("analytic jacobian");

    let h = 1e-7;
    let mut jac_numerical = [0.0; 9];
    let mut f_plus = [0.0; 3];
    let mut f_minus = [0.0; 3];

    for j in 0..3 {
        let mut state_plus = state;
        let mut state_minus = state;
        state_plus[j] += h;
        state_minus[j] -= h;

        chen.derivative(&state_plus, &mut f_plus).expect("f_plus");
        chen.derivative(&state_minus, &mut f_minus)
            .expect("f_minus");

        for i in 0..3 {
            jac_numerical[i * 3 + j] = (f_plus[i] - f_minus[i]) / (2.0 * h);
        }
    }

    for k in 0..9 {
        assert!(
            (jac_analytic[k] - jac_numerical[k]).abs() < 1e-5,
            "jacobian mismatch at [{},{}]: analytic={}, numerical={}",
            k / 3,
            k % 3,
            jac_analytic[k],
            jac_numerical[k]
        );
    }
}
