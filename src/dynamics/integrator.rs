use super::error::DynamicsError;
use super::traits::DynamicalSystem;

/// Default time step for integration (verified stable for Chen system).
pub const DEFAULT_DT: f64 = 0.001;

/// 4th-order Runge-Kutta integrator for autonomous ODE systems.
///
/// Pre-allocates scratch buffers to avoid per-step allocations.
pub struct Rk4 {
    /// Scratch buffers for the four RK4 stages.
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    /// Scratch buffer for intermediate state evaluation.
    scratch: Vec<f64>,
}

impl Rk4 {
    /// Create a new RK4 integrator sized for a system of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            k1: vec![0.0; dimension],
            k2: vec![0.0; dimension],
            k3: vec![0.0; dimension],
            k4: vec![0.0; dimension],
            scratch: vec![0.0; dimension],
        }
    }

    /// Advance `state` by one RK4 step of size `dt`.
    ///
    /// Mutates `state` in-place. Returns an error if dimensions don't match.
    pub fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut [f64],
        dt: f64,
    ) -> Result<(), DynamicsError> {
        let dim = system.dimension();
        if state.len() != dim {
            return Err(DynamicsError::DimensionMismatch {
                expected: dim,
                got: state.len(),
            });
        }

        // k1 = f(x)
        system.derivative(state, &mut self.k1)?;

        // k2 = f(x + dt/2 * k1)
        for (scratch_val, (state_val, k_val)) in self
            .scratch
            .iter_mut()
            .zip(state.iter().zip(self.k1.iter()))
        {
            *scratch_val = state_val + 0.5 * dt * k_val;
        }
        system.derivative(&self.scratch, &mut self.k2)?;

        // k3 = f(x + dt/2 * k2)
        for (scratch_val, (state_val, k_val)) in self
            .scratch
            .iter_mut()
            .zip(state.iter().zip(self.k2.iter()))
        {
            *scratch_val = state_val + 0.5 * dt * k_val;
        }
        system.derivative(&self.scratch, &mut self.k3)?;

        // k4 = f(x + dt * k3)
        for (scratch_val, (state_val, k_val)) in self
            .scratch
            .iter_mut()
            .zip(state.iter().zip(self.k3.iter()))
        {
            *scratch_val = state_val + dt * k_val;
        }
        system.derivative(&self.scratch, &mut self.k4)?;

        // x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for (state_val, (((k1, k2), k3), k4)) in state.iter_mut().zip(
            self.k1
                .iter()
                .zip(self.k2.iter())
                .zip(self.k3.iter())
                .zip(self.k4.iter()),
        ) {
            *state_val += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        Ok(())
    }

    /// Integrate the system for `steps` time steps, returning the full trajectory.
    ///
    /// Returns a vector of length `steps + 1` (including the initial state).
    /// Each entry is a clone of the state vector at that time.
    pub fn integrate(
        &mut self,
        system: &dyn DynamicalSystem,
        initial_state: &[f64],
        dt: f64,
        steps: usize,
    ) -> Result<Vec<Vec<f64>>, DynamicsError> {
        let dim = system.dimension();
        if initial_state.len() != dim {
            return Err(DynamicsError::DimensionMismatch {
                expected: dim,
                got: initial_state.len(),
            });
        }

        let mut trajectory = Vec::with_capacity(steps + 1);
        let mut state = initial_state.to_vec();
        trajectory.push(state.clone());

        for _ in 0..steps {
            self.step(system, &mut state, dt)?;

            // Check for numerical blowup
            if state.iter().any(|x| !x.is_finite()) {
                return Err(DynamicsError::IntegrationFailed {
                    reason: "state diverged to infinity or NaN".to_string(),
                });
            }

            trajectory.push(state.clone());
        }

        Ok(trajectory)
    }

    /// Integrate the system for `steps` time steps, returning only the final state.
    ///
    /// More memory-efficient than `integrate` when the trajectory is not needed.
    pub fn integrate_to_end(
        &mut self,
        system: &dyn DynamicalSystem,
        initial_state: &[f64],
        dt: f64,
        steps: usize,
    ) -> Result<Vec<f64>, DynamicsError> {
        let dim = system.dimension();
        if initial_state.len() != dim {
            return Err(DynamicsError::DimensionMismatch {
                expected: dim,
                got: initial_state.len(),
            });
        }

        let mut state = initial_state.to_vec();
        for step in 0..steps {
            self.step(system, &mut state, dt)?;

            if state.iter().any(|x| !x.is_finite()) {
                return Err(DynamicsError::IntegrationFailed {
                    reason: format!("state diverged at step {step}"),
                });
            }
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;

    #[test]
    fn rk4_dimension_mismatch() {
        let chen = ChenSystem::default_paper();
        let mut rk4 = Rk4::new(3);
        let mut state = [1.0, 2.0]; // wrong dimension
        assert!(rk4.step(&chen, &mut state, 0.001).is_err());
    }

    #[test]
    fn rk4_single_step_produces_finite_output() {
        let chen = ChenSystem::default_paper();
        let mut rk4 = Rk4::new(3);
        let mut state = [1.0, 1.0, 1.0];
        rk4.step(&chen, &mut state, DEFAULT_DT)
            .expect("single step");
        assert!(state.iter().all(|x| x.is_finite()));
    }

    /// Verify RK4 convergence order: error ratio between dt and dt/2
    /// should be approximately 2^4 = 16 for a 4th-order method.
    #[test]
    fn rk4_convergence_order() {
        let chen = ChenSystem::default_paper();
        let dt_coarse = 0.01;
        let dt_fine = dt_coarse / 2.0;
        let dt_finest = dt_coarse / 4.0;
        let steps_coarse = 100;
        let steps_fine = steps_coarse * 2;
        let steps_finest = steps_coarse * 4;

        let initial = [1.0, 1.0, 1.0];

        let mut rk4_c = Rk4::new(3);
        let mut rk4_f = Rk4::new(3);
        let mut rk4_ff = Rk4::new(3);

        let coarse = rk4_c
            .integrate_to_end(&chen, &initial, dt_coarse, steps_coarse)
            .expect("coarse");
        let fine = rk4_f
            .integrate_to_end(&chen, &initial, dt_fine, steps_fine)
            .expect("fine");
        let finest = rk4_ff
            .integrate_to_end(&chen, &initial, dt_finest, steps_finest)
            .expect("finest");

        // Error of coarse vs reference (finest)
        let err_coarse: f64 = coarse
            .iter()
            .zip(finest.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Error of fine vs reference (finest)
        let err_fine: f64 = fine
            .iter()
            .zip(finest.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // For RK4, halving dt should reduce error by ~16x
        // We accept a generous range due to chaotic sensitivity
        let ratio = err_coarse / err_fine;
        assert!(
            ratio > 8.0 && ratio < 32.0,
            "convergence ratio {ratio} outside expected range [8, 32] for RK4"
        );
    }

    #[test]
    fn chen_attractor_bounded() {
        let chen = ChenSystem::default_paper();
        let mut rk4 = Rk4::new(3);
        let trajectory = rk4
            .integrate(&chen, &[1.0, 1.0, 1.0], DEFAULT_DT, 10_000)
            .expect("10k steps");

        // Chen attractor is bounded; typical range is |x| < 50 for each component
        let bound = 100.0;
        for (i, state) in trajectory.iter().enumerate() {
            for (j, &val) in state.iter().enumerate() {
                assert!(
                    val.abs() < bound,
                    "state[{i}][{j}] = {val} exceeds bound {bound}"
                );
            }
        }
    }

    #[test]
    fn integrate_returns_correct_length() {
        let chen = ChenSystem::default_paper();
        let mut rk4 = Rk4::new(3);
        let trajectory = rk4
            .integrate(&chen, &[1.0, 1.0, 1.0], DEFAULT_DT, 100)
            .expect("integrate");
        assert_eq!(trajectory.len(), 101); // initial + 100 steps
    }

    #[test]
    fn integrate_to_end_matches_integrate_last() {
        let chen = ChenSystem::default_paper();
        let initial = [1.0, 2.0, 3.0];

        let mut rk4a = Rk4::new(3);
        let trajectory = rk4a
            .integrate(&chen, &initial, DEFAULT_DT, 500)
            .expect("integrate");

        let mut rk4b = Rk4::new(3);
        let final_state = rk4b
            .integrate_to_end(&chen, &initial, DEFAULT_DT, 500)
            .expect("integrate_to_end");

        let last = trajectory.last().expect("non-empty");
        for i in 0..3 {
            assert!(
                (last[i] - final_state[i]).abs() < 1e-12,
                "mismatch at component {i}: {} vs {}",
                last[i],
                final_state[i]
            );
        }
    }
}
