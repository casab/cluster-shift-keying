use super::error::SyncError;
use crate::dynamics::integrator::Rk4;
use crate::dynamics::traits::DynamicalSystem;
use crate::linalg::Matrix;

/// Represents the stability region where the MSF is negative: μ(η) < 0
/// for η in the interval [eta_low, eta_high].
///
/// If `eta_high` is `None`, the region extends to -∞ (unbounded, type III MSF).
#[derive(Debug, Clone)]
pub struct StabilityRegion {
    /// Upper boundary (closer to zero) where μ(η) crosses zero.
    /// For the Chen system this is η̃ ≈ -10.3.
    pub eta_upper: f64,
    /// Lower boundary (more negative) where μ(η) crosses zero again.
    /// `None` if the stability region extends to -∞.
    pub eta_lower: Option<f64>,
}

impl StabilityRegion {
    /// Check if a given η value lies within the stability region.
    pub fn contains(&self, eta: f64) -> bool {
        let above_lower = match self.eta_lower {
            Some(lo) => eta >= lo,
            None => true,
        };
        above_lower && eta <= self.eta_upper
    }
}

/// Result of evaluating the MSF at a single η value.
#[derive(Debug, Clone)]
pub struct MsfPoint {
    /// The coupling parameter η.
    pub eta: f64,
    /// The maximum Lyapunov exponent μ(η).
    pub lyapunov_exponent: f64,
}

/// Computes the Master Stability Function for a dynamical system.
///
/// The MSF maps a scalar coupling parameter η to the maximum Lyapunov exponent
/// μ(η) of the variational equation:
///
///   δẋ = [Df(s(t)) + η·Γ] δx
///
/// where s(t) is a trajectory on the chaotic attractor, Df is the Jacobian,
/// and Γ is the inner coupling matrix.
pub struct MasterStabilityFunction {
    /// Cached MSF evaluations (sorted by η).
    curve: Vec<MsfPoint>,
}

/// Configuration for MSF computation.
pub struct MsfConfig {
    /// Time step for integration.
    pub dt: f64,
    /// Number of transient steps to discard before computing the LE.
    pub transient_steps: usize,
    /// Number of steps over which to compute the Lyapunov exponent.
    pub compute_steps: usize,
    /// Renormalization interval (number of steps between renormalizations).
    pub renorm_interval: usize,
    /// Initial state for the chaotic system (on or near the attractor).
    pub initial_state: Vec<f64>,
}

impl Default for MsfConfig {
    fn default() -> Self {
        Self {
            dt: 0.001,
            transient_steps: 10_000,
            compute_steps: 100_000,
            renorm_interval: 10,
            initial_state: vec![1.0, 1.0, 1.0],
        }
    }
}

impl MasterStabilityFunction {
    /// Compute the MSF for a system over a range of η values.
    ///
    /// Generates the reference trajectory once, then evaluates μ(η) at each
    /// point in `eta_values`.
    pub fn compute(
        system: &dyn DynamicalSystem,
        inner_coupling: &Matrix,
        eta_values: &[f64],
        config: &MsfConfig,
    ) -> Result<Self, SyncError> {
        let dim = system.dimension();
        if inner_coupling.nrows() != dim || inner_coupling.ncols() != dim {
            return Err(SyncError::MsfFailed {
                reason: format!(
                    "inner coupling matrix {}x{} doesn't match system dimension {dim}",
                    inner_coupling.nrows(),
                    inner_coupling.ncols()
                ),
            });
        }

        // Generate reference trajectory (after transient removal)
        let trajectory = Self::generate_trajectory(system, config)?;

        // Evaluate MSF at each η value
        let mut curve = Vec::with_capacity(eta_values.len());
        for &eta in eta_values {
            let le = Self::max_lyapunov_exponent(system, inner_coupling, eta, &trajectory, config)?;
            curve.push(MsfPoint {
                eta,
                lyapunov_exponent: le,
            });
        }

        // Sort by η
        curve.sort_by(|a, b| {
            a.eta
                .partial_cmp(&b.eta)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Self { curve })
    }

    /// Evaluate the MSF at a single η value.
    pub fn compute_single(
        system: &dyn DynamicalSystem,
        inner_coupling: &Matrix,
        eta: f64,
        config: &MsfConfig,
    ) -> Result<f64, SyncError> {
        let dim = system.dimension();
        if inner_coupling.nrows() != dim || inner_coupling.ncols() != dim {
            return Err(SyncError::MsfFailed {
                reason: format!(
                    "inner coupling matrix {}x{} doesn't match system dimension {dim}",
                    inner_coupling.nrows(),
                    inner_coupling.ncols()
                ),
            });
        }

        let trajectory = Self::generate_trajectory(system, config)?;
        Self::max_lyapunov_exponent(system, inner_coupling, eta, &trajectory, config)
    }

    /// Get the cached MSF curve.
    pub fn curve(&self) -> &[MsfPoint] {
        &self.curve
    }

    /// Find the stability region from the computed MSF curve.
    ///
    /// Looks for zero-crossings of μ(η) to identify where the MSF is negative.
    /// Returns `None` if no stability region is found.
    pub fn find_stability_region(&self) -> Option<StabilityRegion> {
        if self.curve.len() < 2 {
            return None;
        }

        // Walk from most negative η to most positive, looking for sign changes.
        // The curve is sorted by η ascending.
        let mut eta_upper: Option<f64> = None;
        let mut eta_lower: Option<f64> = None;

        for i in 0..self.curve.len() - 1 {
            let p1 = &self.curve[i];
            let p2 = &self.curve[i + 1];

            if p1.lyapunov_exponent * p2.lyapunov_exponent < 0.0 {
                // Zero crossing — interpolate
                let frac = p1.lyapunov_exponent.abs()
                    / (p1.lyapunov_exponent.abs() + p2.lyapunov_exponent.abs());
                let eta_cross = p1.eta + frac * (p2.eta - p1.eta);

                if p1.lyapunov_exponent > 0.0 && p2.lyapunov_exponent < 0.0 {
                    // Crossing from positive to negative (entering stability)
                    // Since η is ascending, this is the upper boundary
                    // (higher η = closer to 0 for negative η range)
                    // Actually: if we're scanning from negative to positive η,
                    // entering stability means eta_lower; exiting means eta_upper
                    if eta_lower.is_none() {
                        eta_lower = Some(eta_cross);
                    }
                } else if p1.lyapunov_exponent < 0.0 && p2.lyapunov_exponent > 0.0 {
                    // Crossing from negative to positive (exiting stability)
                    eta_upper = Some(eta_cross);
                }
            }
        }

        // If the MSF is negative at the leftmost point, the lower bound is unbounded
        if self.curve[0].lyapunov_exponent < 0.0 {
            eta_lower = None;
            // If no upper crossing found, the whole scanned range is stable
            if eta_upper.is_none() {
                // Find the rightmost point that's still negative
                if let Some(last_neg) = self.curve.iter().rev().find(|p| p.lyapunov_exponent < 0.0)
                {
                    eta_upper = Some(last_neg.eta);
                }
            }
        }

        eta_upper.map(|eu| StabilityRegion {
            eta_upper: eu,
            eta_lower,
        })
    }

    /// Find the stability boundary η̃ more precisely using bisection.
    ///
    /// Searches for the zero-crossing of μ(η) between `eta_low` and `eta_high`,
    /// where `eta_low` should be in the stable region (μ < 0) and `eta_high`
    /// in the unstable region (μ > 0).
    pub fn find_threshold_bisection(
        system: &dyn DynamicalSystem,
        inner_coupling: &Matrix,
        eta_low: f64,
        eta_high: f64,
        tolerance: f64,
        max_iterations: usize,
        config: &MsfConfig,
    ) -> Result<f64, SyncError> {
        let dim = system.dimension();
        if inner_coupling.nrows() != dim || inner_coupling.ncols() != dim {
            return Err(SyncError::MsfFailed {
                reason: format!(
                    "inner coupling matrix {}x{} doesn't match system dimension {dim}",
                    inner_coupling.nrows(),
                    inner_coupling.ncols()
                ),
            });
        }

        let trajectory = Self::generate_trajectory(system, config)?;

        let mut lo = eta_low;
        let mut hi = eta_high;
        let mu_lo = Self::max_lyapunov_exponent(system, inner_coupling, lo, &trajectory, config)?;
        let mu_hi = Self::max_lyapunov_exponent(system, inner_coupling, hi, &trajectory, config)?;

        if mu_lo * mu_hi > 0.0 {
            return Err(SyncError::MsfFailed {
                reason: format!(
                    "bisection requires sign change: μ({lo}) = {mu_lo}, μ({hi}) = {mu_hi}"
                ),
            });
        }

        for _ in 0..max_iterations {
            if (hi - lo).abs() < tolerance {
                break;
            }
            let mid = (lo + hi) / 2.0;
            let mu_mid =
                Self::max_lyapunov_exponent(system, inner_coupling, mid, &trajectory, config)?;

            if mu_mid * mu_lo < 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        Ok((lo + hi) / 2.0)
    }

    /// Generate a reference trajectory on the attractor.
    fn generate_trajectory(
        system: &dyn DynamicalSystem,
        config: &MsfConfig,
    ) -> Result<Vec<Vec<f64>>, SyncError> {
        let dim = system.dimension();
        if config.initial_state.len() != dim {
            return Err(SyncError::MsfFailed {
                reason: format!(
                    "initial state dimension {} doesn't match system dimension {dim}",
                    config.initial_state.len()
                ),
            });
        }

        let mut rk4 = Rk4::new(dim);

        // Remove transient
        let state_after_transient = rk4.integrate_to_end(
            system,
            &config.initial_state,
            config.dt,
            config.transient_steps,
        )?;

        // Generate trajectory for LE computation
        let trajectory = rk4.integrate(
            system,
            &state_after_transient,
            config.dt,
            config.compute_steps,
        )?;

        Ok(trajectory)
    }

    /// Compute the maximum Lyapunov exponent using the Benettin algorithm.
    ///
    /// Evolves a perturbation vector δx along the precomputed trajectory
    /// using the variational equation δẋ = [Df(s(t)) + η·Γ] δx.
    fn max_lyapunov_exponent(
        system: &dyn DynamicalSystem,
        inner_coupling: &Matrix,
        eta: f64,
        trajectory: &[Vec<f64>],
        config: &MsfConfig,
    ) -> Result<f64, SyncError> {
        let dim = system.dimension();

        // Pre-extract inner coupling values for the hot loop
        let gamma = Self::extract_gamma(inner_coupling, dim)?;

        // Initialize perturbation vector (unit vector along first axis)
        let mut delta = vec![0.0; dim];
        delta[0] = 1.0;

        // Scratch buffers for variational RK4
        let mut k1 = vec![0.0; dim];
        let mut k2 = vec![0.0; dim];
        let mut k3 = vec![0.0; dim];
        let mut k4 = vec![0.0; dim];
        let mut tmp_delta = vec![0.0; dim];
        let mut jac = vec![0.0; dim * dim];

        let dt = config.dt;
        let mut sum_log = 0.0;
        let mut renorm_count = 0u64;

        // Step through the trajectory
        let steps = trajectory.len().saturating_sub(1);
        for (step, state) in trajectory.iter().enumerate().take(steps) {
            // Compute Jacobian at current state
            system.jacobian(state, &mut jac)?;

            // Form the effective matrix: J_eff = Df(s) + η·Γ
            // We don't form it explicitly; instead, multiply during RK4

            // RK4 step for the variational equation
            // k1 = J_eff * delta
            Self::mat_vec_variational(&jac, &gamma, eta, &delta, &mut k1, dim);

            // k2 = J_eff * (delta + dt/2 * k1)
            for i in 0..dim {
                tmp_delta[i] = delta[i] + 0.5 * dt * k1[i];
            }
            Self::mat_vec_variational(&jac, &gamma, eta, &tmp_delta, &mut k2, dim);

            // k3 = J_eff * (delta + dt/2 * k2)
            for i in 0..dim {
                tmp_delta[i] = delta[i] + 0.5 * dt * k2[i];
            }
            Self::mat_vec_variational(&jac, &gamma, eta, &tmp_delta, &mut k3, dim);

            // k4 = J_eff * (delta + dt * k3)
            for i in 0..dim {
                tmp_delta[i] = delta[i] + dt * k3[i];
            }
            Self::mat_vec_variational(&jac, &gamma, eta, &tmp_delta, &mut k4, dim);

            // Update delta
            for i in 0..dim {
                delta[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }

            // Periodic renormalization
            if (step + 1) % config.renorm_interval == 0 {
                let norm: f64 = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-300 || !norm.is_finite() {
                    return Err(SyncError::MsfFailed {
                        reason: format!("perturbation norm is {norm} at step {step}"),
                    });
                }
                sum_log += norm.ln();
                renorm_count += 1;
                for d in &mut delta {
                    *d /= norm;
                }
            }
        }

        if renorm_count == 0 {
            return Err(SyncError::MsfFailed {
                reason: "no renormalization steps performed".to_string(),
            });
        }

        // μ(η) = average of log(norm) per renormalization interval
        let total_time = (config.renorm_interval as f64) * dt * (renorm_count as f64);
        Ok(sum_log / total_time)
    }

    /// Compute J_eff * v = (Df + η·Γ) * v without forming the full matrix.
    fn mat_vec_variational(
        jac: &[f64],
        gamma: &[f64],
        eta: f64,
        v: &[f64],
        out: &mut [f64],
        dim: usize,
    ) {
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                let j_eff = jac[i * dim + j] + eta * gamma[i * dim + j];
                sum += j_eff * v[j];
            }
            out[i] = sum;
        }
    }

    /// Extract inner coupling matrix into a flat row-major array.
    fn extract_gamma(inner_coupling: &Matrix, dim: usize) -> Result<Vec<f64>, SyncError> {
        let mut gamma = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                gamma[i * dim + j] =
                    inner_coupling.get(i, j).map_err(|e| SyncError::MsfFailed {
                        reason: format!("failed to read inner coupling: {e}"),
                    })?;
            }
        }
        Ok(gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;

    fn default_gamma() -> Matrix {
        // Γ = diag(0, 1, 0) — coupling through the second state variable
        Matrix::from_diagonal(&[0.0, 1.0, 0.0])
    }

    fn fast_config() -> MsfConfig {
        // Faster config for unit tests (shorter trajectory)
        MsfConfig {
            dt: 0.001,
            transient_steps: 5_000,
            compute_steps: 50_000,
            renorm_interval: 10,
            initial_state: vec![1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn msf_positive_at_zero() {
        // At η = 0 (uncoupled), the max LE should be positive (chaotic system)
        let chen = ChenSystem::default_paper();
        let gamma = default_gamma();
        let config = fast_config();

        let mu = MasterStabilityFunction::compute_single(&chen, &gamma, 0.0, &config)
            .expect("compute MSF at η=0");

        assert!(
            mu > 0.0,
            "MSF at η=0 should be positive (chaotic), got {mu}"
        );
    }

    #[test]
    fn msf_negative_at_strong_coupling() {
        // For sufficiently negative η, the coupling stabilizes → μ < 0
        let chen = ChenSystem::default_paper();
        let gamma = default_gamma();
        let config = fast_config();

        let mu = MasterStabilityFunction::compute_single(&chen, &gamma, -20.0, &config)
            .expect("compute MSF at η=-20");

        assert!(
            mu < 0.0,
            "MSF at η=-20 should be negative (stable sync), got {mu}"
        );
    }

    #[test]
    fn msf_threshold_approximation() {
        // The MSF transitions from positive (chaotic) to negative (stable)
        // as η decreases. We verify a sign change occurs in [-10, 0].
        let chen = ChenSystem::default_paper();
        let gamma = default_gamma();
        let config = fast_config();

        let mu_minus10 = MasterStabilityFunction::compute_single(&chen, &gamma, -10.0, &config)
            .expect("MSF at -10");
        let mu_minus2 = MasterStabilityFunction::compute_single(&chen, &gamma, -2.0, &config)
            .expect("MSF at -2");

        // μ(-10) should be negative (deep in stable region)
        assert!(
            mu_minus10 < 0.0,
            "μ(-10) should be negative, got {mu_minus10}"
        );
        // μ(-2) should be positive (still chaotic)
        assert!(mu_minus2 > 0.0, "μ(-2) should be positive, got {mu_minus2}");
    }

    #[test]
    fn msf_curve_computation() {
        let chen = ChenSystem::default_paper();
        let gamma = default_gamma();
        let config = fast_config();

        let eta_values: Vec<f64> = (-20..=0).map(|i| i as f64).collect();
        let msf = MasterStabilityFunction::compute(&chen, &gamma, &eta_values, &config)
            .expect("compute MSF curve");

        let curve = msf.curve();
        assert_eq!(curve.len(), eta_values.len());

        // Curve should be sorted by η
        for i in 1..curve.len() {
            assert!(curve[i].eta >= curve[i - 1].eta);
        }

        // First point (η=-20) should be negative, last (η=0) should be positive
        assert!(
            curve[0].lyapunov_exponent < 0.0,
            "μ(-20) = {} should be negative",
            curve[0].lyapunov_exponent
        );
        assert!(
            curve.last().map_or(false, |p| p.lyapunov_exponent > 0.0),
            "μ(0) should be positive"
        );
    }

    #[test]
    fn stability_region_detection() {
        let chen = ChenSystem::default_paper();
        let gamma = default_gamma();
        let config = fast_config();

        let eta_values: Vec<f64> = (-25..=5).map(|i| i as f64).collect();
        let msf = MasterStabilityFunction::compute(&chen, &gamma, &eta_values, &config)
            .expect("compute MSF curve");

        let region = msf
            .find_stability_region()
            .expect("should find stability region");

        // The upper boundary (where MSF crosses from negative to positive)
        // should be in the range [-8, -2] for Chen with Γ = diag(0,1,0)
        assert!(
            region.eta_upper > -8.0 && region.eta_upper < -2.0,
            "stability boundary η̃ = {} should be in [-8, -2]",
            region.eta_upper
        );
    }

    #[test]
    fn stability_region_contains() {
        let region = StabilityRegion {
            eta_upper: -10.0,
            eta_lower: Some(-50.0),
        };
        assert!(region.contains(-20.0));
        assert!(region.contains(-10.0));
        assert!(region.contains(-50.0));
        assert!(!region.contains(-5.0));
        assert!(!region.contains(-51.0));
        assert!(!region.contains(0.0));

        let unbounded = StabilityRegion {
            eta_upper: -10.0,
            eta_lower: None,
        };
        assert!(unbounded.contains(-100.0));
        assert!(unbounded.contains(-10.0));
        assert!(!unbounded.contains(-5.0));
    }

    #[test]
    fn msf_dimension_mismatch() {
        let chen = ChenSystem::default_paper();
        let wrong_gamma = Matrix::identity(5); // wrong size
        let config = fast_config();

        let result = MasterStabilityFunction::compute_single(&chen, &wrong_gamma, 0.0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn bisection_finds_threshold() {
        let chen = ChenSystem::default_paper();
        let gamma = default_gamma();
        let config = fast_config();

        // Bisect between stable region (η=-20) and unstable region (η=0)
        let threshold = MasterStabilityFunction::find_threshold_bisection(
            &chen, &gamma, -20.0, 0.0, 0.5, 20, &config,
        )
        .expect("bisection");

        // Should be in the range [-8, -2] for Chen with Γ = diag(0,1,0)
        assert!(
            threshold > -8.0 && threshold < -2.0,
            "threshold {threshold} should be in [-8, -2]"
        );
    }
}
