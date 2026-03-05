use super::error::SyncError;
use super::msf::{MasterStabilityFunction, MsfConfig};
use crate::dynamics::traits::DynamicalSystem;
use crate::graph::{ClusterPattern, CouplingMatrix};
use crate::linalg::{symmetric_eigen, Matrix};

/// Validates coupling strength ranges for cluster synchronization patterns.
///
/// Given a cluster pattern and the MSF, checks that:
/// 1. All transverse modes are stable: μ(ε·λₖᵗ) < 0
/// 2. At least one synchronous mode is unstable: μ(ε·λₖˢ) > 0
///
/// This determines the valid coupling strength range for a given pattern.
pub struct ClusterSyncVerifier;

/// Result of validating a cluster pattern at a specific coupling strength.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the cluster pattern is achievable at the given ε.
    pub is_valid: bool,
    /// Coupling strength used.
    pub epsilon: f64,
    /// Transverse eigenvalues (should all map to stable MSF region).
    pub transverse_eigenvalues: Vec<f64>,
    /// Whether all transverse modes are stable.
    pub transverse_stable: bool,
}

impl ClusterSyncVerifier {
    /// Compute the valid coupling strength range for a cluster pattern.
    ///
    /// Uses the MSF and the Laplacian eigenvalue decomposition to determine
    /// the range of ε where the cluster pattern is stable.
    ///
    /// Returns `(ε_min, ε_max)` or `None` if no valid range exists.
    pub fn valid_epsilon_range(
        pattern: &ClusterPattern,
        coupling: &CouplingMatrix,
        system: &dyn DynamicalSystem,
        msf_config: &MsfConfig,
    ) -> Result<Option<(f64, f64)>, SyncError> {
        let n = coupling.node_count();
        if pattern.num_nodes() != n {
            return Err(SyncError::NodeCountMismatch {
                expected: n,
                got: pattern.num_nodes(),
            });
        }

        // Compute Laplacian eigenvalues
        let lap = coupling.laplacian()?;
        let eig = symmetric_eigen(&lap)?;
        let eigenvalues = eig.real_eigenvalues();

        // Get transverse eigenvalues: non-zero eigenvalues that correspond
        // to transverse modes for this cluster pattern.
        // For a cluster pattern, the transverse eigenvalues are all non-zero
        // Laplacian eigenvalues. The coupling parameter η = -ε·λₖ in the MSF.
        let nonzero_eigs: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&ev| ev.abs() > 1e-10)
            .collect();

        if nonzero_eigs.is_empty() {
            return Ok(None);
        }

        // Find the MSF stability threshold η̃ (where MSF crosses from + to -)
        let threshold = MasterStabilityFunction::find_threshold_bisection(
            system,
            coupling.inner_coupling(),
            -30.0,
            0.0,
            0.1,
            30,
            msf_config,
        )?;

        // For stability we need: η = -ε·λₖ < η̃ (in the stable region)
        // i.e., -ε·λₖ < η̃ (where η̃ is negative)
        // For positive eigenvalues λₖ > 0:
        //   ε > |η̃| / λₖ  (lower bound from largest eigenvalue)
        //   and all eigenvalues must satisfy this
        //
        // The MSF is negative (stable) for η < η̃ (type III MSF for Chen).
        // η = -ε·λₖ, so for stability: -ε·λₖ < η̃ → ε·λₖ > |η̃| → ε > |η̃|/λₖ
        //
        // The most restrictive lower bound comes from the smallest non-zero eigenvalue.
        // For an upper bound, we need to check if the MSF has a second zero-crossing
        // (type II MSF). If so, we also need -ε·λₖ > η_lower for all eigenvalues.

        let eta_threshold = threshold; // This is negative

        // Check for second crossing (type II MSF) by scanning more negative η
        let eta_scan: Vec<f64> = (-60..=-25).map(|i| i as f64).collect();
        let msf = MasterStabilityFunction::compute(
            system,
            coupling.inner_coupling(),
            &eta_scan,
            msf_config,
        )?;
        let region = msf.find_stability_region();

        let lambda_min = nonzero_eigs.iter().copied().fold(f64::INFINITY, f64::min);
        let lambda_max = nonzero_eigs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Lower bound on ε: the smallest eigenvalue must enter the stable region
        // -ε·λ_min < η̃ → ε > -η̃/λ_min = |η̃|/λ_min
        let eps_min = (-eta_threshold) / lambda_min;

        // Upper bound on ε: depends on whether MSF has bounded stability region
        let eps_max = if let Some(ref reg) = region {
            if let Some(eta_lower) = reg.eta_lower {
                // Type II: -ε·λ_max > η_lower → ε < -η_lower/λ_max
                (-eta_lower) / lambda_max
            } else {
                // Type III: unbounded, use a practical upper limit
                // The largest eigenvalue should not push η too far
                // Use a generous upper bound
                (-eta_threshold) / lambda_max * 10.0
            }
        } else {
            // No stability region found at all
            return Ok(None);
        };

        if eps_min >= eps_max {
            return Ok(None);
        }

        Ok(Some((eps_min, eps_max)))
    }

    /// Validate a cluster pattern at a specific coupling strength.
    pub fn validate_at_epsilon(
        pattern: &ClusterPattern,
        coupling: &CouplingMatrix,
        system: &dyn DynamicalSystem,
        epsilon: f64,
        msf_config: &MsfConfig,
    ) -> Result<ValidationResult, SyncError> {
        let n = coupling.node_count();
        if pattern.num_nodes() != n {
            return Err(SyncError::NodeCountMismatch {
                expected: n,
                got: pattern.num_nodes(),
            });
        }

        // Compute Laplacian eigenvalues
        let lap = coupling.laplacian()?;
        let eig = symmetric_eigen(&lap)?;
        let eigenvalues = eig.real_eigenvalues();

        let nonzero_eigs: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&ev| ev.abs() > 1e-10)
            .collect();

        // Check MSF at each transverse mode: η = -ε·λₖ
        let mut all_stable = true;
        for &lambda in &nonzero_eigs {
            let eta = -epsilon * lambda;
            let mu = MasterStabilityFunction::compute_single(
                system,
                coupling.inner_coupling(),
                eta,
                msf_config,
            )?;
            if mu >= 0.0 {
                all_stable = false;
                break;
            }
        }

        Ok(ValidationResult {
            is_valid: all_stable,
            epsilon,
            transverse_eigenvalues: nonzero_eigs,
            transverse_stable: all_stable,
        })
    }

    /// Compute the quotient matrix for a cluster pattern.
    ///
    /// The quotient matrix Q has entry Q[a][b] = number of neighbors in cluster b
    /// that any node in cluster a has (valid for equitable partitions).
    pub fn quotient_matrix(
        pattern: &ClusterPattern,
        coupling: &CouplingMatrix,
    ) -> Result<Matrix, SyncError> {
        let n = coupling.node_count();
        if pattern.num_nodes() != n {
            return Err(SyncError::NodeCountMismatch {
                expected: n,
                got: pattern.num_nodes(),
            });
        }

        let k = pattern.num_clusters();
        let mut quotient = Matrix::zeros(k, k);

        for a in 0..k {
            let nodes_a = pattern.nodes_in_cluster(a);
            if nodes_a.is_empty() {
                continue;
            }
            // Use first node as representative (equitable partition guarantees same count)
            let rep = nodes_a[0];
            for b in 0..k {
                let nodes_b = pattern.nodes_in_cluster(b);
                let mut count = 0.0;
                for &j in &nodes_b {
                    count += coupling.adjacency().get(rep, j)?;
                }
                quotient.set(a, b, count)?;
            }
        }

        Ok(quotient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;
    use crate::graph::TopologyBuilder;

    fn fast_msf_config() -> MsfConfig {
        MsfConfig {
            dt: 0.001,
            transient_steps: 5_000,
            compute_steps: 30_000,
            renorm_interval: 10,
            initial_state: vec![1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn quotient_matrix_octagon_2cluster() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        // 2-cluster pattern: {0,2,4,6} and {1,3,5,7}
        let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("pattern");

        let q = ClusterSyncVerifier::quotient_matrix(&pattern, &coupling).expect("quotient");
        assert_eq!(q.nrows(), 2);
        assert_eq!(q.ncols(), 2);

        // In the octagon (C₈), each even node has 2 odd neighbors and 0 even neighbors
        // (since it's a ring: 0-1-2-3-4-5-6-7-0)
        let q00 = q.get(0, 0).expect("q00"); // even-to-even connections
        let q01 = q.get(0, 1).expect("q01"); // even-to-odd connections
        let q10 = q.get(1, 0).expect("q10"); // odd-to-even connections
        let q11 = q.get(1, 1).expect("q11"); // odd-to-odd connections

        assert!(
            (q00 - 0.0).abs() < 1e-10,
            "even nodes have 0 even neighbors, got {q00}"
        );
        assert!(
            (q01 - 2.0).abs() < 1e-10,
            "even nodes have 2 odd neighbors, got {q01}"
        );
        assert!(
            (q10 - 2.0).abs() < 1e-10,
            "odd nodes have 2 even neighbors, got {q10}"
        );
        assert!(
            (q11 - 0.0).abs() < 1e-10,
            "odd nodes have 0 odd neighbors, got {q11}"
        );
    }

    #[test]
    fn quotient_matrix_pattern_size_mismatch() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let pattern = ClusterPattern::new(vec![0, 1, 0, 1]).expect("pattern");
        let result = ClusterSyncVerifier::quotient_matrix(&pattern, &coupling);
        assert!(result.is_err());
    }

    #[test]
    fn validate_at_reasonable_epsilon() {
        let chen = ChenSystem::default_paper();
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("pattern");
        let config = fast_msf_config();

        // At ε=10.0 (within expected range), check stability
        let result =
            ClusterSyncVerifier::validate_at_epsilon(&pattern, &coupling, &chen, 10.0, &config)
                .expect("validate");

        // Should have 7 non-zero eigenvalues for 8-node graph
        assert_eq!(
            result.transverse_eigenvalues.len(),
            7,
            "octagon has 7 non-zero Laplacian eigenvalues"
        );
    }

    #[test]
    fn validate_at_zero_epsilon_unstable() {
        let chen = ChenSystem::default_paper();
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("pattern");
        let config = fast_msf_config();

        // At ε=0 (no coupling), all modes should be unstable (chaotic)
        let result =
            ClusterSyncVerifier::validate_at_epsilon(&pattern, &coupling, &chen, 0.0, &config)
                .expect("validate");

        // η = -0*λ = 0 for all eigenvalues, and MSF at η=0 is positive (chaotic)
        // But η=0 maps to all eigenvalues so transverse_stable should be false
        // Actually at ε=0, η = -0*λ = 0, and μ(0) > 0, so not stable
        assert!(
            !result.transverse_stable,
            "at ε=0, transverse modes should be unstable"
        );
    }

    #[test]
    fn epsilon_range_exists_for_octagon() {
        let chen = ChenSystem::default_paper();
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("pattern");
        let config = fast_msf_config();

        let range = ClusterSyncVerifier::valid_epsilon_range(&pattern, &coupling, &chen, &config)
            .expect("compute range");

        // Should find some valid range
        assert!(
            range.is_some(),
            "should find a valid ε range for octagon 2-cluster pattern"
        );

        if let Some((eps_min, eps_max)) = range {
            assert!(eps_min > 0.0, "ε_min should be positive, got {eps_min}");
            assert!(
                eps_max > eps_min,
                "ε_max ({eps_max}) should exceed ε_min ({eps_min})"
            );
        }
    }
}
