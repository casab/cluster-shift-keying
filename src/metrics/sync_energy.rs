use super::error::MetricsError;
use crate::graph::ClusterPattern;

/// Pairwise synchronization energy matrix.
///
/// Stores the energy E_ij = ∫ ||x_i(t) - x_j(t)||² dt for each node pair,
/// computed via trapezoidal numerical integration over trajectory data.
/// The matrix is symmetric: E_ij = E_ji, and E_ii = 0.
#[derive(Debug, Clone)]
pub struct SyncEnergyMatrix {
    /// Upper-triangular storage: entry (i,j) with i < j at index i*n - i*(i+1)/2 + j - i - 1.
    energies: Vec<f64>,
    /// Number of nodes.
    n: usize,
}

impl SyncEnergyMatrix {
    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Get the synchronization energy between nodes i and j.
    pub fn energy(&self, i: usize, j: usize) -> Result<f64, MetricsError> {
        if i >= self.n || j >= self.n {
            return Err(MetricsError::SimulationFailed {
                reason: format!("node index out of range: ({i},{j}), n={}", self.n),
            });
        }
        if i == j {
            return Ok(0.0);
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let idx = lo * self.n - lo * (lo + 1) / 2 + hi - lo - 1;
        Ok(self.energies[idx])
    }

    /// Get all pairwise energies as a flat vector (upper triangular, row-major).
    pub fn energies_flat(&self) -> &[f64] {
        &self.energies
    }

    /// Compute the mean energy across all pairs.
    pub fn mean_energy(&self) -> f64 {
        if self.energies.is_empty() {
            return 0.0;
        }
        self.energies.iter().sum::<f64>() / self.energies.len() as f64
    }

    /// Compute the auto-threshold γ = mean(all E_ij) as in the paper.
    pub fn auto_threshold(&self) -> f64 {
        self.mean_energy()
    }

    /// Convert to a binary synchronization matrix using a threshold.
    ///
    /// Pairs with E_ij < threshold are considered synchronized (1),
    /// pairs with E_ij >= threshold are not synchronized (0).
    pub fn to_binary(&self, threshold: f64) -> Result<BinarySyncMatrix, MetricsError> {
        if threshold <= 0.0 || !threshold.is_finite() {
            return Err(MetricsError::InvalidThreshold { value: threshold });
        }
        let synced: Vec<bool> = self.energies.iter().map(|&e| e < threshold).collect();
        Ok(BinarySyncMatrix {
            synced,
            n: self.n,
            threshold,
        })
    }

    /// Convert to a binary synchronization matrix using the auto-threshold.
    pub fn to_binary_auto(&self) -> Result<BinarySyncMatrix, MetricsError> {
        let threshold = self.auto_threshold();
        if threshold <= 0.0 || !threshold.is_finite() {
            return Err(MetricsError::InvalidThreshold { value: threshold });
        }
        self.to_binary(threshold)
    }

    /// Compute the mean intra-cluster energy for a given pattern.
    pub fn mean_intra_cluster_energy(&self, pattern: &ClusterPattern) -> Result<f64, MetricsError> {
        if pattern.num_nodes() != self.n {
            return Err(MetricsError::LengthMismatch {
                tx_len: self.n,
                rx_len: pattern.num_nodes(),
            });
        }
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if pattern.are_same_cluster(i, j) {
                    sum += self.energy(i, j)?;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Ok(0.0);
        }
        Ok(sum / count as f64)
    }

    /// Compute the mean inter-cluster energy for a given pattern.
    pub fn mean_inter_cluster_energy(&self, pattern: &ClusterPattern) -> Result<f64, MetricsError> {
        if pattern.num_nodes() != self.n {
            return Err(MetricsError::LengthMismatch {
                tx_len: self.n,
                rx_len: pattern.num_nodes(),
            });
        }
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if !pattern.are_same_cluster(i, j) {
                    sum += self.energy(i, j)?;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Ok(0.0);
        }
        Ok(sum / count as f64)
    }
}

/// Binary synchronization matrix after thresholding.
///
/// A[i][j] = 1 (true) if nodes i and j are synchronized (E_ij < threshold),
/// A[i][j] = 0 (false) otherwise.
#[derive(Debug, Clone)]
pub struct BinarySyncMatrix {
    /// Upper-triangular synchronized flags.
    synced: Vec<bool>,
    /// Number of nodes.
    n: usize,
    /// Threshold used.
    threshold: f64,
}

impl BinarySyncMatrix {
    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Get the threshold used.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Check if nodes i and j are synchronized.
    pub fn is_synchronized(&self, i: usize, j: usize) -> Result<bool, MetricsError> {
        if i >= self.n || j >= self.n {
            return Err(MetricsError::SimulationFailed {
                reason: format!("node index out of range: ({i},{j}), n={}", self.n),
            });
        }
        if i == j {
            return Ok(true);
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let idx = lo * self.n - lo * (lo + 1) / 2 + hi - lo - 1;
        Ok(self.synced[idx])
    }

    /// Extract the cluster pattern implied by the binary sync matrix.
    ///
    /// Uses union-find to group synchronized nodes.
    pub fn to_pattern(&self) -> Result<ClusterPattern, MetricsError> {
        let mut parent: Vec<usize> = (0..self.n).collect();

        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.is_synchronized(i, j)? {
                    let ri = find(&mut parent, i);
                    let rj = find(&mut parent, j);
                    if ri != rj {
                        parent[ri] = rj;
                    }
                }
            }
        }

        let assignment: Vec<usize> = {
            let mut result = Vec::with_capacity(self.n);
            for i in 0..self.n {
                result.push(find(&mut parent, i));
            }
            result
        };

        ClusterPattern::new(assignment).map_err(|e| MetricsError::SimulationFailed {
            reason: format!("failed to create pattern: {e}"),
        })
    }

    /// Check if this binary matrix matches a given cluster pattern.
    pub fn matches_pattern(&self, pattern: &ClusterPattern) -> Result<bool, MetricsError> {
        if pattern.num_nodes() != self.n {
            return Err(MetricsError::LengthMismatch {
                tx_len: self.n,
                rx_len: pattern.num_nodes(),
            });
        }
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let should_sync = pattern.are_same_cluster(i, j);
                let is_sync = self.is_synchronized(i, j)?;
                if should_sync != is_sync {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// Union-find: find root with path halving.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Computes synchronization energy from trajectory data.
///
/// The synchronization energy for a node pair (i, j) over a time interval is:
///
///   E_ij = ∫ ||x_i(t) - x_j(t)||² dt
///
/// computed via the trapezoidal rule over discrete time steps.
pub struct SyncEnergyDetector;

impl SyncEnergyDetector {
    /// Compute the synchronization energy matrix from trajectory data.
    ///
    /// `trajectories` is indexed as `trajectories[time_step][node * dim + d]`,
    /// a flat array of all node states at each time step.
    ///
    /// `n` is the number of nodes, `dim` is the oscillator dimension,
    /// `dt` is the time step between consecutive trajectory samples.
    pub fn compute(
        trajectories: &[Vec<f64>],
        n: usize,
        dim: usize,
        dt: f64,
    ) -> Result<SyncEnergyMatrix, MetricsError> {
        if trajectories.len() < 2 {
            return Err(MetricsError::EmptyInput);
        }
        if n == 0 || dim == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let expected_len = n * dim;
        for (t, traj) in trajectories.iter().enumerate() {
            if traj.len() != expected_len {
                return Err(MetricsError::SimulationFailed {
                    reason: format!(
                        "trajectory[{t}] has length {} but expected {expected_len}",
                        traj.len()
                    ),
                });
            }
        }

        let num_pairs = n * (n - 1) / 2;
        let mut energies = vec![0.0; num_pairs];

        // Trapezoidal rule: E_ij = dt * (f(t0)/2 + f(t1) + ... + f(tN-1) + f(tN)/2)
        let num_steps = trajectories.len();
        for (t, state) in trajectories.iter().enumerate() {
            let weight = if t == 0 || t == num_steps - 1 {
                dt * 0.5
            } else {
                dt
            };

            let mut pair_idx = 0;
            for i in 0..n {
                let oi = i * dim;
                for j in (i + 1)..n {
                    let oj = j * dim;
                    let mut dist_sq = 0.0;
                    for d in 0..dim {
                        let diff = state[oi + d] - state[oj + d];
                        dist_sq += diff * diff;
                    }
                    energies[pair_idx] += weight * dist_sq;
                    pair_idx += 1;
                }
            }
        }

        Ok(SyncEnergyMatrix { energies, n })
    }

    /// Compute sync energy from a CoupledNetwork's trajectory recorded over time.
    ///
    /// `snapshots` is a list of state snapshots, each being a flat vector of
    /// all node states (n * dim elements), taken at uniform dt intervals.
    pub fn from_snapshots(
        snapshots: &[Vec<f64>],
        n: usize,
        dim: usize,
        dt: f64,
    ) -> Result<SyncEnergyMatrix, MetricsError> {
        Self::compute(snapshots, n, dim, dt)
    }

    /// Compute sync energy from per-node trajectory data.
    ///
    /// `node_trajectories[node][time_step]` is a `dim`-element state vector.
    pub fn from_node_trajectories(
        node_trajectories: &[Vec<Vec<f64>>],
        dt: f64,
    ) -> Result<SyncEnergyMatrix, MetricsError> {
        let n = node_trajectories.len();
        if n < 2 {
            return Err(MetricsError::EmptyInput);
        }
        let num_steps = node_trajectories[0].len();
        if num_steps < 2 {
            return Err(MetricsError::EmptyInput);
        }
        let dim = node_trajectories[0][0].len();

        // Convert to flat format
        let mut flat: Vec<Vec<f64>> = Vec::with_capacity(num_steps);
        for t in 0..num_steps {
            let mut state = Vec::with_capacity(n * dim);
            for node in node_trajectories {
                state.extend_from_slice(&node[t]);
            }
            flat.push(state);
        }

        Self::compute(&flat, n, dim, dt)
    }
}

/// Scoring function trait for symbol detection strategies.
///
/// Different scoring functions can be used to map sync energy matrices
/// to symbol decisions (e.g., minimum intra-cluster energy, maximum
/// inter/intra ratio, etc.).
pub trait ScoringFunction {
    /// Score a sync energy matrix against a candidate pattern.
    /// Higher scores indicate better match.
    fn score(
        &self,
        energy: &SyncEnergyMatrix,
        pattern: &ClusterPattern,
    ) -> Result<f64, MetricsError>;
}

/// Default scoring: ratio of inter-cluster to intra-cluster energy.
///
/// score = E_inter / (E_intra + epsilon)
///
/// Higher ratio means better cluster separation → better match.
pub struct RatioScoring {
    /// Small constant to avoid division by zero.
    pub epsilon: f64,
}

impl Default for RatioScoring {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl ScoringFunction for RatioScoring {
    fn score(
        &self,
        energy: &SyncEnergyMatrix,
        pattern: &ClusterPattern,
    ) -> Result<f64, MetricsError> {
        let intra = energy.mean_intra_cluster_energy(pattern)?;
        let inter = energy.mean_inter_cluster_energy(pattern)?;
        Ok(inter / (intra + self.epsilon))
    }
}

/// Scoring based on minimum intra-cluster energy.
///
/// score = -E_intra (negated so higher = better match with lower intra energy)
pub struct MinIntraScoring;

impl ScoringFunction for MinIntraScoring {
    fn score(
        &self,
        energy: &SyncEnergyMatrix,
        pattern: &ClusterPattern,
    ) -> Result<f64, MetricsError> {
        let intra = energy.mean_intra_cluster_energy(pattern)?;
        Ok(-intra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build trajectory data where nodes i and j have identical states.
    fn identical_trajectories(n: usize, dim: usize, steps: usize) -> Vec<Vec<f64>> {
        let mut traj = Vec::with_capacity(steps);
        for t in 0..steps {
            let mut state = Vec::with_capacity(n * dim);
            // All nodes get the same state (time-varying but identical across nodes)
            let base: Vec<f64> = (0..dim)
                .map(|d| (t as f64 * 0.1 + d as f64).sin())
                .collect();
            for _ in 0..n {
                state.extend_from_slice(&base);
            }
            traj.push(state);
        }
        traj
    }

    /// Build trajectory data where each node has a distinct trajectory.
    fn distinct_trajectories(n: usize, dim: usize, steps: usize) -> Vec<Vec<f64>> {
        let mut traj = Vec::with_capacity(steps);
        for t in 0..steps {
            let mut state = Vec::with_capacity(n * dim);
            for node in 0..n {
                for d in 0..dim {
                    // Different phase per node ensures distinct trajectories
                    let val = ((t as f64 * 0.1 + node as f64 * 2.7 + d as f64 * 1.3).sin())
                        * (1.0 + node as f64);
                    state.push(val);
                }
            }
            traj.push(state);
        }
        traj
    }

    /// Build trajectory where nodes 0,1 are synced and nodes 2,3 are synced,
    /// but the two groups differ.
    fn two_cluster_trajectories(steps: usize) -> Vec<Vec<f64>> {
        let dim = 3;
        let mut traj = Vec::with_capacity(steps);
        for t in 0..steps {
            let mut state = Vec::with_capacity(4 * dim);
            // Cluster A: nodes 0, 1
            let a: Vec<f64> = (0..dim)
                .map(|d| (t as f64 * 0.1 + d as f64).sin())
                .collect();
            state.extend_from_slice(&a); // node 0
            state.extend_from_slice(&a); // node 1
                                         // Cluster B: nodes 2, 3
            let b: Vec<f64> = (0..dim)
                .map(|d| (t as f64 * 0.1 + d as f64 + 3.0).cos() * 2.0)
                .collect();
            state.extend_from_slice(&b); // node 2
            state.extend_from_slice(&b); // node 3
            traj.push(state);
        }
        traj
    }

    #[test]
    fn identical_trajectories_zero_energy() {
        let traj = identical_trajectories(4, 3, 100);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        assert_eq!(em.num_nodes(), 4);
        for i in 0..4 {
            for j in 0..4 {
                let e = em.energy(i, j).expect("energy");
                assert!(
                    e.abs() < 1e-10,
                    "identical trajectories should have zero energy, E({i},{j}) = {e}"
                );
            }
        }
        assert!(em.mean_energy().abs() < 1e-10);
    }

    #[test]
    fn distinct_trajectories_nonzero_energy() {
        let traj = distinct_trajectories(4, 3, 100);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        let mean = em.mean_energy();
        assert!(
            mean > 0.01,
            "distinct trajectories should have nonzero energy, mean = {mean}"
        );

        // All pairs should have positive energy
        for i in 0..4 {
            for j in (i + 1)..4 {
                let e = em.energy(i, j).expect("energy");
                assert!(e > 0.0, "E({i},{j}) = {e} should be positive");
            }
        }
    }

    #[test]
    fn two_clusters_intra_vs_inter() {
        let traj = two_cluster_trajectories(200);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        // Pattern: {0,1} and {2,3}
        let pattern = ClusterPattern::new(vec![0, 0, 1, 1]).expect("pattern");

        let intra = em.mean_intra_cluster_energy(&pattern).expect("intra");
        let inter = em.mean_inter_cluster_energy(&pattern).expect("inter");

        // Intra-cluster energy should be ~0 (identical within cluster)
        assert!(
            intra < 1e-10,
            "intra-cluster energy should be ~0, got {intra}"
        );
        // Inter-cluster energy should be large
        assert!(
            inter > 0.01,
            "inter-cluster energy should be large, got {inter}"
        );
    }

    #[test]
    fn threshold_separates_clusters() {
        let traj = two_cluster_trajectories(200);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        let threshold = em.auto_threshold();
        assert!(threshold > 0.0, "auto-threshold should be positive");

        let binary = em.to_binary(threshold).expect("binary");

        // Nodes 0,1 should be synchronized
        assert!(binary.is_synchronized(0, 1).expect("0,1"));
        // Nodes 2,3 should be synchronized
        assert!(binary.is_synchronized(2, 3).expect("2,3"));
        // Cross-cluster should NOT be synchronized
        assert!(!binary.is_synchronized(0, 2).expect("0,2"));
        assert!(!binary.is_synchronized(1, 3).expect("1,3"));
    }

    #[test]
    fn binary_to_pattern() {
        let traj = two_cluster_trajectories(200);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");
        let binary = em.to_binary_auto().expect("binary");
        let pattern = binary.to_pattern().expect("pattern");

        assert_eq!(pattern.num_clusters(), 2);
        assert!(pattern.are_same_cluster(0, 1));
        assert!(pattern.are_same_cluster(2, 3));
        assert!(!pattern.are_same_cluster(0, 2));
    }

    #[test]
    fn binary_matches_pattern() {
        let traj = two_cluster_trajectories(200);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");
        let binary = em.to_binary_auto().expect("binary");

        let correct = ClusterPattern::new(vec![0, 0, 1, 1]).expect("pattern");
        assert!(binary.matches_pattern(&correct).expect("match"));

        let wrong = ClusterPattern::new(vec![0, 1, 0, 1]).expect("pattern");
        assert!(!binary.matches_pattern(&wrong).expect("no match"));
    }

    #[test]
    fn energy_symmetry() {
        let traj = distinct_trajectories(4, 3, 50);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        for i in 0..4 {
            for j in 0..4 {
                let eij = em.energy(i, j).expect("ij");
                let eji = em.energy(j, i).expect("ji");
                assert!(
                    (eij - eji).abs() < 1e-15,
                    "energy should be symmetric: E({i},{j})={eij} vs E({j},{i})={eji}"
                );
            }
        }
    }

    #[test]
    fn self_energy_zero() {
        let traj = distinct_trajectories(4, 3, 50);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        for i in 0..4 {
            assert!(
                em.energy(i, i).expect("self").abs() < 1e-15,
                "self-energy should be 0"
            );
        }
    }

    #[test]
    fn too_few_samples_error() {
        let traj = vec![vec![0.0; 12]]; // only 1 sample
        let result = SyncEnergyDetector::compute(&traj, 4, 3, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_trajectory_length() {
        let traj = vec![vec![0.0; 10], vec![0.0; 12]]; // inconsistent
        let result = SyncEnergyDetector::compute(&traj, 4, 3, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_threshold() {
        let traj = distinct_trajectories(4, 3, 50);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        assert!(em.to_binary(0.0).is_err());
        assert!(em.to_binary(-1.0).is_err());
        assert!(em.to_binary(f64::NAN).is_err());
    }

    #[test]
    fn ratio_scoring() {
        let traj = two_cluster_trajectories(200);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        let scorer = RatioScoring::default();

        // Correct pattern should have high score
        let correct = ClusterPattern::new(vec![0, 0, 1, 1]).expect("correct");
        let score_correct = scorer.score(&em, &correct).expect("score");

        // Wrong pattern should have lower score
        let wrong = ClusterPattern::new(vec![0, 1, 0, 1]).expect("wrong");
        let score_wrong = scorer.score(&em, &wrong).expect("score");

        assert!(
            score_correct > score_wrong,
            "correct pattern score ({score_correct}) should exceed wrong ({score_wrong})"
        );
    }

    #[test]
    fn min_intra_scoring() {
        let traj = two_cluster_trajectories(200);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");

        let scorer = MinIntraScoring;

        let correct = ClusterPattern::new(vec![0, 0, 1, 1]).expect("correct");
        let score_correct = scorer.score(&em, &correct).expect("score");

        let wrong = ClusterPattern::new(vec![0, 1, 0, 1]).expect("wrong");
        let score_wrong = scorer.score(&em, &wrong).expect("score");

        // Correct pattern has ~0 intra energy → score ~0 (higher than negative)
        assert!(
            score_correct > score_wrong,
            "correct pattern score ({score_correct}) should exceed wrong ({score_wrong})"
        );
    }

    #[test]
    fn from_node_trajectories() {
        // 4 nodes, 3 dims, 50 steps
        let mut node_traj: Vec<Vec<Vec<f64>>> = Vec::new();
        for node in 0..4 {
            let mut traj = Vec::new();
            for t in 0..50 {
                let state: Vec<f64> = (0..3)
                    .map(|d| (t as f64 * 0.1 + node as f64 * 2.7 + d as f64).sin())
                    .collect();
                traj.push(state);
            }
            node_traj.push(traj);
        }

        let em = SyncEnergyDetector::from_node_trajectories(&node_traj, 0.01).expect("compute");
        assert_eq!(em.num_nodes(), 4);
        assert!(em.mean_energy() > 0.0);
    }

    #[test]
    fn out_of_range_node() {
        let traj = distinct_trajectories(4, 3, 50);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");
        assert!(em.energy(0, 5).is_err());
    }

    #[test]
    fn pattern_size_mismatch() {
        let traj = distinct_trajectories(4, 3, 50);
        let em = SyncEnergyDetector::compute(&traj, 4, 3, 0.01).expect("compute");
        let pattern = ClusterPattern::new(vec![0, 1, 0]).expect("3 nodes");
        assert!(em.mean_intra_cluster_energy(&pattern).is_err());
    }
}
