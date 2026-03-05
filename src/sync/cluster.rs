use super::error::SyncError;
use super::network::CoupledNetwork;
use crate::graph::ClusterPattern;

/// Runtime state tracking which nodes are currently synchronized.
///
/// Compares pairwise synchronization errors against a threshold to
/// determine the emergent cluster pattern from simulation data.
#[derive(Debug, Clone)]
pub struct ClusterState {
    /// Pairwise synchronization errors (upper triangular, row-major).
    /// Entry (i,j) with i < j is at index i*n - i*(i+1)/2 + j - i - 1.
    errors: Vec<f64>,
    /// Whether each pair is synchronized (error < threshold).
    synchronized: Vec<bool>,
    /// Number of nodes.
    n: usize,
    /// Threshold used for synchronization detection.
    threshold: f64,
}

impl ClusterState {
    /// Detect the current cluster state from a coupled network simulation.
    ///
    /// Two nodes are considered synchronized if their Euclidean state distance
    /// is below `threshold`.
    pub fn from_network(network: &CoupledNetwork, threshold: f64) -> Result<Self, SyncError> {
        let n = network.node_count();
        let num_pairs = n * (n - 1) / 2;
        let mut errors = Vec::with_capacity(num_pairs);
        let mut synchronized = Vec::with_capacity(num_pairs);

        for i in 0..n {
            for j in (i + 1)..n {
                let err = network.sync_error(i, j)?;
                synchronized.push(err < threshold);
                errors.push(err);
            }
        }

        Ok(Self {
            errors,
            synchronized,
            n,
            threshold,
        })
    }

    /// Create a ClusterState from explicit pairwise errors.
    pub fn from_errors(errors: Vec<Vec<f64>>, threshold: f64) -> Result<Self, SyncError> {
        let n = errors.len();
        if n == 0 {
            return Err(SyncError::NodeCountMismatch {
                expected: 1,
                got: 0,
            });
        }

        let num_pairs = n * (n - 1) / 2;
        let mut flat_errors = Vec::with_capacity(num_pairs);
        let mut synchronized = Vec::with_capacity(num_pairs);

        for (i, row) in errors.iter().enumerate() {
            for &err in row.iter().skip(i + 1) {
                synchronized.push(err < threshold);
                flat_errors.push(err);
            }
        }

        Ok(Self {
            errors: flat_errors,
            synchronized,
            n,
            threshold,
        })
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Threshold used for synchronization detection.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Check if nodes `i` and `j` are synchronized.
    pub fn are_synchronized(&self, i: usize, j: usize) -> Result<bool, SyncError> {
        if i >= self.n || j >= self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: i.max(j) + 1,
            });
        }
        if i == j {
            return Ok(true);
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let idx = lo * self.n - lo * (lo + 1) / 2 + hi - lo - 1;
        Ok(self.synchronized[idx])
    }

    /// Get the synchronization error between nodes `i` and `j`.
    pub fn error(&self, i: usize, j: usize) -> Result<f64, SyncError> {
        if i >= self.n || j >= self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: i.max(j) + 1,
            });
        }
        if i == j {
            return Ok(0.0);
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let idx = lo * self.n - lo * (lo + 1) / 2 + hi - lo - 1;
        Ok(self.errors[idx])
    }

    /// Extract the emergent cluster pattern from the synchronization state.
    ///
    /// Uses union-find to group synchronized nodes into clusters.
    pub fn to_pattern(&self) -> Result<ClusterPattern, SyncError> {
        // Union-find
        let mut parent: Vec<usize> = (0..self.n).collect();

        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.are_synchronized(i, j)? {
                    let ri = Self::find(&mut parent, i);
                    let rj = Self::find(&mut parent, j);
                    if ri != rj {
                        parent[ri] = rj;
                    }
                }
            }
        }

        // Flatten
        let assignment: Vec<usize> = {
            let mut result = Vec::with_capacity(self.n);
            for i in 0..self.n {
                result.push(Self::find(&mut parent, i));
            }
            result
        };

        // Convert to ClusterPattern (canonicalizes labels)
        Ok(ClusterPattern::new(assignment)?)
    }

    /// Check if the observed cluster state matches an expected pattern.
    ///
    /// Returns true if the emergent synchronization groups match the pattern.
    pub fn matches_pattern(&self, pattern: &ClusterPattern) -> Result<bool, SyncError> {
        if pattern.num_nodes() != self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: pattern.num_nodes(),
            });
        }

        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let should_sync = pattern.are_same_cluster(i, j);
                let is_sync = self.are_synchronized(i, j)?;
                if should_sync != is_sync {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Compute the mean synchronization error for pairs within the same cluster.
    pub fn mean_intra_cluster_error(&self, pattern: &ClusterPattern) -> Result<f64, SyncError> {
        if pattern.num_nodes() != self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: pattern.num_nodes(),
            });
        }

        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if pattern.are_same_cluster(i, j) {
                    sum += self.error(i, j)?;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        Ok(sum / count as f64)
    }

    /// Compute the mean synchronization error for pairs in different clusters.
    pub fn mean_inter_cluster_error(&self, pattern: &ClusterPattern) -> Result<f64, SyncError> {
        if pattern.num_nodes() != self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: pattern.num_nodes(),
            });
        }

        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if !pattern.are_same_cluster(i, j) {
                    sum += self.error(i, j)?;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        Ok(sum / count as f64)
    }

    /// Union-find: find root with path compression.
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_errors_matrix(n: usize, sync_pairs: &[(usize, usize)]) -> Vec<Vec<f64>> {
        let mut errors = vec![vec![10.0; n]; n]; // default: large error (not synced)
        for i in 0..n {
            errors[i][i] = 0.0;
        }
        for &(i, j) in sync_pairs {
            errors[i][j] = 0.001;
            errors[j][i] = 0.001;
        }
        errors
    }

    #[test]
    fn cluster_state_all_synced() {
        let n = 4;
        let mut sync_pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                sync_pairs.push((i, j));
            }
        }
        let errors = make_errors_matrix(n, &sync_pairs);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");

        for i in 0..n {
            for j in 0..n {
                assert!(
                    cs.are_synchronized(i, j).expect("check"),
                    "({i},{j}) should be synced"
                );
            }
        }

        let pattern = cs.to_pattern().expect("pattern");
        assert_eq!(pattern.num_clusters(), 1);
    }

    #[test]
    fn cluster_state_two_clusters() {
        // Nodes 0,1 synced; nodes 2,3 synced; cross-cluster not synced
        let sync_pairs = vec![(0, 1), (2, 3)];
        let errors = make_errors_matrix(4, &sync_pairs);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");

        assert!(cs.are_synchronized(0, 1).expect("01"));
        assert!(cs.are_synchronized(2, 3).expect("23"));
        assert!(!cs.are_synchronized(0, 2).expect("02"));
        assert!(!cs.are_synchronized(1, 3).expect("13"));

        let pattern = cs.to_pattern().expect("pattern");
        assert_eq!(pattern.num_clusters(), 2);
        assert!(pattern.are_same_cluster(0, 1));
        assert!(pattern.are_same_cluster(2, 3));
        assert!(!pattern.are_same_cluster(0, 2));
    }

    #[test]
    fn matches_pattern_check() {
        let sync_pairs = vec![(0, 2), (1, 3)];
        let errors = make_errors_matrix(4, &sync_pairs);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");

        // This should match: {0,2} and {1,3}
        let matching = ClusterPattern::new(vec![0, 1, 0, 1]).expect("pattern");
        assert!(cs.matches_pattern(&matching).expect("match"));

        // This should NOT match: {0,1} and {2,3}
        let non_matching = ClusterPattern::new(vec![0, 0, 1, 1]).expect("pattern");
        assert!(!cs.matches_pattern(&non_matching).expect("no match"));
    }

    #[test]
    fn intra_inter_cluster_errors() {
        let sync_pairs = vec![(0, 1), (2, 3)];
        let errors = make_errors_matrix(4, &sync_pairs);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");

        let pattern = ClusterPattern::new(vec![0, 0, 1, 1]).expect("pattern");

        let intra = cs.mean_intra_cluster_error(&pattern).expect("intra");
        let inter = cs.mean_inter_cluster_error(&pattern).expect("inter");

        // Intra: (0,1) has error 0.001, (2,3) has error 0.001 → mean = 0.001
        assert!(
            (intra - 0.001).abs() < 1e-10,
            "intra-cluster error = {intra}, expected 0.001"
        );
        // Inter: (0,2), (0,3), (1,2), (1,3) all have error 10.0 → mean = 10.0
        assert!(
            (inter - 10.0).abs() < 1e-10,
            "inter-cluster error = {inter}, expected 10.0"
        );
    }

    #[test]
    fn error_symmetry() {
        let errors = make_errors_matrix(4, &[(0, 2)]);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");

        let e02 = cs.error(0, 2).expect("0,2");
        let e20 = cs.error(2, 0).expect("2,0");
        assert!(
            (e02 - e20).abs() < 1e-15,
            "error should be symmetric: {e02} vs {e20}"
        );
    }

    #[test]
    fn self_error_zero() {
        let errors = make_errors_matrix(4, &[]);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");

        for i in 0..4 {
            assert!(
                cs.error(i, i).expect("self") < 1e-15,
                "self-error should be 0"
            );
        }
    }

    #[test]
    fn out_of_range_error() {
        let errors = make_errors_matrix(4, &[]);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");
        assert!(cs.are_synchronized(0, 5).is_err());
        assert!(cs.error(4, 0).is_err());
    }

    #[test]
    fn pattern_size_mismatch() {
        let errors = make_errors_matrix(4, &[]);
        let cs = ClusterState::from_errors(errors, 1.0).expect("create");
        let wrong_pattern = ClusterPattern::new(vec![0, 1, 0]).expect("pattern");
        assert!(cs.matches_pattern(&wrong_pattern).is_err());
    }
}
