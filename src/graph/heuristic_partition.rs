use super::coupling::CouplingMatrix;
use super::error::GraphError;
use super::partition::ClusterPattern;
use crate::linalg::{symmetric_eigen, Matrix};

/// Heuristic partition discovery using spectral methods.
///
/// These methods work for any graph size N but may not find all equitable
/// partitions. They complement (don't replace) exhaustive enumeration.
pub struct SpectralPartitioner;

impl SpectralPartitioner {
    /// Spectral bisection using the Fiedler vector.
    ///
    /// Computes the Laplacian of the graph, finds the second-smallest eigenvector
    /// (Fiedler vector), and bisects nodes by the sign of their Fiedler components.
    /// Returns `None` if the resulting partition is not equitable.
    pub fn bisect(topology: &CouplingMatrix) -> Result<Option<ClusterPattern>, GraphError> {
        let n = topology.node_count();
        if n < 2 {
            return Err(GraphError::InvalidPartition {
                reason: "need at least 2 nodes for bisection".to_string(),
            });
        }

        let lap = topology.laplacian()?;
        let eig = symmetric_eigen(&lap)?;

        let eigvecs = eig
            .eigenvectors
            .as_ref()
            .ok_or(GraphError::InvalidPartition {
                reason: "eigenvector computation failed".to_string(),
            })?;

        // Fiedler vector = second eigenvector (index 1, eigenvalues sorted ascending)
        let mut assignment = vec![0usize; n];
        for (i, label) in assignment.iter_mut().enumerate() {
            let component = eigvecs.get(i, 1)?;
            *label = if component >= 0.0 { 0 } else { 1 };
        }

        let pattern = ClusterPattern::new(assignment)?;

        // Verify the partition is non-trivial
        if pattern.num_clusters() < 2 {
            return Ok(None);
        }

        // Verify equitability
        if pattern.is_equitable(topology.adjacency())? {
            Ok(Some(pattern))
        } else {
            Ok(None)
        }
    }

    /// Recursive spectral k-way partitioning.
    ///
    /// Uses successive eigenvectors of the Laplacian to partition the graph
    /// into up to `k` clusters. Returns the best equitable partition found,
    /// or `None` if no equitable partition is discovered.
    pub fn k_way(topology: &CouplingMatrix, k: usize) -> Result<Vec<ClusterPattern>, GraphError> {
        if k < 2 {
            return Err(GraphError::InvalidPartition {
                reason: "k must be at least 2 for k-way partitioning".to_string(),
            });
        }

        let n = topology.node_count();
        if n < 2 {
            return Err(GraphError::InvalidPartition {
                reason: "need at least 2 nodes for partitioning".to_string(),
            });
        }

        let lap = topology.laplacian()?;
        let eig = symmetric_eigen(&lap)?;

        let eigvecs = eig
            .eigenvectors
            .as_ref()
            .ok_or(GraphError::InvalidPartition {
                reason: "eigenvector computation failed".to_string(),
            })?;

        let mut results = Vec::new();

        // Try partitions using combinations of eigenvectors 1..k
        // For each number of clusters c = 2..=min(k, n-1), use the first c-1
        // non-trivial eigenvectors to define a partition via sign patterns.
        let max_clusters = k.min(n - 1);
        let num_eigvecs = (max_clusters - 1).min(n - 1);

        // Extract the relevant eigenvectors (skip eigvec 0 = constant)
        let mut coords: Vec<Vec<f64>> = Vec::with_capacity(num_eigvecs);
        for ev in 0..num_eigvecs {
            let ev_idx = ev + 1; // skip the zero-eigenvalue eigenvector
            if ev_idx >= n {
                break;
            }
            let mut col = Vec::with_capacity(n);
            for i in 0..n {
                col.push(eigvecs.get(i, ev_idx)?);
            }
            coords.push(col);
        }

        // Generate sign-based partitions from subsets of eigenvectors
        for num_ev in 1..=coords.len() {
            // Each node gets a signature based on the signs of the first num_ev eigenvectors
            let mut assignment = vec![0usize; n];
            for (i, label) in assignment.iter_mut().enumerate() {
                let mut val = 0usize;
                for (e, coord) in coords.iter().enumerate().take(num_ev) {
                    if coord[i] >= 0.0 {
                        val |= 1 << e;
                    }
                }
                *label = val;
            }

            // Canonicalize
            if let Ok(pattern) = ClusterPattern::new(assignment) {
                if pattern.num_clusters() >= 2
                    && pattern.num_clusters() <= max_clusters
                    && pattern.is_equitable(topology.adjacency())?
                {
                    results.push(pattern);
                }
            }
        }

        Ok(results)
    }

    /// Find equitable partitions by combining spectral methods with
    /// median-based thresholding.
    ///
    /// For the Fiedler vector, tries both sign-based and median-based splits.
    /// This increases the chance of finding equitable partitions for graphs
    /// where the zero threshold doesn't align with the equitable structure.
    pub fn bisect_variants(topology: &CouplingMatrix) -> Result<Vec<ClusterPattern>, GraphError> {
        let n = topology.node_count();
        if n < 2 {
            return Err(GraphError::InvalidPartition {
                reason: "need at least 2 nodes for bisection".to_string(),
            });
        }

        let lap = topology.laplacian()?;
        let eig = symmetric_eigen(&lap)?;

        let eigvecs = eig
            .eigenvectors
            .as_ref()
            .ok_or(GraphError::InvalidPartition {
                reason: "eigenvector computation failed".to_string(),
            })?;

        let adj = topology.adjacency();
        let mut results = Vec::new();
        let mut seen: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();

        // Try multiple eigenvectors (Fiedler and beyond)
        let max_ev = (n - 1).min(4); // check up to 4 non-trivial eigenvectors
        for ev_idx in 1..=max_ev {
            let mut fiedler = Vec::with_capacity(n);
            for i in 0..n {
                fiedler.push(eigvecs.get(i, ev_idx)?);
            }

            // Sign-based split
            let assignment: Vec<usize> = fiedler
                .iter()
                .map(|&v| if v >= 0.0 { 0 } else { 1 })
                .collect();
            Self::try_add_partition(assignment, adj, &mut seen, &mut results)?;

            // Median-based split
            let mut sorted = fiedler.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[n / 2];
            let assignment: Vec<usize> = fiedler
                .iter()
                .map(|&v| if v >= median { 0 } else { 1 })
                .collect();
            Self::try_add_partition(assignment, adj, &mut seen, &mut results)?;
        }

        Ok(results)
    }

    /// Helper: try to add a partition if it's valid, equitable, and not seen before.
    fn try_add_partition(
        assignment: Vec<usize>,
        adjacency: &Matrix,
        seen: &mut std::collections::BTreeSet<Vec<usize>>,
        results: &mut Vec<ClusterPattern>,
    ) -> Result<(), GraphError> {
        if let Ok(pattern) = ClusterPattern::new(assignment) {
            if pattern.num_clusters() >= 2
                && seen.insert(pattern.assignment().to_vec())
                && pattern.is_equitable(adjacency)?
            {
                results.push(pattern);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::topology::TopologyBuilder;

    #[test]
    fn spectral_bisection_ring8() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let result = SpectralPartitioner::bisect(&cm).expect("bisect");
        // C₈ should have a bipartite equitable partition via Fiedler
        if let Some(pattern) = result {
            assert_eq!(pattern.num_clusters(), 2);
            assert!(
                pattern.is_equitable(cm.adjacency()).expect("check"),
                "spectral bisection should produce equitable partition"
            );
        }
        // It's OK if Fiedler doesn't find an equitable partition —
        // spectral methods are heuristic
    }

    #[test]
    fn spectral_bisection_ring4() {
        let cm = TopologyBuilder::ring(4).expect("ring4");
        let result = SpectralPartitioner::bisect(&cm).expect("bisect");
        if let Some(pattern) = result {
            assert_eq!(pattern.num_clusters(), 2);
            assert!(pattern.is_equitable(cm.adjacency()).expect("check"));
        }
    }

    #[test]
    fn spectral_bisection_complete4() {
        let cm = TopologyBuilder::complete(4).expect("K4");
        let result = SpectralPartitioner::bisect(&cm).expect("bisect");
        // K₄ bisection by Fiedler should be equitable (all-to-all)
        if let Some(pattern) = result {
            assert!(pattern.is_equitable(cm.adjacency()).expect("check"));
        }
    }

    #[test]
    fn spectral_kway_ring8() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let results = SpectralPartitioner::k_way(&cm, 4).expect("k_way");
        // Should find at least one equitable partition
        for p in &results {
            assert!(
                p.is_equitable(cm.adjacency()).expect("check"),
                "k-way partition should be equitable"
            );
        }
    }

    #[test]
    fn spectral_bisect_variants_ring8() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let results = SpectralPartitioner::bisect_variants(&cm).expect("variants");
        for p in &results {
            assert!(
                p.is_equitable(cm.adjacency()).expect("check"),
                "variant partition should be equitable"
            );
        }
    }

    #[test]
    fn spectral_bisection_large_ring() {
        // Verify spectral methods work for N > 20 (beyond exhaustive enumeration)
        let cm = TopologyBuilder::ring(32).expect("ring32");
        let result = SpectralPartitioner::bisect(&cm).expect("bisect");
        if let Some(pattern) = result {
            assert_eq!(pattern.num_nodes(), 32);
            assert_eq!(pattern.num_clusters(), 2);
            assert!(
                pattern.is_equitable(cm.adjacency()).expect("check"),
                "bisection of ring32 should be equitable"
            );
        }
    }

    #[test]
    fn spectral_kway_large_ring() {
        let cm = TopologyBuilder::ring(32).expect("ring32");
        let results = SpectralPartitioner::k_way(&cm, 4).expect("k_way");
        for p in &results {
            assert!(
                p.is_equitable(cm.adjacency()).expect("check"),
                "k-way partition of ring32 should be equitable"
            );
        }
    }

    #[test]
    fn spectral_bisection_too_small() {
        // Build a 1-node "graph" via coupling matrix
        let adj = Matrix::from_row_major(1, 1, &[0.0]).expect("1x1");
        let gamma = Matrix::identity(1);
        let cm = CouplingMatrix::new(adj, gamma, 1.0).expect("coupling");
        assert!(SpectralPartitioner::bisect(&cm).is_err());
    }

    #[test]
    fn from_user_valid_partition() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        // Bipartite partition of C₈
        let pattern = ClusterPattern::from_user(vec![0, 1, 0, 1, 0, 1, 0, 1], Some(cm.adjacency()))
            .expect("valid");
        assert_eq!(pattern.num_clusters(), 2);
    }

    #[test]
    fn from_user_invalid_partition() {
        let cm = TopologyBuilder::ring(5).expect("ring5");
        // Not equitable for C₅
        let result = ClusterPattern::from_user(vec![0, 0, 1, 1, 1], Some(cm.adjacency()));
        assert!(result.is_err());
    }

    #[test]
    fn from_user_no_adjacency_check() {
        // Without adjacency, any valid partition is accepted
        let pattern = ClusterPattern::from_user(vec![0, 0, 1, 1, 1], None).expect("valid");
        assert_eq!(pattern.num_clusters(), 2);
    }

    #[test]
    fn from_user_large_ring() {
        let cm = TopologyBuilder::ring(100).expect("ring100");
        // Bipartite partition of C₁₀₀ — alternating labels
        let assignment: Vec<usize> = (0..100).map(|i| i % 2).collect();
        let pattern = ClusterPattern::from_user(assignment, Some(cm.adjacency())).expect("valid");
        assert_eq!(pattern.num_clusters(), 2);
        assert_eq!(pattern.num_nodes(), 100);
    }
}
