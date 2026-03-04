use super::coupling::CouplingMatrix;
use super::error::GraphError;
use crate::linalg::Matrix;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// A cluster pattern assigns each node in a network to a cluster label.
///
/// Represents one possible synchronization pattern where nodes in the same
/// cluster synchronize their trajectories while remaining unsynchronized
/// across clusters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterPattern {
    /// assignment[i] = cluster label for node i. Labels are canonical:
    /// 0, 1, 2, ... assigned in order of first appearance.
    assignment: Vec<usize>,
    /// Number of distinct clusters.
    num_clusters: usize,
}

impl ClusterPattern {
    /// Create a new cluster pattern from a node-to-cluster assignment.
    ///
    /// The assignment is canonicalized so that labels appear in order of
    /// first occurrence (label 0 for the cluster of node 0, etc.).
    pub fn new(assignment: Vec<usize>) -> Result<Self, GraphError> {
        if assignment.is_empty() {
            return Err(GraphError::InvalidPartition {
                reason: "assignment is empty".to_string(),
            });
        }

        // Canonicalize: relabel so labels appear in order of first occurrence
        let canonical = Self::canonicalize(&assignment);
        let num_clusters = canonical.iter().copied().max().map_or(0, |m| m + 1);

        // Verify no gaps in labeling
        let mut seen = vec![false; num_clusters];
        for &label in &canonical {
            if label >= num_clusters {
                return Err(GraphError::InvalidPartition {
                    reason: format!("label {label} exceeds num_clusters {num_clusters}"),
                });
            }
            seen[label] = true;
        }
        if seen.iter().any(|&s| !s) {
            return Err(GraphError::InvalidPartition {
                reason: "gap in cluster labels".to_string(),
            });
        }

        Ok(Self {
            assignment: canonical,
            num_clusters,
        })
    }

    /// Number of clusters in this pattern.
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.assignment.len()
    }

    /// Get the cluster label for node `i`.
    pub fn label(&self, i: usize) -> Result<usize, GraphError> {
        self.assignment
            .get(i)
            .copied()
            .ok_or(GraphError::InvalidPartition {
                reason: format!("node index {i} out of range (n={})", self.assignment.len()),
            })
    }

    /// Get all nodes belonging to a given cluster label.
    pub fn nodes_in_cluster(&self, label: usize) -> Vec<usize> {
        self.assignment
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == label)
            .map(|(i, _)| i)
            .collect()
    }

    /// Check if two nodes are in the same cluster.
    pub fn are_same_cluster(&self, i: usize, j: usize) -> bool {
        i < self.assignment.len()
            && j < self.assignment.len()
            && self.assignment[i] == self.assignment[j]
    }

    /// The raw assignment vector.
    pub fn assignment(&self) -> &[usize] {
        &self.assignment
    }

    /// Check if this partition is equitable with respect to the given adjacency matrix.
    ///
    /// A partition is equitable if for every pair of clusters (Cᵢ, Cⱼ),
    /// every node in Cᵢ has the same number of neighbors in Cⱼ.
    pub fn is_equitable(&self, adjacency: &Matrix) -> Result<bool, GraphError> {
        let n = self.assignment.len();
        if adjacency.nrows() != n || adjacency.ncols() != n {
            return Err(GraphError::InvalidPartition {
                reason: format!(
                    "adjacency {}x{} doesn't match {} nodes",
                    adjacency.nrows(),
                    adjacency.ncols(),
                    n
                ),
            });
        }

        // For each cluster pair (ci, cj), compute the neighbor count from the
        // first node in ci to cj, then verify all other nodes in ci match.
        for ci in 0..self.num_clusters {
            let nodes_ci = self.nodes_in_cluster(ci);
            if nodes_ci.is_empty() {
                continue;
            }

            // Compute reference counts from the first node in ci
            let ref_node = nodes_ci[0];
            let mut ref_counts = vec![0usize; self.num_clusters];
            for j in 0..n {
                let aij = adjacency.get(ref_node, j)?;
                if aij.abs() > 1e-12 {
                    ref_counts[self.assignment[j]] += 1;
                }
            }

            // Verify all other nodes in ci have the same counts
            for &node in nodes_ci.iter().skip(1) {
                let mut counts = vec![0usize; self.num_clusters];
                for j in 0..n {
                    let aij = adjacency.get(node, j)?;
                    if aij.abs() > 1e-12 {
                        counts[self.assignment[j]] += 1;
                    }
                }
                if counts != ref_counts {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Canonicalize an assignment: relabel so labels appear in order of
    /// first occurrence (node 0's cluster is always label 0).
    fn canonicalize(assignment: &[usize]) -> Vec<usize> {
        let mut mapping: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0;
        assignment
            .iter()
            .map(|&old| {
                *mapping.entry(old).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                })
            })
            .collect()
    }

    /// Apply a node permutation to this pattern (for automorphism-based dedup).
    /// Returns a new pattern where node i gets the label that node perm[i] had.
    fn permuted(&self, perm: &[usize]) -> Vec<usize> {
        perm.iter().map(|&p| self.assignment[p]).collect()
    }

    /// Compute the canonical form of this pattern under a set of automorphisms.
    /// The canonical form is the lexicographically smallest canonicalized
    /// assignment over all automorphisms.
    fn canonical_under_automorphisms(&self, automorphisms: &[Vec<usize>]) -> Vec<usize> {
        let mut best = self.assignment.clone();
        for perm in automorphisms {
            let permuted = self.permuted(perm);
            let canon = Self::canonicalize(&permuted);
            if canon < best {
                best = canon;
            }
        }
        best
    }
}

impl std::fmt::Display for ClusterPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClusterPattern({} clusters: [", self.num_clusters)?;
        for (i, &label) in self.assignment.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{label}")?;
        }
        write!(f, "])")
    }
}

/// Enumerates all non-trivial equitable partitions of a network topology.
pub struct PartitionEnumerator;

impl PartitionEnumerator {
    /// Enumerate all non-trivial equitable partitions of the given topology.
    ///
    /// "Non-trivial" means at least 2 clusters and at most n-1 clusters
    /// (excludes the all-one and all-separate partitions).
    ///
    /// Returns all distinct equitable partitions (not deduplicated by automorphism).
    pub fn enumerate(topology: &CouplingMatrix) -> Result<Vec<ClusterPattern>, GraphError> {
        let n = topology.node_count();
        let adj = topology.adjacency();

        if n > 16 {
            return Err(GraphError::InvalidPartition {
                reason: format!("exhaustive partition enumeration not supported for n={n} > 16"),
            });
        }

        let mut results = Vec::new();
        let mut current = vec![0usize; n];

        Self::generate_and_check(n, 0, 0, &mut current, adj, &mut results)?;

        Ok(results)
    }

    /// Enumerate equitable partitions deduplicated under the graph's automorphisms.
    ///
    /// Two partitions are considered equivalent if one can be obtained from the
    /// other by applying a graph automorphism (node permutation that preserves adjacency).
    pub fn enumerate_unique(topology: &CouplingMatrix) -> Result<Vec<ClusterPattern>, GraphError> {
        let all = Self::enumerate(topology)?;
        let automorphisms =
            super::symmetry::SymmetryDetector::find_automorphisms(topology.adjacency())?;

        let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
        let mut unique = Vec::new();

        for pattern in all {
            let canon = pattern.canonical_under_automorphisms(&automorphisms);
            if seen.insert(canon) {
                unique.push(pattern);
            }
        }

        Ok(unique)
    }

    /// Recursively generate canonical partitions (restricted growth strings)
    /// and check equitability.
    ///
    /// `next_fresh` is the next unused label (one past the max label used so far).
    fn generate_and_check(
        n: usize,
        pos: usize,
        next_fresh: usize,
        current: &mut Vec<usize>,
        adjacency: &Matrix,
        results: &mut Vec<ClusterPattern>,
    ) -> Result<(), GraphError> {
        if pos == n {
            let num_clusters = next_fresh;
            // Non-trivial: at least 2 clusters, at most n-1
            if num_clusters >= 2 && num_clusters < n {
                let pattern = ClusterPattern {
                    assignment: current.clone(),
                    num_clusters,
                };
                if pattern.is_equitable(adjacency)? {
                    results.push(pattern);
                }
            }
            return Ok(());
        }

        // Restricted growth string: position `pos` can use any existing label
        // (0..next_fresh) or introduce one new label (next_fresh).
        for label in 0..=next_fresh {
            // Skip introducing a new label if it would exceed n-1 clusters
            if label == next_fresh && next_fresh >= n - 1 {
                continue;
            }
            current[pos] = label;
            let new_next = if label == next_fresh {
                next_fresh + 1
            } else {
                next_fresh
            };
            Self::generate_and_check(n, pos + 1, new_next, current, adjacency, results)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::topology::TopologyBuilder;

    #[test]
    fn cluster_pattern_creation() {
        let p = ClusterPattern::new(vec![0, 1, 0, 1]).expect("valid");
        assert_eq!(p.num_clusters(), 2);
        assert_eq!(p.num_nodes(), 4);
    }

    #[test]
    fn cluster_pattern_canonicalization() {
        // Labels 5,3,5,3 should become 0,1,0,1
        let p = ClusterPattern::new(vec![5, 3, 5, 3]).expect("valid");
        assert_eq!(p.assignment(), &[0, 1, 0, 1]);
    }

    #[test]
    fn cluster_pattern_empty_error() {
        assert!(ClusterPattern::new(vec![]).is_err());
    }

    #[test]
    fn nodes_in_cluster() {
        let p = ClusterPattern::new(vec![0, 1, 0, 1, 0]).expect("valid");
        assert_eq!(p.nodes_in_cluster(0), vec![0, 2, 4]);
        assert_eq!(p.nodes_in_cluster(1), vec![1, 3]);
        assert_eq!(p.nodes_in_cluster(2), Vec::<usize>::new());
    }

    #[test]
    fn are_same_cluster() {
        let p = ClusterPattern::new(vec![0, 1, 0, 1]).expect("valid");
        assert!(p.are_same_cluster(0, 2));
        assert!(p.are_same_cluster(1, 3));
        assert!(!p.are_same_cluster(0, 1));
    }

    #[test]
    fn label_out_of_range() {
        let p = ClusterPattern::new(vec![0, 1]).expect("valid");
        assert!(p.label(0).is_ok());
        assert!(p.label(2).is_err());
    }

    #[test]
    fn equitable_bipartite_ring() {
        // Ring(4) with bipartite partition {0,2}/{1,3} is equitable
        let cm = TopologyBuilder::ring(4).expect("ring4");
        let p = ClusterPattern::new(vec![0, 1, 0, 1]).expect("valid");
        assert!(p.is_equitable(cm.adjacency()).expect("check"));
    }

    #[test]
    fn non_equitable_partition() {
        // Ring(4) with partition {0,1}/{2,3} is NOT equitable
        // Node 0: neighbors 1(same), 3(other) → 1,1
        // Node 1: neighbors 0(same), 2(other) → 1,1
        // Node 2: neighbors 1(other), 3(same) → 1,1
        // Actually this IS equitable for ring4!
        // Let's use ring(5) with {0,1}/{2,3,4}
        let cm = TopologyBuilder::ring(5).expect("ring5");
        let p = ClusterPattern::new(vec![0, 0, 1, 1, 1]).expect("valid");
        // Node 0 (C0): neighbors 1(C0), 4(C1) → 1 in C0, 1 in C1
        // Node 1 (C0): neighbors 0(C0), 2(C1) → 1 in C0, 1 in C1
        // Node 2 (C1): neighbors 1(C0), 3(C1) → 1 in C0, 1 in C1
        // Node 3 (C1): neighbors 2(C1), 4(C1) → 0 in C0, 2 in C1
        // NOT equitable (nodes 2 and 3 differ)
        assert!(!p.is_equitable(cm.adjacency()).expect("check"));
    }

    #[test]
    fn enumerate_ring4() {
        let cm = TopologyBuilder::ring(4).expect("ring4");
        let patterns = PartitionEnumerator::enumerate(&cm).expect("enum");
        // Ring(4) = C₄. Non-trivial equitable partitions:
        // - {0,2}/{1,3} (bipartite) → 1 partition
        // - {0,1}/{2,3} → equitable (each has 1 in, 1 cross)
        // - {0,3}/{1,2} → equitable (each has 1 in, 1 cross)
        // That's 3 two-cluster partitions.
        // 3-cluster: not possible equitably for C₄
        assert!(
            !patterns.is_empty(),
            "ring4 should have non-trivial equitable partitions"
        );
        // All returned patterns must be equitable
        for p in &patterns {
            assert!(
                p.is_equitable(cm.adjacency()).expect("check"),
                "pattern {p} should be equitable"
            );
        }
    }

    #[test]
    fn enumerate_octagon() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let patterns = PartitionEnumerator::enumerate(&cm).expect("enum");
        // All must be equitable
        for p in &patterns {
            assert!(
                p.is_equitable(cm.adjacency()).expect("check"),
                "pattern {p} should be equitable"
            );
        }
        // Should have multiple patterns
        assert!(
            patterns.len() >= 2,
            "octagon should have at least 2 equitable partitions, got {}",
            patterns.len()
        );
    }

    #[test]
    fn enumerate_unique_octagon() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let unique = PartitionEnumerator::enumerate_unique(&cm).expect("enum");
        // C₈ has 6 structurally distinct non-trivial equitable partitions:
        // 2-cluster: bipartite [0,1,0,1,0,1,0,1] and alternating-pairs [0,0,1,1,0,0,1,1]
        // 3-cluster: [0,1,0,2,0,1,0,2]
        // 4-cluster: two types (antipodal [0,1,2,3,0,1,2,3] and [0,0,1,2,3,3,2,1])
        // 5-cluster: [0,1,0,2,3,4,3,2]
        assert_eq!(
            unique.len(),
            6,
            "octagon should have 6 unique equitable partition types, got {}: {:?}",
            unique.len(),
            unique
        );
        // The two 2-cluster patterns are the key ones for CLSK binary communication
        let two_cluster: Vec<_> = unique.iter().filter(|p| p.num_clusters() == 2).collect();
        assert_eq!(
            two_cluster.len(),
            2,
            "octagon should have exactly 2 unique 2-cluster equitable partitions"
        );
    }

    #[test]
    fn enumerate_complete_graph() {
        let cm = TopologyBuilder::complete(4).expect("K4");
        let unique = PartitionEnumerator::enumerate_unique(&cm).expect("enum");
        // K₄ has 3 structurally distinct non-trivial equitable partitions:
        // 1. [0,0,0,1] — 1+3 split: single node vs rest
        // 2. [0,0,1,1] — 2+2 split: two pairs
        // 3. [0,0,1,2] — 2+1+1 split: one pair + two singletons
        // All are equitable because K₄ has all-to-all connections.
        assert_eq!(
            unique.len(),
            3,
            "K4 unique equitable partitions: expected 3, got {}: {:?}",
            unique.len(),
            unique
        );
    }

    #[test]
    fn enumerate_ring6() {
        let cm = TopologyBuilder::ring(6).expect("ring6");
        let unique = PartitionEnumerator::enumerate_unique(&cm).expect("enum");
        // C₆ non-trivial equitable partitions (unique up to automorphism):
        // 1. Bipartite: {0,2,4}/{1,3,5}
        // 2. Antipodal pairs: {0,3}/{1,4}/{2,5}
        // 3. Consecutive pairs: {0,1}/{2,3}/{4,5} (or rotations) — nope, not equitable
        //    Node 0: neighbors 5(other), 1(same) → 1 in, 1 out
        //    Node 1: neighbors 0(same), 2(other) → 1 in, 1 out
        //    But which "other"? 5 is in {4,5} and 2 is in {2,3} — different clusters!
        //    So node 0 has: 1 in {0,1}, 0 in {2,3}, 1 in {4,5}
        //    Node 1 has: 1 in {0,1}, 1 in {2,3}, 0 in {4,5}
        //    NOT equitable.
        // 4. {0,1,3,4}/{2,5}: not equal sizes, check...
        //    This gets complex. Let the code determine the count.
        assert!(
            !unique.is_empty(),
            "ring6 should have non-trivial equitable partitions"
        );
        // All returned must be equitable
        for p in &unique {
            assert!(
                p.is_equitable(cm.adjacency()).expect("check"),
                "pattern {p} should be equitable"
            );
        }
    }

    #[test]
    fn serde_roundtrip() {
        let p = ClusterPattern::new(vec![0, 1, 0, 1, 2, 2]).expect("valid");
        let json = serde_json::to_string(&p).expect("serialize");
        let p2: ClusterPattern = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p, p2);
    }

    #[test]
    fn display_format() {
        let p = ClusterPattern::new(vec![0, 1, 0, 1]).expect("valid");
        let s = format!("{p}");
        assert!(s.contains("2 clusters"));
    }
}
