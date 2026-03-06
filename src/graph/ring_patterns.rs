use super::coupling::CouplingMatrix;
use super::error::GraphError;
use super::partition::ClusterPattern;
use super::topology::TopologyBuilder;

/// Generate the canonical equitable 2-cluster partition for ring(n).
///
/// For ring(n) with even n, the alternating partition `[0,1,0,1,...]` is always
/// equitable: every node has both neighbors in the opposite cluster. This is the
/// bipartite partition of the cycle graph.
///
/// For M-ary CLSK on rings, all symbols share this single partition and are
/// distinguished by different coupling strengths ε.
pub fn generate_ring_partition(n: usize) -> Result<ClusterPattern, GraphError> {
    if n < 4 {
        return Err(GraphError::InvalidPartition {
            reason: format!("ring must have at least 4 nodes for 2-cluster partitions, got {n}"),
        });
    }
    if !n.is_multiple_of(2) {
        return Err(GraphError::InvalidPartition {
            reason: format!("ring({n}) must have even n for a bipartite 2-cluster partition"),
        });
    }

    let assignment: Vec<usize> = (0..n).map(|i| i % 2).collect();
    ClusterPattern::new(assignment)
}

/// Generate M evenly-spaced coupling strengths (epsilon values).
///
/// Spreads M values across `[eps_min, eps_max]`. The values are spaced to
/// maximize the signal-to-noise ratio for the matched filter detector.
pub fn generate_epsilon_values(
    m: usize,
    eps_min: f64,
    eps_max: f64,
) -> Result<Vec<f64>, GraphError> {
    if m < 2 {
        return Err(GraphError::InvalidPartition {
            reason: format!("need at least 2 epsilon values, got {m}"),
        });
    }
    if eps_min >= eps_max || !eps_min.is_finite() || !eps_max.is_finite() {
        return Err(GraphError::InvalidPartition {
            reason: format!("invalid epsilon range [{eps_min}, {eps_max}]"),
        });
    }

    let step = (eps_max - eps_min) / (m - 1) as f64;
    let values: Vec<f64> = (0..m).map(|i| eps_min + i as f64 * step).collect();
    Ok(values)
}

/// Select channel link nodes satisfying the covertness condition.
///
/// For the alternating partition, nodes 0 (cluster 0) and 1 (cluster 1)
/// are always in different clusters — they are the simplest valid pair.
///
/// Returns `count` node indices suitable as channel links.
pub fn select_channel_links(
    pattern: &ClusterPattern,
    count: usize,
) -> Result<Vec<usize>, GraphError> {
    if count < 1 {
        return Err(GraphError::InvalidPartition {
            reason: "need at least 1 channel link".to_string(),
        });
    }

    let n = pattern.num_nodes();

    // Start with node 0, then find nodes in the other cluster.
    let mut links = vec![0usize];

    for candidate in 1..n {
        if links.len() >= count {
            break;
        }

        // Check that this candidate is in a different cluster from ALL
        // existing links.
        let valid = links
            .iter()
            .all(|&existing| !pattern.are_same_cluster(existing, candidate));

        if valid {
            links.push(candidate);
        }
    }

    if links.len() < count {
        return Err(GraphError::InvalidPartition {
            reason: format!(
                "could only find {} channel links satisfying covertness (need {count})",
                links.len()
            ),
        });
    }

    Ok(links)
}

/// Build a complete M-ary CLSK configuration for ring(n).
///
/// This is the main entry point for setting up multi-bit CLSK on ring graphs.
/// It uses a single equitable partition (alternating bipartite) with
/// M = 2^bits_per_symbol different coupling strengths evenly spaced in
/// `[eps_min, eps_max]`.
///
/// All symbols share the same partition but are distinguished by their
/// coupling strength ε, which produces different synchronization dynamics.
pub fn build_ring_clsk(
    n: usize,
    bits_per_symbol: usize,
    eps_min: f64,
    eps_max: f64,
) -> Result<RingClskConfig, GraphError> {
    let m = 1usize << bits_per_symbol; // 2^bits_per_symbol

    let coupling = TopologyBuilder::ring(n)?;
    let pattern = generate_ring_partition(n)?;
    let epsilons = generate_epsilon_values(m, eps_min, eps_max)?;
    let channel_links = select_channel_links(&pattern, 2)?;

    // All symbols share the same partition, differ only in epsilon.
    let entries: Vec<(usize, ClusterPattern, f64)> = epsilons
        .iter()
        .enumerate()
        .map(|(i, &eps)| (i, pattern.clone(), eps))
        .collect();

    Ok(RingClskConfig {
        coupling,
        entries,
        channel_links,
        bits_per_symbol,
    })
}

/// Configuration produced by `build_ring_clsk`.
pub struct RingClskConfig {
    /// The ring coupling matrix.
    pub coupling: CouplingMatrix,
    /// Symbol entries: (symbol_index, pattern, epsilon).
    pub entries: Vec<(usize, ClusterPattern, f64)>,
    /// Channel link node indices.
    pub channel_links: Vec<usize>,
    /// Bits encoded per symbol (log₂ M).
    pub bits_per_symbol: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_ring8_partition() {
        let pattern = generate_ring_partition(8).expect("ring8 partition");
        assert_eq!(pattern.num_nodes(), 8);
        assert_eq!(pattern.num_clusters(), 2);
        assert_eq!(pattern.assignment(), &[0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn generate_ring8_partition_equitable() {
        let cm = TopologyBuilder::ring(8).expect("ring8");
        let pattern = generate_ring_partition(8).expect("ring8 partition");
        assert!(
            pattern.is_equitable(cm.adjacency()).expect("check"),
            "alternating partition should be equitable for ring8"
        );
    }

    #[test]
    fn generate_ring512_partition() {
        let pattern = generate_ring_partition(512).expect("ring512 partition");
        assert_eq!(pattern.num_nodes(), 512);
        assert_eq!(pattern.num_clusters(), 2);
    }

    #[test]
    fn generate_ring512_partition_equitable() {
        let cm = TopologyBuilder::ring(512).expect("ring512");
        let pattern = generate_ring_partition(512).expect("ring512 partition");
        assert!(
            pattern.is_equitable(cm.adjacency()).expect("check"),
            "alternating partition should be equitable for ring512"
        );
    }

    #[test]
    fn generate_ring_odd_error() {
        assert!(generate_ring_partition(7).is_err());
    }

    #[test]
    fn generate_ring_small_error() {
        assert!(generate_ring_partition(3).is_err());
    }

    #[test]
    fn generate_epsilon_values_basic() {
        let eps = generate_epsilon_values(4, 5.0, 17.0).expect("4 eps");
        assert_eq!(eps.len(), 4);
        assert!((eps[0] - 5.0).abs() < 1e-10);
        assert!((eps[3] - 17.0).abs() < 1e-10);
        // Monotonically increasing
        for i in 1..eps.len() {
            assert!(eps[i] > eps[i - 1]);
        }
    }

    #[test]
    fn generate_epsilon_invalid_range() {
        assert!(generate_epsilon_values(4, 17.0, 5.0).is_err());
        assert!(generate_epsilon_values(1, 5.0, 17.0).is_err());
    }

    #[test]
    fn select_channel_links_basic() {
        let pattern = generate_ring_partition(8).expect("ring8 partition");
        let links = select_channel_links(&pattern, 2).expect("2 links");
        assert_eq!(links.len(), 2);

        // Verify covertness: links are in different clusters
        assert!(
            !pattern.are_same_cluster(links[0], links[1]),
            "links {:?} should be in different clusters",
            links,
        );
    }

    #[test]
    fn select_channel_links_ring512() {
        let pattern = generate_ring_partition(512).expect("ring512 partition");
        let links = select_channel_links(&pattern, 2).expect("2 links");
        assert_eq!(links.len(), 2);
        assert!(!pattern.are_same_cluster(links[0], links[1]));
    }

    #[test]
    fn build_ring_clsk_basic() {
        let config = build_ring_clsk(16, 2, 5.0, 17.0).expect("ring16 2-bit");
        assert_eq!(config.entries.len(), 4); // M=4
        assert_eq!(config.bits_per_symbol, 2);
        assert_eq!(config.channel_links.len(), 2);
        assert_eq!(config.coupling.node_count(), 16);
    }

    #[test]
    fn build_ring_clsk_all_symbols_same_pattern() {
        let config = build_ring_clsk(16, 2, 5.0, 17.0).expect("ring16 2-bit");
        // All entries should share the same pattern
        for i in 1..config.entries.len() {
            assert_eq!(
                config.entries[0].1.assignment(),
                config.entries[i].1.assignment(),
                "all symbols should share the same partition"
            );
        }
    }

    #[test]
    fn build_ring_clsk_distinct_epsilons() {
        let config = build_ring_clsk(16, 2, 5.0, 17.0).expect("ring16 2-bit");
        for i in 1..config.entries.len() {
            assert!(
                (config.entries[i].2 - config.entries[i - 1].2).abs() > 1e-10,
                "epsilons should be distinct"
            );
        }
    }

    #[test]
    fn build_ring512_clsk() {
        let config = build_ring_clsk(512, 2, 5.0, 17.0).expect("ring512 2-bit");
        assert_eq!(config.entries.len(), 4);
        assert_eq!(config.coupling.node_count(), 512);
    }

    #[test]
    fn build_ring512_3bit() {
        let config = build_ring_clsk(512, 3, 5.0, 17.0).expect("ring512 3-bit");
        assert_eq!(config.entries.len(), 8); // M=8
        assert_eq!(config.bits_per_symbol, 3);
    }
}
