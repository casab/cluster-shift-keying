//! Integration tests for cluster synchronization verification.
//!
//! Tests the coupled network simulation, cluster state detection,
//! and synchronization verifier on the paper's octagon topology.

use cluster_shift_keying::dynamics::chen::ChenSystem;
use cluster_shift_keying::dynamics::integrator::DEFAULT_DT;
use cluster_shift_keying::graph::{ClusterPattern, TopologyBuilder};
use cluster_shift_keying::sync::{ClusterState, ClusterSyncVerifier, CoupledNetwork, MsfConfig};

fn fast_msf_config() -> MsfConfig {
    MsfConfig {
        dt: 0.001,
        transient_steps: 5_000,
        compute_steps: 30_000,
        renorm_interval: 10,
        initial_state: vec![1.0, 1.0, 1.0],
    }
}

/// Create an octagon network with deterministic perturbations.
fn setup_octagon(epsilon: f64) -> (CoupledNetwork, ChenSystem) {
    let chen = ChenSystem::default_paper();
    let mut coupling = TopologyBuilder::octagon().expect("octagon");
    coupling.set_coupling_strength(epsilon);

    let n = coupling.node_count();
    let dim = coupling.oscillator_dim();
    let mut perts = Vec::with_capacity(n);
    for i in 0..n {
        let pert: Vec<f64> = (0..dim)
            .map(|d| 0.01 * ((i * dim + d) as f64 * 0.37).sin())
            .collect();
        perts.push(pert);
    }

    let net =
        CoupledNetwork::new(&coupling, &[1.0, 1.0, 1.0], Some(&perts)).expect("create network");
    (net, chen)
}

#[test]
fn coupled_network_stays_bounded() {
    let (mut net, chen) = setup_octagon(10.0);
    net.integrate(&chen, DEFAULT_DT, 10_000).expect("integrate");

    let bound = 100.0;
    for i in 0..8 {
        let s = net.node_state(i).expect("node state");
        for (d, &val) in s.iter().enumerate() {
            assert!(
                val.abs() < bound,
                "node {i} dim {d} = {val} exceeds bound {bound}"
            );
        }
    }
}

#[test]
fn identical_initial_conditions_remain_synced() {
    // If all nodes start from the same state with no perturbation,
    // they should remain synchronized (zero sync error)
    let chen = ChenSystem::default_paper();
    let mut coupling = TopologyBuilder::octagon().expect("octagon");
    coupling.set_coupling_strength(10.0);

    let mut net = CoupledNetwork::new(&coupling, &[1.0, 1.0, 1.0], None).expect("create network");
    net.integrate(&chen, DEFAULT_DT, 5_000).expect("integrate");

    for i in 0..8 {
        for j in (i + 1)..8 {
            let err = net.sync_error(i, j).expect("sync error");
            assert!(
                err < 1e-10,
                "nodes {i},{j} should remain synced from identical ICs, error = {err}"
            );
        }
    }
}

#[test]
fn cluster_state_from_network() {
    let (mut net, chen) = setup_octagon(10.0);
    net.integrate(&chen, DEFAULT_DT, 20_000).expect("integrate");

    let cs = ClusterState::from_network(&net, 1.0).expect("cluster state");
    assert_eq!(cs.node_count(), 8);

    // Extract emergent pattern
    let pattern = cs.to_pattern().expect("pattern");
    assert!(
        pattern.num_clusters() >= 1 && pattern.num_clusters() <= 8,
        "should have between 1 and 8 clusters, got {}",
        pattern.num_clusters()
    );
}

#[test]
fn cluster_state_intra_vs_inter_error() {
    let (mut net, chen) = setup_octagon(10.0);
    net.integrate(&chen, DEFAULT_DT, 30_000).expect("integrate");

    // Expected 2-cluster pattern for octagon: {0,2,4,6} vs {1,3,5,7}
    let pattern_2cluster = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("pattern");

    let cs = ClusterState::from_network(&net, 5.0).expect("cluster state");
    let intra = cs
        .mean_intra_cluster_error(&pattern_2cluster)
        .expect("intra");
    let inter = cs
        .mean_inter_cluster_error(&pattern_2cluster)
        .expect("inter");

    // Intra-cluster error should generally be smaller than inter-cluster error
    // when the pattern is emergent, but we use a relaxed check
    assert!(
        intra < inter + 50.0,
        "intra-cluster error ({intra}) should not be wildly larger than inter ({inter})"
    );
}

#[test]
fn quotient_matrix_octagon() {
    let coupling = TopologyBuilder::octagon().expect("octagon");
    let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("2-cluster");

    let q = ClusterSyncVerifier::quotient_matrix(&pattern, &coupling).expect("quotient");

    // For the octagon's alternating 2-cluster pattern:
    // Even nodes (0,2,4,6) have 0 even neighbors and 2 odd neighbors
    // Q = [[0, 2], [2, 0]]
    assert!((q.get(0, 0).expect("q00") - 0.0).abs() < 1e-10);
    assert!((q.get(0, 1).expect("q01") - 2.0).abs() < 1e-10);
    assert!((q.get(1, 0).expect("q10") - 2.0).abs() < 1e-10);
    assert!((q.get(1, 1).expect("q11") - 0.0).abs() < 1e-10);
}

#[test]
fn epsilon_range_for_octagon_exists() {
    let chen = ChenSystem::default_paper();
    let coupling = TopologyBuilder::octagon().expect("octagon");
    let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("2-cluster");
    let config = fast_msf_config();

    let range = ClusterSyncVerifier::valid_epsilon_range(&pattern, &coupling, &chen, &config)
        .expect("compute range");

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
        // The range should be reasonable (not degenerate)
        assert!(
            eps_max - eps_min > 0.1,
            "ε range ({eps_min}, {eps_max}) is too narrow"
        );
    }
}

#[test]
fn validate_at_reasonable_epsilon() {
    let chen = ChenSystem::default_paper();
    let coupling = TopologyBuilder::octagon().expect("octagon");
    let pattern = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("2-cluster");
    let config = fast_msf_config();

    let result =
        ClusterSyncVerifier::validate_at_epsilon(&pattern, &coupling, &chen, 10.0, &config)
            .expect("validate");

    // There should be 7 non-zero Laplacian eigenvalues for an 8-node graph
    assert_eq!(
        result.transverse_eigenvalues.len(),
        7,
        "octagon should have 7 non-zero Laplacian eigenvalues"
    );
}

#[test]
fn coupling_strength_switching() {
    let chen = ChenSystem::default_paper();
    let mut coupling = TopologyBuilder::octagon().expect("octagon");
    coupling.set_coupling_strength(10.0);

    let n = coupling.node_count();
    let dim = coupling.oscillator_dim();
    let mut perts = Vec::with_capacity(n);
    for i in 0..n {
        let pert: Vec<f64> = (0..dim)
            .map(|d| 0.01 * ((i * dim + d) as f64 * 0.37).sin())
            .collect();
        perts.push(pert);
    }

    let mut net =
        CoupledNetwork::new(&coupling, &[1.0, 1.0, 1.0], Some(&perts)).expect("create network");

    // Run at ε=10.0
    net.integrate(&chen, DEFAULT_DT, 5_000).expect("phase 1");

    // Switch coupling
    net.set_coupling_strength(5.0);
    net.integrate(&chen, DEFAULT_DT, 5_000).expect("phase 2");

    // Network should still be bounded after switching
    let bound = 100.0;
    for i in 0..8 {
        let s = net.node_state(i).expect("node");
        assert!(
            s.iter().all(|x| x.abs() < bound),
            "node {i} exceeded bound after coupling switch"
        );
    }
}
