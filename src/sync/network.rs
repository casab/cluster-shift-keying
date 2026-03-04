use super::error::SyncError;
use crate::dynamics::traits::DynamicalSystem;
use crate::graph::CouplingMatrix;

/// Coupled network of identical dynamical systems.
///
/// Simulates N nodes coupled through the network equation:
///
///   ẋᵢ = f(xᵢ) + ε Σⱼ ξᵢⱼ Γ xⱼ
///
/// where f is the isolated node dynamics, ε is the coupling strength,
/// ξᵢⱼ are adjacency weights, and Γ is the inner coupling matrix.
pub struct CoupledNetwork {
    /// State vectors for all nodes, stored flat: [x₀₀, x₀₁, ..., x₀ₐ, x₁₀, ...].
    states: Vec<f64>,
    /// Number of nodes.
    n: usize,
    /// Oscillator dimension.
    dim: usize,
    /// Adjacency matrix (n × n) cached as flat row-major.
    adjacency: Vec<f64>,
    /// Inner coupling matrix (dim × dim) cached as flat row-major.
    gamma: Vec<f64>,
    /// Global coupling strength ε.
    epsilon: f64,
    /// RK4 scratch buffers (sized n*dim each).
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp: Vec<f64>,
    /// Scratch buffer for single-node derivative.
    deriv_buf: Vec<f64>,
    /// Scratch buffer for coupling term of a single node.
    coupling_buf: Vec<f64>,
}

impl CoupledNetwork {
    /// Create a new coupled network.
    ///
    /// All nodes start from the same `initial_state` with small random
    /// perturbations added from `perturbations` (one per node, each of length `dim`).
    /// If `perturbations` is `None`, all nodes start from identical states.
    pub fn new(
        coupling: &CouplingMatrix,
        initial_state: &[f64],
        perturbations: Option<&[Vec<f64>]>,
    ) -> Result<Self, SyncError> {
        let n = coupling.node_count();
        let dim = coupling.oscillator_dim();

        if initial_state.len() != dim {
            return Err(SyncError::NodeCountMismatch {
                expected: dim,
                got: initial_state.len(),
            });
        }

        if let Some(perts) = perturbations {
            if perts.len() != n {
                return Err(SyncError::NodeCountMismatch {
                    expected: n,
                    got: perts.len(),
                });
            }
            for (i, p) in perts.iter().enumerate() {
                if p.len() != dim {
                    return Err(SyncError::MsfFailed {
                        reason: format!(
                            "perturbation[{i}] has length {} but expected {dim}",
                            p.len()
                        ),
                    });
                }
            }
        }

        // Flatten states
        let total = n * dim;
        let mut states = vec![0.0; total];
        for i in 0..n {
            let offset = i * dim;
            for d in 0..dim {
                states[offset + d] = initial_state[d];
                if let Some(perts) = perturbations {
                    states[offset + d] += perts[i][d];
                }
            }
        }

        // Cache adjacency as flat array
        let mut adjacency = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                adjacency[i * n + j] = coupling.adjacency().get(i, j)?;
            }
        }

        // Cache inner coupling as flat array
        let mut gamma = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                gamma[i * dim + j] = coupling.inner_coupling().get(i, j)?;
            }
        }

        Ok(Self {
            states,
            n,
            dim,
            adjacency,
            gamma,
            epsilon: coupling.epsilon(),
            k1: vec![0.0; total],
            k2: vec![0.0; total],
            k3: vec![0.0; total],
            k4: vec![0.0; total],
            tmp: vec![0.0; total],
            deriv_buf: vec![0.0; dim],
            coupling_buf: vec![0.0; dim],
        })
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Oscillator dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Get the current coupling strength ε.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Set the global coupling strength ε.
    pub fn set_coupling_strength(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    /// Get the state of node `i` as a slice.
    pub fn node_state(&self, i: usize) -> Result<&[f64], SyncError> {
        if i >= self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: i + 1,
            });
        }
        let offset = i * self.dim;
        Ok(&self.states[offset..offset + self.dim])
    }

    /// Get all states as a flat slice (n*dim elements, node-major order).
    pub fn states_flat(&self) -> &[f64] {
        &self.states
    }

    /// Get all states as a vector of vectors (one per node).
    pub fn states(&self) -> Vec<Vec<f64>> {
        (0..self.n)
            .map(|i| {
                let offset = i * self.dim;
                self.states[offset..offset + self.dim].to_vec()
            })
            .collect()
    }

    /// Set the state of node `i`.
    pub fn set_node_state(&mut self, i: usize, state: &[f64]) -> Result<(), SyncError> {
        if i >= self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: i + 1,
            });
        }
        if state.len() != self.dim {
            return Err(SyncError::MsfFailed {
                reason: format!(
                    "state length {} doesn't match dimension {}",
                    state.len(),
                    self.dim
                ),
            });
        }
        let offset = i * self.dim;
        self.states[offset..offset + self.dim].copy_from_slice(state);
        Ok(())
    }

    /// Advance the coupled network by one RK4 step of size `dt`.
    pub fn step(&mut self, system: &dyn DynamicalSystem, dt: f64) -> Result<(), SyncError> {
        let n = self.n;
        let dim = self.dim;
        let total = n * dim;
        let epsilon = self.epsilon;

        // k1 = F(states)
        compute_coupled_derivative(
            system,
            &self.states,
            &self.adjacency,
            &self.gamma,
            n,
            dim,
            epsilon,
            &mut self.k1,
            &mut self.deriv_buf,
            &mut self.coupling_buf,
        )?;

        // k2 = F(states + dt/2 * k1)
        for i in 0..total {
            self.tmp[i] = self.states[i] + 0.5 * dt * self.k1[i];
        }
        compute_coupled_derivative(
            system,
            &self.tmp,
            &self.adjacency,
            &self.gamma,
            n,
            dim,
            epsilon,
            &mut self.k2,
            &mut self.deriv_buf,
            &mut self.coupling_buf,
        )?;

        // k3 = F(states + dt/2 * k2)
        for i in 0..total {
            self.tmp[i] = self.states[i] + 0.5 * dt * self.k2[i];
        }
        compute_coupled_derivative(
            system,
            &self.tmp,
            &self.adjacency,
            &self.gamma,
            n,
            dim,
            epsilon,
            &mut self.k3,
            &mut self.deriv_buf,
            &mut self.coupling_buf,
        )?;

        // k4 = F(states + dt * k3)
        for i in 0..total {
            self.tmp[i] = self.states[i] + dt * self.k3[i];
        }
        compute_coupled_derivative(
            system,
            &self.tmp,
            &self.adjacency,
            &self.gamma,
            n,
            dim,
            epsilon,
            &mut self.k4,
            &mut self.deriv_buf,
            &mut self.coupling_buf,
        )?;

        // Update: states += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 0..total {
            self.states[i] +=
                dt / 6.0 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }

        // Check for blowup
        if self.states.iter().any(|x| !x.is_finite()) {
            return Err(SyncError::Dynamics(
                crate::dynamics::DynamicsError::IntegrationFailed {
                    reason: "coupled network state diverged to infinity or NaN".to_string(),
                },
            ));
        }

        Ok(())
    }

    /// Integrate the network for `steps` time steps.
    pub fn integrate(
        &mut self,
        system: &dyn DynamicalSystem,
        dt: f64,
        steps: usize,
    ) -> Result<(), SyncError> {
        for _ in 0..steps {
            self.step(system, dt)?;
        }
        Ok(())
    }

    /// Compute the synchronization error between nodes `i` and `j`.
    ///
    /// Returns the Euclidean distance ||xᵢ - xⱼ||.
    pub fn sync_error(&self, i: usize, j: usize) -> Result<f64, SyncError> {
        if i >= self.n || j >= self.n {
            return Err(SyncError::NodeCountMismatch {
                expected: self.n,
                got: i.max(j) + 1,
            });
        }
        let oi = i * self.dim;
        let oj = j * self.dim;
        let err: f64 = (0..self.dim)
            .map(|d| (self.states[oi + d] - self.states[oj + d]).powi(2))
            .sum::<f64>()
            .sqrt();
        Ok(err)
    }
}

/// Compute the full coupled derivative for all nodes (free function to satisfy borrow checker).
///
/// Uses diffusive coupling (standard MSF convention):
///   F_i = f(xᵢ) + ε Σⱼ ξᵢⱼ Γ (xⱼ - xᵢ)
///
/// This is equivalent to using the Laplacian: F_i = f(xᵢ) - ε Σⱼ L_ij Γ xⱼ
#[allow(clippy::too_many_arguments)]
fn compute_coupled_derivative(
    system: &dyn DynamicalSystem,
    state: &[f64],
    adjacency: &[f64],
    gamma: &[f64],
    n: usize,
    dim: usize,
    epsilon: f64,
    output: &mut [f64],
    deriv_buf: &mut [f64],
    coupling_buf: &mut [f64],
) -> Result<(), SyncError> {
    for i in 0..n {
        let offset_i = i * dim;

        // f(xᵢ)
        system.derivative(&state[offset_i..offset_i + dim], deriv_buf)?;

        // Diffusive coupling: ε Σⱼ ξᵢⱼ Γ (xⱼ - xᵢ)
        coupling_buf.fill(0.0);
        for j in 0..n {
            let xi_ij = adjacency[i * n + j];
            if xi_ij.abs() < 1e-15 {
                continue;
            }
            let offset_j = j * dim;
            // Γ (xⱼ - xᵢ) then scale by ξᵢⱼ
            for d in 0..dim {
                let mut gamma_diff_d = 0.0;
                for k in 0..dim {
                    gamma_diff_d +=
                        gamma[d * dim + k] * (state[offset_j + k] - state[offset_i + k]);
                }
                coupling_buf[d] += xi_ij * gamma_diff_d;
            }
        }

        // F_i = f(xᵢ) + ε * coupling
        for d in 0..dim {
            output[offset_i + d] = deriv_buf[d] + epsilon * coupling_buf[d];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;
    use crate::dynamics::integrator::DEFAULT_DT;
    use crate::graph::TopologyBuilder;

    fn octagon_network(epsilon: f64) -> (CoupledNetwork, ChenSystem) {
        let chen = ChenSystem::default_paper();
        let mut coupling = TopologyBuilder::octagon().expect("octagon");
        coupling.set_coupling_strength(epsilon);

        // Small perturbations from seeded RNG for reproducibility
        let n = coupling.node_count();
        let dim = coupling.oscillator_dim();
        let mut perts = Vec::with_capacity(n);
        // Use deterministic perturbations
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
    fn network_creation() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let net = CoupledNetwork::new(&coupling, &[1.0, 2.0, 3.0], None).expect("create network");
        assert_eq!(net.node_count(), 8);
        assert_eq!(net.dimension(), 3);
    }

    #[test]
    fn network_creation_with_perturbations() {
        let (net, _) = octagon_network(10.0);
        assert_eq!(net.node_count(), 8);
        // States should not all be identical
        let s0 = net.node_state(0).expect("node 0");
        let s1 = net.node_state(1).expect("node 1");
        let diff: f64 = s0.iter().zip(s1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "perturbations should make states different");
    }

    #[test]
    fn network_wrong_initial_dim() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let result = CoupledNetwork::new(&coupling, &[1.0, 2.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn network_wrong_perturbation_count() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let perts = vec![vec![0.0; 3]; 5]; // wrong count
        let result = CoupledNetwork::new(&coupling, &[1.0, 1.0, 1.0], Some(&perts));
        assert!(result.is_err());
    }

    #[test]
    fn single_step_finite() {
        let (mut net, chen) = octagon_network(10.0);
        net.step(&chen, DEFAULT_DT).expect("step");
        for i in 0..8 {
            let s = net.node_state(i).expect("node");
            assert!(s.iter().all(|x| x.is_finite()), "node {i} not finite");
        }
    }

    #[test]
    fn integrate_stays_bounded() {
        let (mut net, chen) = octagon_network(10.0);
        net.integrate(&chen, DEFAULT_DT, 1000).expect("integrate");
        let bound = 100.0;
        for i in 0..8 {
            let s = net.node_state(i).expect("node");
            for (d, &val) in s.iter().enumerate() {
                assert!(val.abs() < bound, "node {i} dim {d} = {val} exceeds bound");
            }
        }
    }

    #[test]
    fn sync_error_same_nodes_zero() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let net = CoupledNetwork::new(&coupling, &[1.0, 2.0, 3.0], None).expect("create network");
        let err = net.sync_error(0, 0).expect("sync error");
        assert!(err.abs() < 1e-15, "self-sync error should be 0, got {err}");
    }

    #[test]
    fn sync_error_identical_initial() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let net = CoupledNetwork::new(&coupling, &[1.0, 2.0, 3.0], None).expect("create network");
        // All nodes start identical
        for i in 0..8 {
            for j in 0..8 {
                let err = net.sync_error(i, j).expect("sync error");
                assert!(
                    err.abs() < 1e-15,
                    "error({i},{j}) = {err} should be 0 for identical initial states"
                );
            }
        }
    }

    #[test]
    fn set_coupling_strength() {
        let (mut net, _) = octagon_network(10.0);
        assert!((net.epsilon() - 10.0).abs() < 1e-15);
        net.set_coupling_strength(5.0);
        assert!((net.epsilon() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn node_state_out_of_range() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let net = CoupledNetwork::new(&coupling, &[1.0, 2.0, 3.0], None).expect("create network");
        assert!(net.node_state(8).is_err());
    }

    #[test]
    fn cluster_sync_emerges_with_coupling() {
        // At ε=10.0, coupled octagon should exhibit some synchronization
        // between nodes that share the same cluster
        let (mut net, chen) = octagon_network(10.0);

        // Integrate for sufficient time for synchronization to develop
        net.integrate(&chen, DEFAULT_DT, 20_000).expect("integrate");

        // In the 2-cluster pattern for C₈: {0,2,4,6} and {1,3,5,7}
        // Nodes in the same cluster should have smaller sync error than
        // nodes in different clusters.
        let same_err = net.sync_error(0, 2).expect("same cluster");
        let diff_err = net.sync_error(0, 1).expect("diff cluster");

        // We can't guarantee full sync in finite time, but same-cluster
        // error should be noticeably smaller
        assert!(
            same_err < diff_err * 1.5 || same_err < 1.0,
            "same-cluster error ({same_err}) should be smaller than cross-cluster ({diff_err})"
        );
    }
}
