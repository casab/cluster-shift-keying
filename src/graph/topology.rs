use super::coupling::CouplingMatrix;
use super::error::GraphError;
use crate::linalg::Matrix;

/// Builder for common network topologies.
///
/// Each method returns a `CouplingMatrix` with the given adjacency
/// structure, a default inner coupling Γ = diag(0,1,0) (coupling through
/// the second state variable of a 3D oscillator), and ε = 1.0.
pub struct TopologyBuilder;

impl TopologyBuilder {
    /// Default inner coupling matrix Γ = diag(0, 1, 0) for 3D oscillators.
    fn default_inner_coupling() -> Matrix {
        Matrix::from_diagonal(&[0.0, 1.0, 0.0])
    }

    /// Build the paper's 8-node octagon topology.
    ///
    /// An 8-node ring (cycle graph C₈) where each node is connected to
    /// its two nearest neighbours. This is the primary topology used in
    /// the CLSK paper.
    pub fn octagon() -> Result<CouplingMatrix, GraphError> {
        Self::ring(8)
    }

    /// Build a ring (cycle) graph with `n` nodes.
    ///
    /// Node i is connected to nodes (i-1) mod n and (i+1) mod n.
    pub fn ring(n: usize) -> Result<CouplingMatrix, GraphError> {
        if n < 3 {
            return Err(GraphError::InvalidNodeCount {
                count: n,
                minimum: 3,
            });
        }

        let mut edges = Vec::with_capacity(2 * n);
        for i in 0..n {
            let next = (i + 1) % n;
            edges.push((i, next, 1.0));
            edges.push((next, i, 1.0));
        }

        let adj = Matrix::from_adjacency(n, &edges)?;
        CouplingMatrix::new(adj, Self::default_inner_coupling(), 1.0)
    }

    /// Build a complete (fully connected) graph with `n` nodes.
    ///
    /// Every node is connected to every other node with weight 1.
    pub fn complete(n: usize) -> Result<CouplingMatrix, GraphError> {
        if n < 2 {
            return Err(GraphError::InvalidNodeCount {
                count: n,
                minimum: 2,
            });
        }

        let mut edges = Vec::with_capacity(n * (n - 1));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i, j, 1.0));
                }
            }
        }

        let adj = Matrix::from_adjacency(n, &edges)?;
        CouplingMatrix::new(adj, Self::default_inner_coupling(), 1.0)
    }

    /// Build a 2D lattice (grid) graph with periodic boundary conditions (torus).
    ///
    /// Each node is connected to its 4 nearest neighbours on a `rows × cols` grid.
    pub fn lattice_2d(rows: usize, cols: usize) -> Result<CouplingMatrix, GraphError> {
        let n = rows * cols;
        if n < 4 {
            return Err(GraphError::InvalidNodeCount {
                count: n,
                minimum: 4,
            });
        }

        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                // Right neighbour
                let right = r * cols + (c + 1) % cols;
                edges.push((idx, right, 1.0));
                edges.push((right, idx, 1.0));
                // Down neighbour
                let down = ((r + 1) % rows) * cols + c;
                edges.push((idx, down, 1.0));
                edges.push((down, idx, 1.0));
            }
        }

        let adj = Matrix::from_adjacency(n, &edges)?;
        CouplingMatrix::new(adj, Self::default_inner_coupling(), 1.0)
    }

    /// Build a coupling matrix from a user-supplied adjacency matrix.
    ///
    /// The adjacency matrix must be square. Inner coupling defaults to Γ = diag(0,1,0).
    pub fn from_adjacency(adj: Matrix) -> Result<CouplingMatrix, GraphError> {
        if !adj.is_square() {
            return Err(GraphError::NotSquare {
                rows: adj.nrows(),
                cols: adj.ncols(),
            });
        }
        CouplingMatrix::new(adj, Self::default_inner_coupling(), 1.0)
    }

    /// Build a coupling matrix from a user-supplied adjacency and inner coupling.
    pub fn from_adjacency_with_gamma(
        adj: Matrix,
        gamma: Matrix,
    ) -> Result<CouplingMatrix, GraphError> {
        CouplingMatrix::new(adj, gamma, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::symmetric_eigen;

    #[test]
    fn octagon_is_8_node_ring() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        assert_eq!(cm.node_count(), 8);
        assert_eq!(cm.oscillator_dim(), 3);
        assert!(cm.is_symmetric(1e-12).expect("sym"));
    }

    #[test]
    fn ring_adjacency_structure() {
        let cm = TopologyBuilder::ring(5).expect("ring5");
        let adj = cm.adjacency();
        // Node 0 connects to 1 and 4
        assert!((adj.get(0, 1).unwrap() - 1.0).abs() < 1e-12);
        assert!((adj.get(0, 4).unwrap() - 1.0).abs() < 1e-12);
        // Node 0 does not connect to 2 or 3
        assert!(adj.get(0, 2).unwrap().abs() < 1e-12);
        assert!(adj.get(0, 3).unwrap().abs() < 1e-12);
        // No self-loops
        assert!(adj.get(0, 0).unwrap().abs() < 1e-12);
    }

    #[test]
    fn ring_too_small() {
        assert!(TopologyBuilder::ring(2).is_err());
        assert!(TopologyBuilder::ring(1).is_err());
    }

    #[test]
    fn ring_laplacian_eigenvalues() {
        // Ring(n) Laplacian eigenvalues: 2 - 2cos(2πk/n) for k=0..n-1
        let n = 6;
        let cm = TopologyBuilder::ring(n).expect("ring6");
        let lap = cm.laplacian().expect("laplacian");
        let eig = symmetric_eigen(&lap).expect("eigen");
        let computed = eig.real_eigenvalues();

        let mut expected: Vec<f64> = (0..n)
            .map(|k| 2.0 - 2.0 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (i, (c, e)) in computed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (c - e).abs() < 1e-10,
                "ring6 Laplacian eigenvalue[{i}]: computed={c}, expected={e}"
            );
        }
    }

    #[test]
    fn octagon_laplacian_spectrum() {
        // C₈ Laplacian eigenvalues: 2 - 2cos(2πk/8) for k=0..7
        let cm = TopologyBuilder::octagon().expect("octagon");
        let lap = cm.laplacian().expect("laplacian");
        let eig = symmetric_eigen(&lap).expect("eigen");
        let computed = eig.real_eigenvalues();

        // Should have eigenvalue 0 (connected graph)
        assert!(computed[0].abs() < 1e-10, "first eigenvalue should be ~0");

        // Max eigenvalue for C₈ is 2 - 2cos(π) = 4
        let max_eig = computed.last().unwrap();
        assert!(
            (max_eig - 4.0).abs() < 1e-10,
            "max eigenvalue = {max_eig}, expected 4"
        );
    }

    #[test]
    fn complete_graph_structure() {
        let cm = TopologyBuilder::complete(4).expect("K4");
        assert_eq!(cm.node_count(), 4);
        let adj = cm.adjacency();
        // All off-diagonal = 1
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 0.0 } else { 1.0 };
                assert!(
                    (adj.get(i, j).unwrap() - expected).abs() < 1e-12,
                    "K4 adj[{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn complete_laplacian_eigenvalues() {
        // K_n Laplacian: eigenvalue 0 (once), eigenvalue n (n-1 times)
        let n = 5;
        let cm = TopologyBuilder::complete(n).expect("K5");
        let lap = cm.laplacian().expect("laplacian");
        let eig = symmetric_eigen(&lap).expect("eigen");
        let reals = eig.real_eigenvalues();

        assert!(reals[0].abs() < 1e-10, "K5 should have eigenvalue 0");
        for i in 1..n {
            assert!(
                (reals[i] - n as f64).abs() < 1e-10,
                "K5 eigenvalue[{i}] = {}, expected {n}",
                reals[i]
            );
        }
    }

    #[test]
    fn complete_too_small() {
        assert!(TopologyBuilder::complete(1).is_err());
    }

    #[test]
    fn lattice_2d_basic() {
        let cm = TopologyBuilder::lattice_2d(3, 3).expect("3x3 torus");
        assert_eq!(cm.node_count(), 9);
        assert!(cm.is_symmetric(1e-12).expect("sym"));

        // Each node on a torus has degree 4
        let d = cm.degree_matrix().expect("degree");
        for i in 0..9 {
            assert!(
                (d.get(i, i).unwrap() - 4.0).abs() < 1e-12,
                "torus node {i} degree != 4"
            );
        }
    }

    #[test]
    fn lattice_too_small() {
        assert!(TopologyBuilder::lattice_2d(1, 2).is_err());
    }

    #[test]
    fn from_adjacency_custom() {
        // Star graph: node 0 connects to all others
        let n = 4;
        let mut edges = Vec::new();
        for i in 1..n {
            edges.push((0, i, 1.0));
            edges.push((i, 0, 1.0));
        }
        let adj = Matrix::from_adjacency(n, &edges).unwrap();
        let cm = TopologyBuilder::from_adjacency(adj).expect("star");
        assert_eq!(cm.node_count(), 4);
    }

    #[test]
    fn from_adjacency_non_square_error() {
        let adj = Matrix::zeros(3, 4);
        assert!(TopologyBuilder::from_adjacency(adj).is_err());
    }

    #[test]
    fn coupling_strength_scales_effective() {
        let mut cm = TopologyBuilder::ring(4).expect("ring4");
        let eff1 = cm.effective_coupling();

        cm.set_coupling_strength(3.0);
        let eff3 = cm.effective_coupling();

        // eff3 should be 3× eff1 everywhere
        for i in 0..eff1.nrows() {
            for j in 0..eff1.ncols() {
                let v1 = eff1.get(i, j).unwrap();
                let v3 = eff3.get(i, j).unwrap();
                if v1.abs() > 1e-15 {
                    assert!(
                        (v3 / v1 - 3.0).abs() < 1e-12,
                        "scaling mismatch at ({i},{j})"
                    );
                } else {
                    assert!(v3.abs() < 1e-12);
                }
            }
        }
    }
}
