use super::error::GraphError;
use crate::linalg::Matrix;

/// Coupling configuration for a network of coupled oscillators.
///
/// Stores the adjacency matrix `Ξ` (xi), the inner coupling matrix `Γ` (gamma),
/// and a global coupling strength `ε` (epsilon). The effective coupling is `ε·Ξ⊗Γ`.
///
/// Supports per-edge coupling strengths via weighted adjacency entries.
#[derive(Debug, Clone)]
pub struct CouplingMatrix {
    /// Adjacency matrix Ξ (n × n), possibly weighted.
    adjacency: Matrix,
    /// Inner coupling matrix Γ (d × d), where d is the oscillator dimension.
    inner_coupling: Matrix,
    /// Global coupling strength ε.
    epsilon: f64,
    /// Number of nodes.
    n: usize,
    /// Oscillator dimension.
    dim: usize,
}

impl CouplingMatrix {
    /// Create a new coupling configuration.
    ///
    /// - `adjacency`: n×n adjacency matrix Ξ
    /// - `inner_coupling`: d×d inner coupling matrix Γ
    /// - `epsilon`: global coupling strength
    pub fn new(
        adjacency: Matrix,
        inner_coupling: Matrix,
        epsilon: f64,
    ) -> Result<Self, GraphError> {
        if !adjacency.is_square() {
            return Err(GraphError::NotSquare {
                rows: adjacency.nrows(),
                cols: adjacency.ncols(),
            });
        }
        if !inner_coupling.is_square() {
            return Err(GraphError::NotSquare {
                rows: inner_coupling.nrows(),
                cols: inner_coupling.ncols(),
            });
        }
        let n = adjacency.nrows();
        let dim = inner_coupling.nrows();
        Ok(Self {
            adjacency,
            inner_coupling,
            epsilon,
            n,
            dim,
        })
    }

    /// Number of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Oscillator state dimension.
    pub fn oscillator_dim(&self) -> usize {
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

    /// Reference to the adjacency matrix Ξ.
    pub fn adjacency(&self) -> &Matrix {
        &self.adjacency
    }

    /// Reference to the inner coupling matrix Γ.
    pub fn inner_coupling(&self) -> &Matrix {
        &self.inner_coupling
    }

    /// Compute the degree matrix D = diag(row_sums(Ξ)).
    pub fn degree_matrix(&self) -> Result<Matrix, GraphError> {
        let mut degrees = vec![0.0; self.n];
        for (i, degree) in degrees.iter_mut().enumerate() {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += self.adjacency.get(i, j)?;
            }
            *degree = sum;
        }
        Ok(Matrix::from_diagonal(&degrees))
    }

    /// Compute the Laplacian matrix L = D - Ξ.
    ///
    /// The Laplacian has eigenvalue 0 (with eigenvector 1_n) and all other
    /// eigenvalues are positive for connected graphs.
    pub fn laplacian(&self) -> Result<Matrix, GraphError> {
        let mut lap = Matrix::zeros(self.n, self.n);
        for i in 0..self.n {
            let mut row_sum = 0.0;
            for j in 0..self.n {
                let aij = self.adjacency.get(i, j)?;
                if i != j {
                    lap.set(i, j, -aij)?;
                    row_sum += aij;
                }
            }
            lap.set(i, i, row_sum)?;
        }
        Ok(lap)
    }

    /// Compute the effective coupling matrix: ε · Ξ ⊗ Γ.
    ///
    /// This is the full (n·d × n·d) coupling matrix used in the coupled network ODE.
    pub fn effective_coupling(&self) -> Matrix {
        let scaled = self.scaled_adjacency();
        scaled.kronecker(&self.inner_coupling)
    }

    /// Compute ε · Ξ (scaled adjacency, without Kronecker).
    pub fn scaled_adjacency(&self) -> Matrix {
        let mut m = self.adjacency.clone();
        for i in 0..self.n {
            for j in 0..self.n {
                if let Ok(val) = m.get(i, j) {
                    // Silently ignore set errors since we know dimensions match
                    let _ = m.set(i, j, self.epsilon * val);
                }
            }
        }
        m
    }

    /// Check if the adjacency matrix is symmetric (undirected graph).
    pub fn is_symmetric(&self, tol: f64) -> Result<bool, GraphError> {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let aij = self.adjacency.get(i, j)?;
                let aji = self.adjacency.get(j, i)?;
                if (aij - aji).abs() > tol {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_ring_4() -> CouplingMatrix {
        // 4-node ring: 0-1-2-3-0
        let adj = Matrix::from_row_major(
            4,
            4,
            &[
                0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        let gamma = Matrix::from_diagonal(&[0.0, 1.0, 0.0]); // Γ = diag(0,1,0)
        CouplingMatrix::new(adj, gamma, 1.0).expect("ring4")
    }

    #[test]
    fn node_count_and_dim() {
        let cm = simple_ring_4();
        assert_eq!(cm.node_count(), 4);
        assert_eq!(cm.oscillator_dim(), 3);
    }

    #[test]
    fn coupling_strength() {
        let mut cm = simple_ring_4();
        assert!((cm.epsilon() - 1.0).abs() < 1e-15);
        cm.set_coupling_strength(5.0);
        assert!((cm.epsilon() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn degree_matrix() {
        let cm = simple_ring_4();
        let d = cm.degree_matrix().expect("degree");
        // Each node in 4-ring has degree 2
        for i in 0..4 {
            assert!((d.get(i, i).unwrap() - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn laplacian_row_sums_zero() {
        let cm = simple_ring_4();
        let lap = cm.laplacian().expect("laplacian");
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| lap.get(i, j).unwrap()).sum();
            assert!(
                row_sum.abs() < 1e-12,
                "laplacian row {i} sum = {row_sum}, expected 0"
            );
        }
    }

    #[test]
    fn laplacian_eigenvalue_zero() {
        use crate::linalg::symmetric_eigen;

        let cm = simple_ring_4();
        let lap = cm.laplacian().expect("laplacian");
        let eig = symmetric_eigen(&lap).expect("eigen");
        let reals = eig.real_eigenvalues();
        // Smallest eigenvalue should be 0
        assert!(
            reals[0].abs() < 1e-10,
            "smallest laplacian eigenvalue = {}, expected ~0",
            reals[0]
        );
        // All others positive for connected graph
        for (i, &val) in reals.iter().enumerate().skip(1) {
            assert!(
                val > -1e-10,
                "laplacian eigenvalue[{i}] = {val} is negative"
            );
        }
    }

    #[test]
    fn effective_coupling_dimensions() {
        let cm = simple_ring_4();
        let eff = cm.effective_coupling();
        // 4 nodes × 3 dim = 12×12
        assert_eq!(eff.nrows(), 12);
        assert_eq!(eff.ncols(), 12);
    }

    #[test]
    fn coupling_strength_scaling() {
        let mut cm = simple_ring_4();
        cm.set_coupling_strength(3.0);
        let scaled = cm.scaled_adjacency();
        // adj[0,1] = 1.0, so scaled[0,1] = 3.0
        assert!((scaled.get(0, 1).unwrap() - 3.0).abs() < 1e-12);
        // adj[0,2] = 0.0, so scaled[0,2] = 0.0
        assert!((scaled.get(0, 2).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn is_symmetric_check() {
        let cm = simple_ring_4();
        assert!(cm.is_symmetric(1e-12).expect("sym check"));
    }

    #[test]
    fn non_square_adjacency_error() {
        let adj = Matrix::zeros(3, 4);
        let gamma = Matrix::identity(3);
        assert!(CouplingMatrix::new(adj, gamma, 1.0).is_err());
    }

    #[test]
    fn non_square_inner_coupling_error() {
        let adj = Matrix::identity(4);
        let gamma = Matrix::zeros(3, 4);
        assert!(CouplingMatrix::new(adj, gamma, 1.0).is_err());
    }
}
