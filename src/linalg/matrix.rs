use super::error::LinalgError;
use nalgebra::DMatrix;

/// Thin wrapper around `nalgebra::DMatrix<f64>` providing construction helpers
/// and domain-specific utilities for the CLSK pipeline.
#[derive(Debug, Clone)]
pub struct Matrix {
    inner: DMatrix<f64>,
}

impl Matrix {
    /// Create a matrix from a `nalgebra::DMatrix`.
    pub fn from_nalgebra(m: DMatrix<f64>) -> Self {
        Self { inner: m }
    }

    /// Create a matrix from row-major data.
    ///
    /// `data` must have exactly `rows * cols` elements.
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64]) -> Result<Self, LinalgError> {
        if data.len() != rows * cols {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("{rows}x{cols} = {} elements", rows * cols),
                got: format!("{} elements", data.len()),
            });
        }
        Ok(Self {
            inner: DMatrix::from_row_slice(rows, cols, data),
        })
    }

    /// Create a zero matrix of size `rows x cols`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            inner: DMatrix::zeros(rows, cols),
        }
    }

    /// Create an identity matrix of size `n x n`.
    pub fn identity(n: usize) -> Self {
        Self {
            inner: DMatrix::identity(n, n),
        }
    }

    /// Build a matrix from an adjacency list representation.
    ///
    /// `n` is the number of nodes. `edges` are `(i, j, weight)` tuples.
    /// The resulting matrix is **not** automatically symmetrized.
    pub fn from_adjacency(n: usize, edges: &[(usize, usize, f64)]) -> Result<Self, LinalgError> {
        let mut m = DMatrix::zeros(n, n);
        for &(i, j, w) in edges {
            if i >= n || j >= n {
                return Err(LinalgError::DimensionMismatch {
                    expected: format!("indices < {n}"),
                    got: format!("edge ({i}, {j})"),
                });
            }
            m[(i, j)] = w;
        }
        Ok(Self { inner: m })
    }

    /// Compute the Kronecker product `self ⊗ other`.
    pub fn kronecker(&self, other: &Matrix) -> Matrix {
        let a = &self.inner;
        let b = &other.inner;
        let (ar, ac) = a.shape();
        let (br, bc) = b.shape();
        let mut result = DMatrix::zeros(ar * br, ac * bc);

        for i in 0..ar {
            for j in 0..ac {
                let aij = a[(i, j)];
                let block = b * aij;
                result
                    .view_mut((i * br, j * bc), (br, bc))
                    .copy_from(&block);
            }
        }

        Matrix { inner: result }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Result<f64, LinalgError> {
        if row >= self.inner.nrows() || col >= self.inner.ncols() {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("indices < {}x{}", self.inner.nrows(), self.inner.ncols()),
                got: format!("({row}, {col})"),
            });
        }
        Ok(self.inner[(row, col)])
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), LinalgError> {
        if row >= self.inner.nrows() || col >= self.inner.ncols() {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("indices < {}x{}", self.inner.nrows(), self.inner.ncols()),
                got: format!("({row}, {col})"),
            });
        }
        self.inner[(row, col)] = value;
        Ok(())
    }

    /// Return a reference to the underlying nalgebra matrix.
    pub fn as_nalgebra(&self) -> &DMatrix<f64> {
        &self.inner
    }

    /// Consume self and return the underlying nalgebra matrix.
    pub fn into_nalgebra(self) -> DMatrix<f64> {
        self.inner
    }

    /// Check if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.inner.nrows() == self.inner.ncols()
    }

    /// Ensure the matrix is square, returning an error if not.
    pub fn ensure_square(&self) -> Result<(), LinalgError> {
        if !self.is_square() {
            return Err(LinalgError::NotSquare {
                rows: self.inner.nrows(),
                cols: self.inner.ncols(),
            });
        }
        Ok(())
    }

    /// Return the diagonal as a vector.
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.inner.nrows().min(self.inner.ncols());
        (0..n).map(|i| self.inner[(i, i)]).collect()
    }

    /// Create a diagonal matrix from a slice.
    pub fn from_diagonal(diag: &[f64]) -> Self {
        let v = nalgebra::DVector::from_column_slice(diag);
        Self {
            inner: DMatrix::from_diagonal(&v),
        }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Matrix {
        Matrix {
            inner: self.inner.transpose(),
        }
    }

    /// Matrix-matrix multiply: self * other.
    pub fn mul(&self, other: &Matrix) -> Result<Matrix, LinalgError> {
        if self.inner.ncols() != other.inner.nrows() {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("ncols={} to match nrows", self.inner.ncols()),
                got: format!("nrows={}", other.inner.nrows()),
            });
        }
        Ok(Matrix {
            inner: &self.inner * &other.inner,
        })
    }

    /// Compute the Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.inner.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Pretty-print the matrix for debugging.
    pub fn pretty_print(&self) -> String {
        let (r, c) = self.inner.shape();
        let mut lines = Vec::with_capacity(r + 1);
        lines.push(format!("Matrix {r}x{c}:"));
        for i in 0..r {
            let row: Vec<String> = (0..c)
                .map(|j| format!("{:10.4}", self.inner[(i, j)]))
                .collect();
            lines.push(format!("  [{}]", row.join(", ")));
        }
        lines.join("\n")
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pretty_print())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_identity() {
        let z = Matrix::zeros(3, 4);
        assert_eq!(z.nrows(), 3);
        assert_eq!(z.ncols(), 4);
        assert!((z.get(0, 0).unwrap()).abs() < 1e-15);

        let id = Matrix::identity(3);
        assert!((id.get(0, 0).unwrap() - 1.0).abs() < 1e-15);
        assert!((id.get(0, 1).unwrap()).abs() < 1e-15);
        assert!((id.get(1, 1).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn from_row_major() {
        let m = Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!((m.get(0, 0).unwrap() - 1.0).abs() < 1e-15);
        assert!((m.get(0, 2).unwrap() - 3.0).abs() < 1e-15);
        assert!((m.get(1, 0).unwrap() - 4.0).abs() < 1e-15);
        assert!((m.get(1, 2).unwrap() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn from_row_major_dimension_error() {
        assert!(Matrix::from_row_major(2, 3, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn from_adjacency_basic() {
        let m = Matrix::from_adjacency(3, &[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        assert!((m.get(0, 1).unwrap() - 1.0).abs() < 1e-15);
        assert!((m.get(1, 2).unwrap() - 2.0).abs() < 1e-15);
        assert!((m.get(0, 0).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn from_adjacency_out_of_bounds() {
        assert!(Matrix::from_adjacency(3, &[(0, 5, 1.0)]).is_err());
    }

    #[test]
    fn kronecker_product_identity() {
        // I_2 ⊗ I_3 = I_6
        let i2 = Matrix::identity(2);
        let i3 = Matrix::identity(3);
        let result = i2.kronecker(&i3);
        assert_eq!(result.nrows(), 6);
        assert_eq!(result.ncols(), 6);

        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.get(i, j).unwrap() - expected).abs() < 1e-15,
                    "I2⊗I3 at ({i},{j}): got {}, expected {expected}",
                    result.get(i, j).unwrap()
                );
            }
        }
    }

    #[test]
    fn kronecker_product_known_values() {
        // [[1,2],[3,4]] ⊗ [[0,5],[6,7]] — standard textbook example
        let a = Matrix::from_row_major(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_row_major(2, 2, &[0.0, 5.0, 6.0, 7.0]).unwrap();
        let k = a.kronecker(&b);

        assert_eq!(k.nrows(), 4);
        assert_eq!(k.ncols(), 4);

        // Expected:
        // [[ 0,  5,  0, 10],
        //  [ 6,  7, 12, 14],
        //  [ 0, 15,  0, 20],
        //  [18, 21, 24, 28]]
        let expected = [
            0.0, 5.0, 0.0, 10.0, 6.0, 7.0, 12.0, 14.0, 0.0, 15.0, 0.0, 20.0, 18.0, 21.0, 24.0, 28.0,
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (k.get(i, j).unwrap() - expected[i * 4 + j]).abs() < 1e-12,
                    "kronecker at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn diagonal_roundtrip() {
        let diag = vec![1.0, 2.0, 3.0];
        let m = Matrix::from_diagonal(&diag);
        let extracted = m.diagonal();
        for (a, b) in diag.iter().zip(extracted.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        // Off-diagonal should be zero
        assert!((m.get(0, 1).unwrap()).abs() < 1e-15);
        assert!((m.get(1, 0).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn matrix_multiply() {
        let a = Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_row_major(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((c.get(0, 0).unwrap() - 58.0).abs() < 1e-12);
        assert!((c.get(0, 1).unwrap() - 64.0).abs() < 1e-12);
        assert!((c.get(1, 0).unwrap() - 139.0).abs() < 1e-12);
        assert!((c.get(1, 1).unwrap() - 154.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_multiply_dimension_error() {
        let a = Matrix::from_row_major(2, 3, &[1.0; 6]).unwrap();
        let b = Matrix::from_row_major(2, 2, &[1.0; 4]).unwrap();
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn transpose() {
        let m = Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert!((t.get(0, 0).unwrap() - 1.0).abs() < 1e-15);
        assert!((t.get(2, 1).unwrap() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn ensure_square_error() {
        let m = Matrix::zeros(2, 3);
        assert!(m.ensure_square().is_err());
        let s = Matrix::zeros(3, 3);
        assert!(s.ensure_square().is_ok());
    }

    #[test]
    fn frobenius_norm() {
        let m = Matrix::from_row_major(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((m.frobenius_norm() - expected).abs() < 1e-12);
    }

    #[test]
    fn get_out_of_bounds() {
        let m = Matrix::zeros(2, 2);
        assert!(m.get(5, 0).is_err());
    }

    #[test]
    fn set_and_get() {
        let mut m = Matrix::zeros(2, 2);
        m.set(1, 0, 42.0).unwrap();
        assert!((m.get(1, 0).unwrap() - 42.0).abs() < 1e-15);
    }

    #[test]
    fn set_out_of_bounds() {
        let mut m = Matrix::zeros(2, 2);
        assert!(m.set(5, 0, 1.0).is_err());
    }

    #[test]
    fn display_format() {
        let m = Matrix::identity(2);
        let s = format!("{m}");
        assert!(s.contains("Matrix 2x2"));
    }
}
