use super::error::LinalgError;
use super::matrix::Matrix;

/// Compressed Sparse Row (CSR) matrix representation.
///
/// Stores only non-zero entries, reducing memory from O(N²) to O(nnz)
/// where nnz is the number of non-zero elements. For sparse graphs
/// (degree << N), this is a significant memory reduction.
///
/// For a ring graph with N=10000, dense storage requires 800MB while
/// CSR requires ~480KB (N·deg entries ≈ 20000 non-zeros).
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Row pointers: row_ptrs[i] is the index into col_indices/values
    /// where row i starts. row_ptrs[nrows] = nnz.
    row_ptrs: Vec<usize>,
    /// Column indices of non-zero entries.
    col_indices: Vec<usize>,
    /// Values of non-zero entries.
    values: Vec<f64>,
}

impl SparseMatrix {
    /// Create a sparse matrix from a dense `Matrix`, dropping entries
    /// with absolute value below `tol`.
    pub fn from_dense(m: &Matrix, tol: f64) -> Result<Self, LinalgError> {
        let nrows = m.nrows();
        let ncols = m.ncols();
        let mut row_ptrs = Vec::with_capacity(nrows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..nrows {
            row_ptrs.push(col_indices.len());
            for j in 0..ncols {
                let val = m.get(i, j)?;
                if val.abs() > tol {
                    col_indices.push(j);
                    values.push(val);
                }
            }
        }
        row_ptrs.push(col_indices.len());

        Ok(Self {
            nrows,
            ncols,
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Create a sparse matrix from COO (coordinate) format triplets.
    ///
    /// Entries with the same (row, col) are summed. Entries below `tol`
    /// are dropped after summation.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, f64)],
        tol: f64,
    ) -> Result<Self, LinalgError> {
        let mut dense = Matrix::zeros(nrows, ncols);
        for &(r, c, v) in triplets {
            if r >= nrows || c >= ncols {
                return Err(LinalgError::DimensionMismatch {
                    expected: format!("indices < {nrows}x{ncols}"),
                    got: format!("({r}, {c})"),
                });
            }
            let existing = dense.get(r, c)?;
            dense.set(r, c, existing + v)?;
        }
        Self::from_dense(&dense, tol)
    }

    /// Convert back to a dense `Matrix`.
    pub fn to_dense(&self) -> Matrix {
        let mut m = Matrix::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for idx in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let j = self.col_indices[idx];
                let v = self.values[idx];
                // We know dimensions are valid since we built the sparse matrix
                let _ = m.set(i, j, v);
            }
        }
        m
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get a value at (row, col). Returns 0.0 for entries not stored.
    pub fn get(&self, row: usize, col: usize) -> Result<f64, LinalgError> {
        if row >= self.nrows || col >= self.ncols {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("indices < {}x{}", self.nrows, self.ncols),
                got: format!("({row}, {col})"),
            });
        }
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        // Binary search in the column indices for this row
        match self.col_indices[start..end].binary_search(&col) {
            Ok(pos) => Ok(self.values[start + pos]),
            Err(_) => Ok(0.0),
        }
    }

    /// Get the index range for row `row` into col_indices/values.
    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        self.row_ptrs[row]..self.row_ptrs[row + 1]
    }

    /// Get the column indices slice.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get the values slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Sparse matrix-vector multiply: y = A * x.
    ///
    /// `x` must have length `ncols`, `y` must have length `nrows`.
    pub fn mul_vec(&self, x: &[f64], y: &mut [f64]) -> Result<(), LinalgError> {
        if x.len() != self.ncols {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("vector length {}", self.ncols),
                got: format!("{}", x.len()),
            });
        }
        if y.len() != self.nrows {
            return Err(LinalgError::DimensionMismatch {
                expected: format!("output length {}", self.nrows),
                got: format!("{}", y.len()),
            });
        }

        for (i, yi) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for idx in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                sum += self.values[idx] * x[self.col_indices[idx]];
            }
            *yi = sum;
        }
        Ok(())
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        // row_ptrs: (nrows+1) * 8
        // col_indices: nnz * 8
        // values: nnz * 8
        (self.nrows + 1) * 8 + self.values.len() * 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_dense_roundtrip() {
        let dense = Matrix::from_row_major(3, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
            .expect("dense");
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        assert_eq!(sparse.nnz(), 5); // five non-zero entries: 1, 2, 3, 4, 5
        assert_eq!(sparse.nrows(), 3);
        assert_eq!(sparse.ncols(), 3);

        let recovered = sparse.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (dense.get(i, j).unwrap() - recovered.get(i, j).unwrap()).abs() < 1e-15,
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn sparse_get() {
        let dense = Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 2.0]).expect("dense");
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        assert!((sparse.get(0, 0).unwrap() - 1.0).abs() < 1e-15);
        assert!((sparse.get(0, 1).unwrap() - 0.0).abs() < 1e-15);
        assert!((sparse.get(1, 0).unwrap() - 0.0).abs() < 1e-15);
        assert!((sparse.get(1, 1).unwrap() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn sparse_get_out_of_bounds() {
        let dense = Matrix::identity(2);
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        assert!(sparse.get(2, 0).is_err());
    }

    #[test]
    fn sparse_mul_vec() {
        // A = [[1, 2], [3, 4]], x = [1, 1] => y = [3, 7]
        let dense = Matrix::from_row_major(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("dense");
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        let x = [1.0, 1.0];
        let mut y = [0.0; 2];
        sparse.mul_vec(&x, &mut y).expect("mul_vec");
        assert!((y[0] - 3.0).abs() < 1e-15);
        assert!((y[1] - 7.0).abs() < 1e-15);
    }

    #[test]
    fn sparse_mul_vec_dimension_mismatch() {
        let dense = Matrix::identity(3);
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        let x = [1.0, 2.0]; // wrong length
        let mut y = [0.0; 3];
        assert!(sparse.mul_vec(&x, &mut y).is_err());
    }

    #[test]
    fn sparse_identity() {
        let dense = Matrix::identity(5);
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        assert_eq!(sparse.nnz(), 5);
        for i in 0..5 {
            assert!((sparse.get(i, i).unwrap() - 1.0).abs() < 1e-15);
            for j in 0..5 {
                if i != j {
                    assert!((sparse.get(i, j).unwrap()).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn sparse_ring_adjacency() {
        // Ring(8) adjacency: each node connects to 2 neighbors
        let n = 8;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push(((i + 1) % n, i, 1.0));
        }
        let dense = Matrix::from_adjacency(n, &edges).expect("adj");
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");
        // 8 nodes × 2 neighbors = 16 non-zero entries
        assert_eq!(sparse.nnz(), 16);
        // Memory: much less than dense 8×8×8 = 512 bytes
        assert!(sparse.memory_bytes() < 512);
    }

    #[test]
    fn sparse_memory_scales() {
        // For a ring with N nodes and degree 2, sparse nnz = 2N
        // Dense memory = N² × 8, Sparse memory ≈ (N+1)*8 + 2N*16
        let n = 100;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push(((i + 1) % n, i, 1.0));
        }
        let dense = Matrix::from_adjacency(n, &edges).expect("adj");
        let sparse = SparseMatrix::from_dense(&dense, 1e-15).expect("sparse");

        let dense_mem = n * n * 8;
        let sparse_mem = sparse.memory_bytes();
        assert!(
            sparse_mem < dense_mem / 10,
            "sparse ({sparse_mem}) should be <10% of dense ({dense_mem})"
        );
    }

    #[test]
    fn from_triplets_basic() {
        let triplets = vec![(0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0), (2, 0, 4.0)];
        let sparse = SparseMatrix::from_triplets(3, 3, &triplets, 1e-15).expect("triplets");
        assert_eq!(sparse.nnz(), 4);
        assert!((sparse.get(0, 0).unwrap() - 1.0).abs() < 1e-15);
        assert!((sparse.get(0, 2).unwrap() - 3.0).abs() < 1e-15);
        assert!((sparse.get(1, 1).unwrap() - 2.0).abs() < 1e-15);
        assert!((sparse.get(2, 0).unwrap() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn from_triplets_out_of_bounds() {
        let triplets = vec![(5, 0, 1.0)];
        assert!(SparseMatrix::from_triplets(3, 3, &triplets, 1e-15).is_err());
    }
}
