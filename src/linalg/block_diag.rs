use super::eigen::symmetric_eigen;
use super::error::LinalgError;
use super::matrix::Matrix;

/// Result of simultaneous block-diagonalization.
///
/// Given commuting symmetric matrices A and B, find an orthogonal matrix P
/// such that P^T A P and P^T B P are both (approximately) block-diagonal.
#[derive(Debug, Clone)]
pub struct BlockDiagResult {
    /// The transformation matrix P (columns are the shared eigenbasis).
    pub transform: Matrix,
    /// The block-diagonalized version of the first matrix (P^T A P).
    pub diag_a: Matrix,
    /// The block-diagonalized version of the second matrix (P^T B P).
    pub diag_b: Matrix,
}

/// Simultaneously block-diagonalize two commuting symmetric matrices.
///
/// For the CLSK pipeline, this is used to decompose the coupling matrix
/// and inner coupling into synchronous/transverse modes.
///
/// **Algorithm**: Eigendecompose A to get P, then compute P^T B P.
/// If A and B commute and are both symmetric, P simultaneously diagonalizes both.
///
/// Returns an error if either matrix is not square, or if dimensions don't match.
pub fn simultaneous_block_diag(a: &Matrix, b: &Matrix) -> Result<BlockDiagResult, LinalgError> {
    a.ensure_square()?;
    b.ensure_square()?;

    if a.nrows() != b.nrows() {
        return Err(LinalgError::DimensionMismatch {
            expected: format!("{}x{}", a.nrows(), a.ncols()),
            got: format!("{}x{}", b.nrows(), b.ncols()),
        });
    }

    let eig = symmetric_eigen(a)?;
    let p = eig
        .eigenvectors
        .ok_or_else(|| LinalgError::BlockDiagFailed {
            reason: "eigendecomposition did not produce eigenvectors".to_string(),
        })?;

    let p_transpose = p.transpose();

    // P^T A P should be diagonal (eigenvalues of A)
    let diag_a = p_transpose.mul(a)?.mul(&p)?;
    // P^T B P should also be diagonal if A, B commute
    let diag_b = p_transpose.mul(b)?.mul(&p)?;

    Ok(BlockDiagResult {
        transform: p,
        diag_a,
        diag_b,
    })
}

/// Check if a matrix is approximately diagonal (off-diagonal elements < tol).
pub fn is_approx_diagonal(m: &Matrix, tol: f64) -> bool {
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if i != j {
                if let Ok(val) = m.get(i, j) {
                    if val.abs() > tol {
                        return false;
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_diag_commuting_diagonal_matrices() {
        // Two diagonal matrices trivially commute
        let a = Matrix::from_diagonal(&[1.0, 2.0, 3.0]);
        let b = Matrix::from_diagonal(&[4.0, 5.0, 6.0]);

        let result = simultaneous_block_diag(&a, &b).expect("block diag of diagonal matrices");

        // Both should remain diagonal
        assert!(
            is_approx_diagonal(&result.diag_a, 1e-10),
            "diag_a not diagonal:\n{}",
            result.diag_a
        );
        assert!(
            is_approx_diagonal(&result.diag_b, 1e-10),
            "diag_b not diagonal:\n{}",
            result.diag_b
        );

        // Eigenvalues of A on diagonal (sorted ascending)
        let da = result.diag_a.diagonal();
        assert!((da[0] - 1.0).abs() < 1e-10);
        assert!((da[1] - 2.0).abs() < 1e-10);
        assert!((da[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn block_diag_commuting_symmetric() {
        // A = [[2,1],[1,2]], B = [[3,1],[1,3]] — these commute since AB = BA
        // for matrices of the form aI + bJ where J = [[0,1],[1,0]]
        let a = Matrix::from_row_major(2, 2, &[2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Matrix::from_row_major(2, 2, &[3.0, 1.0, 1.0, 3.0]).unwrap();

        let result = simultaneous_block_diag(&a, &b).expect("commuting symmetric");

        assert!(
            is_approx_diagonal(&result.diag_a, 1e-10),
            "diag_a:\n{}",
            result.diag_a
        );
        assert!(
            is_approx_diagonal(&result.diag_b, 1e-10),
            "diag_b:\n{}",
            result.diag_b
        );

        // A has eigenvalues 1, 3; B has eigenvalues 2, 4
        let da = result.diag_a.diagonal();
        let db = result.diag_b.diagonal();

        let mut da_sorted = da.clone();
        da_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((da_sorted[0] - 1.0).abs() < 1e-10);
        assert!((da_sorted[1] - 3.0).abs() < 1e-10);

        let mut db_sorted = db.clone();
        db_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((db_sorted[0] - 2.0).abs() < 1e-10);
        assert!((db_sorted[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn block_diag_roundtrip() {
        // Verify P^T A P = D and P D P^T = A
        let a =
            Matrix::from_row_major(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
        let b = Matrix::from_diagonal(&[1.0, 1.0, 1.0]); // I commutes with everything

        let result = simultaneous_block_diag(&a, &b).expect("roundtrip");

        // Reconstruct A = P * diag_a * P^T
        let p = &result.transform;
        let pt = p.transpose();
        let reconstructed = p.mul(&result.diag_a).unwrap().mul(&pt).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let orig = a.get(i, j).unwrap();
                let recon = reconstructed.get(i, j).unwrap();
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "roundtrip mismatch at ({i},{j}): {orig} vs {recon}"
                );
            }
        }
    }

    #[test]
    fn block_diag_dimension_mismatch() {
        let a = Matrix::identity(3);
        let b = Matrix::identity(4);
        assert!(simultaneous_block_diag(&a, &b).is_err());
    }

    #[test]
    fn block_diag_non_square_error() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        assert!(simultaneous_block_diag(&a, &b).is_err());
    }

    #[test]
    fn is_approx_diagonal_true() {
        let m = Matrix::from_diagonal(&[1.0, 2.0, 3.0]);
        assert!(is_approx_diagonal(&m, 1e-15));
    }

    #[test]
    fn is_approx_diagonal_false() {
        let m = Matrix::from_row_major(2, 2, &[1.0, 0.5, 0.5, 2.0]).unwrap();
        assert!(!is_approx_diagonal(&m, 0.1));
    }
}
