use super::error::LinalgError;
use super::matrix::Matrix;
use nalgebra::DMatrix;

/// Result of an eigendecomposition: eigenvalues (possibly complex) and eigenvectors.
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    /// Eigenvalues as (real, imaginary) pairs, sorted by real part ascending.
    pub eigenvalues: Vec<(f64, f64)>,
    /// Eigenvectors as columns (stored in the same order as eigenvalues).
    /// For complex eigenvalues, consecutive columns represent the real and
    /// imaginary parts of the complex eigenvector pair.
    pub eigenvectors: Option<Matrix>,
}

impl EigenDecomposition {
    /// Return just the real parts of the eigenvalues.
    pub fn real_eigenvalues(&self) -> Vec<f64> {
        self.eigenvalues.iter().map(|(re, _)| *re).collect()
    }

    /// Return just the imaginary parts of the eigenvalues.
    pub fn imag_eigenvalues(&self) -> Vec<f64> {
        self.eigenvalues.iter().map(|(_, im)| *im).collect()
    }

    /// Check if all eigenvalues are real (imaginary part < tol).
    pub fn is_real(&self, tol: f64) -> bool {
        self.eigenvalues.iter().all(|(_, im)| im.abs() < tol)
    }
}

/// Compute eigenvalues of a real symmetric matrix.
///
/// Returns eigenvalues sorted ascending. Eigenvectors are the columns of
/// the returned matrix (orthonormal).
pub fn symmetric_eigen(m: &Matrix) -> Result<EigenDecomposition, LinalgError> {
    m.ensure_square()?;

    let inner = m.as_nalgebra();
    let se = inner.clone().symmetric_eigen();

    // nalgebra returns eigenvalues unsorted — sort by value ascending
    let n = se.eigenvalues.len();
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, se.eigenvalues[i])).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<(f64, f64)> = indexed.iter().map(|(_, v)| (*v, 0.0)).collect();

    // Reorder eigenvectors
    let mut evec_data = DMatrix::zeros(n, n);
    for (new_col, (old_col, _)) in indexed.iter().enumerate() {
        for row in 0..n {
            evec_data[(row, new_col)] = se.eigenvectors[(row, *old_col)];
        }
    }

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors: Some(Matrix::from_nalgebra(evec_data)),
    })
}

/// Compute eigenvalues of a general (possibly non-symmetric) real matrix.
///
/// Uses Schur decomposition to extract eigenvalues. Eigenvectors are not
/// computed (set to `None`) — use `symmetric_eigen` when eigenvectors are needed.
pub fn general_eigen(m: &Matrix) -> Result<EigenDecomposition, LinalgError> {
    m.ensure_square()?;

    let inner = m.as_nalgebra();
    let schur = inner.clone().schur();
    let (_, t) = schur.unpack();

    let n = t.nrows();
    let mut eigenvalues: Vec<(f64, f64)> = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1, i)].abs() > 1e-12 {
            // 2x2 block on diagonal → complex conjugate pair
            let a = t[(i, i)];
            let b = t[(i, i + 1)];
            let c = t[(i + 1, i)];
            let d = t[(i + 1, i + 1)];
            let trace = a + d;
            let det = a * d - b * c;
            let disc = trace * trace - 4.0 * det;

            if disc < 0.0 {
                let real = trace / 2.0;
                let imag = (-disc).sqrt() / 2.0;
                eigenvalues.push((real, imag));
                eigenvalues.push((real, -imag));
            } else {
                // Real eigenvalues from 2x2 block (unusual but possible)
                let sqrt_disc = disc.sqrt();
                eigenvalues.push(((trace + sqrt_disc) / 2.0, 0.0));
                eigenvalues.push(((trace - sqrt_disc) / 2.0, 0.0));
            }
            i += 2;
        } else {
            eigenvalues.push((t[(i, i)], 0.0));
            i += 1;
        }
    }

    // Sort by real part ascending, then by imaginary part ascending
    eigenvalues.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_identity_eigenvalues() {
        let id = Matrix::identity(4);
        let eig = symmetric_eigen(&id).expect("identity eigen");
        for (re, im) in &eig.eigenvalues {
            assert!((*re - 1.0).abs() < 1e-12, "eigenvalue real part: {re}");
            assert!(im.abs() < 1e-12, "eigenvalue imag part: {im}");
        }
        assert!(eig.is_real(1e-12));
    }

    #[test]
    fn symmetric_diagonal_eigenvalues() {
        let m = Matrix::from_diagonal(&[3.0, 1.0, 2.0]);
        let eig = symmetric_eigen(&m).expect("diagonal eigen");
        let reals = eig.real_eigenvalues();
        // Should be sorted ascending: 1, 2, 3
        assert!((reals[0] - 1.0).abs() < 1e-12);
        assert!((reals[1] - 2.0).abs() < 1e-12);
        assert!((reals[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_known_matrix() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let m = Matrix::from_row_major(2, 2, &[2.0, 1.0, 1.0, 2.0]).unwrap();
        let eig = symmetric_eigen(&m).expect("2x2 symmetric eigen");
        let reals = eig.real_eigenvalues();
        assert!((reals[0] - 1.0).abs() < 1e-12);
        assert!((reals[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_eigenvectors_orthonormal() {
        let m =
            Matrix::from_row_major(3, 3, &[2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
        let eig = symmetric_eigen(&m).expect("3x3 symmetric eigen");
        let vecs = eig.eigenvectors.as_ref().expect("should have eigenvectors");

        // Check columns are orthonormal
        for i in 0..3 {
            // Norm ≈ 1
            let norm: f64 = (0..3)
                .map(|r| vecs.get(r, i).unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "eigenvector {i} norm = {norm}");

            // Orthogonal to other columns
            for j in (i + 1)..3 {
                let dot: f64 = (0..3)
                    .map(|r| vecs.get(r, i).unwrap() * vecs.get(r, j).unwrap())
                    .sum();
                assert!(dot.abs() < 1e-10, "dot({i},{j}) = {dot}");
            }
        }
    }

    #[test]
    fn symmetric_non_square_error() {
        let m = Matrix::zeros(2, 3);
        assert!(symmetric_eigen(&m).is_err());
    }

    #[test]
    fn general_identity_eigenvalues() {
        let id = Matrix::identity(3);
        let eig = general_eigen(&id).expect("identity general eigen");
        for (re, im) in &eig.eigenvalues {
            assert!((*re - 1.0).abs() < 1e-12);
            assert!(im.abs() < 1e-12);
        }
    }

    #[test]
    fn general_non_symmetric_real_eigenvalues() {
        // [[0, 1], [-2, -3]] has eigenvalues -1 and -2
        let m = Matrix::from_row_major(2, 2, &[0.0, 1.0, -2.0, -3.0]).unwrap();
        let eig = general_eigen(&m).expect("general eigen");
        let reals = eig.real_eigenvalues();
        assert!(
            (reals[0] - (-2.0)).abs() < 1e-10,
            "first eigenvalue: {}",
            reals[0]
        );
        assert!(
            (reals[1] - (-1.0)).abs() < 1e-10,
            "second eigenvalue: {}",
            reals[1]
        );
    }

    #[test]
    fn general_complex_eigenvalues() {
        // [[0, -1], [1, 0]] has eigenvalues ±i
        let m = Matrix::from_row_major(2, 2, &[0.0, -1.0, 1.0, 0.0]).unwrap();
        let eig = general_eigen(&m).expect("rotation eigen");
        assert_eq!(eig.eigenvalues.len(), 2);
        // Both have real part 0
        for (re, _) in &eig.eigenvalues {
            assert!(re.abs() < 1e-10, "real part should be ~0, got {re}");
        }
        // Imaginary parts: -1 and +1
        let mut imags: Vec<f64> = eig.eigenvalues.iter().map(|(_, im)| *im).collect();
        imags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((imags[0] - (-1.0)).abs() < 1e-10);
        assert!((imags[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn general_circulant_3() {
        // Circulant matrix for 3-cycle: [[0,1,0],[0,0,1],[1,0,0]]
        // Eigenvalues: 1, e^{2πi/3}, e^{-2πi/3} = 1, -1/2 ± i√3/2
        let m =
            Matrix::from_row_major(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
        let eig = general_eigen(&m).expect("circulant eigen");
        assert_eq!(eig.eigenvalues.len(), 3);

        // One eigenvalue should be real = 1
        let has_one = eig
            .eigenvalues
            .iter()
            .any(|(re, im)| (*re - 1.0).abs() < 1e-10 && im.abs() < 1e-10);
        assert!(
            has_one,
            "should have eigenvalue 1, got {:?}",
            eig.eigenvalues
        );

        // Two complex eigenvalues with real part = -0.5
        let complex_count = eig
            .eigenvalues
            .iter()
            .filter(|(re, im)| (*re - (-0.5)).abs() < 1e-10 && im.abs() > 0.1)
            .count();
        assert_eq!(
            complex_count, 2,
            "should have 2 complex eigenvalues with re=-0.5"
        );
    }

    #[test]
    fn general_non_square_error() {
        let m = Matrix::zeros(2, 3);
        assert!(general_eigen(&m).is_err());
    }
}
