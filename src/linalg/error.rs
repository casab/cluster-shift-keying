use thiserror::Error;

#[derive(Debug, Error)]
pub enum LinalgError {
    #[error("matrix dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("matrix is not square: {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("eigendecomposition failed: {reason}")]
    EigenDecompositionFailed { reason: String },

    #[error("block diagonalization failed: {reason}")]
    BlockDiagFailed { reason: String },
}
