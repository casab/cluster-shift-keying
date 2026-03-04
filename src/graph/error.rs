use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("invalid node count: {count} (minimum is {minimum})")]
    InvalidNodeCount { count: usize, minimum: usize },

    #[error("adjacency matrix is not square: {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("adjacency matrix is not symmetric at ({row}, {col})")]
    NotSymmetric { row: usize, col: usize },

    #[error("partition is invalid: {reason}")]
    InvalidPartition { reason: String },

    #[error("no valid coupling strength range found for the given cluster pattern")]
    NoCouplingRange,

    #[error(transparent)]
    Linalg(#[from] crate::linalg::LinalgError),
}
