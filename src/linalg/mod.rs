pub mod block_diag;
pub mod eigen;
pub mod error;
pub mod matrix;
pub mod sparse;

pub use block_diag::{is_approx_diagonal, simultaneous_block_diag, BlockDiagResult};
pub use eigen::{general_eigen, symmetric_eigen, EigenDecomposition};
pub use error::LinalgError;
pub use matrix::Matrix;
pub use sparse::SparseMatrix;
