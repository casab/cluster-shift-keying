use thiserror::Error;

#[derive(Debug, Error)]
pub enum SyncError {
    #[error("network has {got} nodes but expected {expected}")]
    NodeCountMismatch { expected: usize, got: usize },

    #[error("coupling strength {epsilon} is outside valid range [{min}, {max}]")]
    CouplingOutOfRange { epsilon: f64, min: f64, max: f64 },

    #[error("master stability function computation failed: {reason}")]
    MsfFailed { reason: String },

    #[error("cluster pattern did not converge within {steps} steps")]
    ConvergenceFailed { steps: usize },

    #[error(transparent)]
    Dynamics(#[from] crate::dynamics::DynamicsError),

    #[error(transparent)]
    Graph(#[from] crate::graph::GraphError),

    #[error(transparent)]
    Linalg(#[from] crate::linalg::LinalgError),
}
