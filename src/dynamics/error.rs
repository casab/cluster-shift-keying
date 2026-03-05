use thiserror::Error;

#[derive(Debug, Error)]
pub enum DynamicsError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("integration step failed: {reason}")]
    IntegrationFailed { reason: String },

    #[error("jacobian not implemented for system `{system}`")]
    JacobianNotImplemented { system: String },

    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: String, reason: String },
}
