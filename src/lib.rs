//! # Cluster Shift Keying
//!
//! Rust implementation of Cluster Shift Keying (CLSK), a chaos-based
//! communication scheme encoding information into spatio-temporal
//! synchronization patterns of coupled chaotic networks.
//!
//! Based on: Sarı & Günel, "Cluster Shift Keying: Covert Transmission
//! of Information via Cluster Synchronization in Chaotic Networks",
//! Physica Scripta 99 (2024) 035204.

pub mod channel;
pub mod codec;
pub mod dynamics;
pub mod graph;
pub mod linalg;
pub mod metrics;
pub mod pipeline;
pub mod sync;
pub mod utils;

// Re-export core traits at crate root for ergonomic imports.
pub use channel::ChannelModel;
pub use codec::{Decoder, Encoder};
pub use dynamics::DynamicalSystem;

#[cfg(test)]
mod tests {
    //! Crate-level smoke tests verifying the module structure compiles
    //! and core traits are object-safe.

    use super::*;

    /// Verify `DynamicalSystem` is object-safe (can be used as `dyn`).
    #[test]
    fn dynamical_system_is_object_safe() {
        fn _assert_object_safe(_: &dyn DynamicalSystem) {}
    }

    /// Verify `ChannelModel` is object-safe.
    #[test]
    fn channel_model_is_object_safe() {
        fn _assert_object_safe(_: &dyn ChannelModel) {}
    }

    /// Verify all error types implement `std::error::Error + Send + Sync`.
    #[test]
    fn error_types_are_send_sync() {
        fn _assert_send_sync<T: std::error::Error + Send + Sync>() {}
        _assert_send_sync::<dynamics::DynamicsError>();
        _assert_send_sync::<codec::CodecError>();
        _assert_send_sync::<channel::ChannelError>();
        _assert_send_sync::<linalg::LinalgError>();
        _assert_send_sync::<graph::GraphError>();
        _assert_send_sync::<sync::SyncError>();
        _assert_send_sync::<metrics::MetricsError>();
        _assert_send_sync::<pipeline::PipelineError>();
    }

    /// Verify error type conversions work (from-chain).
    #[test]
    fn error_conversions() {
        // DynamicsError -> CodecError
        let dyn_err = dynamics::DynamicsError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        let codec_err: codec::CodecError = dyn_err.into();
        assert!(codec_err.to_string().contains("dimension mismatch"));

        // LinalgError -> GraphError
        let lin_err = linalg::LinalgError::NotSquare { rows: 3, cols: 4 };
        let graph_err: graph::GraphError = lin_err.into();
        assert!(graph_err.to_string().contains("not square"));

        // DynamicsError -> SyncError
        let dyn_err2 = dynamics::DynamicsError::IntegrationFailed {
            reason: "diverged".to_string(),
        };
        let sync_err: sync::SyncError = dyn_err2.into();
        assert!(sync_err.to_string().contains("diverged"));
    }
}
