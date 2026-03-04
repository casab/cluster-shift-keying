use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("configuration error: {reason}")]
    ConfigError { reason: String },

    #[error("transmission failed: {reason}")]
    TransmissionFailed { reason: String },

    #[error("reception failed: {reason}")]
    ReceptionFailed { reason: String },

    #[error(transparent)]
    Codec(#[from] crate::codec::CodecError),

    #[error(transparent)]
    Channel(#[from] crate::channel::ChannelError),

    #[error(transparent)]
    Sync(#[from] crate::sync::SyncError),

    #[error(transparent)]
    Metrics(#[from] crate::metrics::MetricsError),
}
