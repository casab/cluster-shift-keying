use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("unknown symbol: {symbol}")]
    UnknownSymbol { symbol: usize },

    #[error("covertness condition violated: nodes {node_a} and {node_b} share a cluster in symbol {symbol}")]
    CovertnessViolation {
        node_a: usize,
        node_b: usize,
        symbol: usize,
    },

    #[error("symbol map is empty — at least 2 symbols required")]
    EmptySymbolMap,

    #[error("invalid symbol map: {reason}")]
    InvalidSymbolMap { reason: String },

    #[error("detection failed: {reason}")]
    DetectionFailed { reason: String },

    #[error("framing error: {reason}")]
    FramingError { reason: String },

    #[error(transparent)]
    Dynamics(#[from] crate::dynamics::DynamicsError),

    #[error(transparent)]
    Sync(#[from] crate::sync::SyncError),

    #[error(transparent)]
    Graph(#[from] crate::graph::GraphError),

    #[error(transparent)]
    Metrics(#[from] crate::metrics::MetricsError),
}
