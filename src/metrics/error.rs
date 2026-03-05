use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("empty input: cannot compute metrics on zero-length data")]
    EmptyInput,

    #[error("length mismatch: transmitted {tx_len} symbols, received {rx_len}")]
    LengthMismatch { tx_len: usize, rx_len: usize },

    #[error("invalid threshold: {value} (must be positive)")]
    InvalidThreshold { value: f64 },

    #[error("monte carlo simulation failed: {reason}")]
    SimulationFailed { reason: String },
}
