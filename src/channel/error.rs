use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChannelError {
    #[error("length mismatch: input has {input_len} samples, output has {output_len}")]
    LengthMismatch { input_len: usize, output_len: usize },

    #[error("invalid noise parameter `{name}`: {reason}")]
    InvalidParameter { name: String, reason: String },
}
