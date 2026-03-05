/// Trait for channel models that transform signals between transmitter and receiver.
///
/// Implementations model physical channel effects (noise, attenuation, etc.)
/// applied to the signals traversing channel links `L_c`.
pub trait ChannelModel: Send + Sync {
    /// Transform `input` signal through the channel, writing result to `output`.
    ///
    /// Both slices must have the same length.
    fn transmit(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), ChannelError>;
}

use super::error::ChannelError;
