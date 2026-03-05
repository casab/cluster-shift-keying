use super::error::ChannelError;
use super::traits::ChannelModel;

/// Ideal (noiseless) channel: output = input.
///
/// Used as a baseline for testing and for noiseless simulations.
#[derive(Debug, Clone, Default)]
pub struct IdealChannel;

impl IdealChannel {
    /// Create a new ideal channel.
    pub fn new() -> Self {
        Self
    }
}

impl ChannelModel for IdealChannel {
    fn transmit(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), ChannelError> {
        if input.len() != output.len() {
            return Err(ChannelError::LengthMismatch {
                input_len: input.len(),
                output_len: output.len(),
            });
        }
        output.copy_from_slice(input);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideal_passthrough() {
        let mut ch = IdealChannel::new();
        let input = vec![1.0, 2.0, 3.0, -4.5, 0.0];
        let mut output = vec![0.0; 5];
        ch.transmit(&input, &mut output).expect("transmit");
        assert_eq!(input, output);
    }

    #[test]
    fn ideal_empty() {
        let mut ch = IdealChannel::new();
        let input: Vec<f64> = vec![];
        let mut output: Vec<f64> = vec![];
        ch.transmit(&input, &mut output).expect("transmit empty");
    }

    #[test]
    fn ideal_length_mismatch() {
        let mut ch = IdealChannel::new();
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 3];
        assert!(ch.transmit(&input, &mut output).is_err());
    }

    #[test]
    fn ideal_preserves_special_values() {
        let mut ch = IdealChannel::new();
        let input = vec![f64::MIN, f64::MAX, f64::EPSILON, -0.0];
        let mut output = vec![0.0; 4];
        ch.transmit(&input, &mut output).expect("transmit");
        for (i, o) in input.iter().zip(output.iter()) {
            assert!(i.to_bits() == o.to_bits());
        }
    }
}
