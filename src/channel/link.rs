use super::error::ChannelError;
use super::traits::ChannelModel;

/// Represents the physical channel links (L_c) between transmitter and receiver.
///
/// Applies a `ChannelModel` to each channel link signal independently.
/// This models the physical link between transmitter and receiver subnetworks,
/// where each link carries one signal component (e.g., the y-component of a
/// channel link node).
pub struct ChannelLink {
    /// Number of channel link signals.
    num_links: usize,
    /// Scratch buffer for single-signal transmission.
    scratch: Vec<f64>,
}

impl ChannelLink {
    /// Create a new channel link with the given number of links.
    pub fn new(num_links: usize) -> Result<Self, ChannelError> {
        if num_links == 0 {
            return Err(ChannelError::InvalidParameter {
                name: "num_links".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }

        Ok(Self {
            num_links,
            scratch: Vec::new(),
        })
    }

    /// Get the number of channel links.
    pub fn num_links(&self) -> usize {
        self.num_links
    }

    /// Transmit all channel link signals through the channel model.
    ///
    /// `input_signals[link_index]` and `output_signals[link_index]` are the
    /// signal vectors for each link. Each pair must have the same length.
    pub fn transmit_all(
        &mut self,
        input_signals: &[Vec<f64>],
        output_signals: &mut [Vec<f64>],
        channel: &mut dyn ChannelModel,
    ) -> Result<(), ChannelError> {
        if input_signals.len() != self.num_links {
            return Err(ChannelError::LengthMismatch {
                input_len: input_signals.len(),
                output_len: self.num_links,
            });
        }
        if output_signals.len() != self.num_links {
            return Err(ChannelError::LengthMismatch {
                input_len: self.num_links,
                output_len: output_signals.len(),
            });
        }

        for link_idx in 0..self.num_links {
            let input = &input_signals[link_idx];
            let output = &mut output_signals[link_idx];

            // Ensure output has the right length
            output.resize(input.len(), 0.0);
            channel.transmit(input, output)?;
        }

        Ok(())
    }

    /// Transmit a single time step's worth of signals through the channel.
    ///
    /// `input[link_index]` is the signal value at the current time step
    /// for each channel link. Returns the noisy output values.
    pub fn transmit_step(
        &mut self,
        input: &[f64],
        channel: &mut dyn ChannelModel,
    ) -> Result<Vec<f64>, ChannelError> {
        if input.len() != self.num_links {
            return Err(ChannelError::LengthMismatch {
                input_len: input.len(),
                output_len: self.num_links,
            });
        }

        self.scratch.resize(self.num_links, 0.0);
        channel.transmit(input, &mut self.scratch)?;
        Ok(self.scratch.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::gaussian::GaussianChannel;
    use crate::channel::ideal::IdealChannel;

    #[test]
    fn channel_link_creation() {
        let link = ChannelLink::new(2).expect("link");
        assert_eq!(link.num_links(), 2);
    }

    #[test]
    fn channel_link_zero_links_error() {
        assert!(ChannelLink::new(0).is_err());
    }

    #[test]
    fn transmit_all_ideal() {
        let mut link = ChannelLink::new(2).expect("link");
        let mut ch = IdealChannel::new();
        let input = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut output = vec![vec![]; 2];

        link.transmit_all(&input, &mut output, &mut ch)
            .expect("transmit");

        assert_eq!(output[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(output[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn transmit_all_gaussian() {
        let mut link = ChannelLink::new(2).expect("link");
        let mut ch = GaussianChannel::new(0.1, 42).expect("channel");
        let input = vec![vec![1.0; 1000], vec![2.0; 1000]];
        let mut output = vec![vec![]; 2];

        link.transmit_all(&input, &mut output, &mut ch)
            .expect("transmit");

        // Check that noise was added (output differs from input)
        let diff0: f64 = input[0]
            .iter()
            .zip(output[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        assert!(diff0 > 0.0, "noise should have been added to link 0");

        // Check mean is close to input
        let mean0: f64 = output[0].iter().sum::<f64>() / 1000.0;
        assert!(
            (mean0 - 1.0).abs() < 0.05,
            "mean should be close to 1.0: got {mean0}"
        );
    }

    #[test]
    fn transmit_all_wrong_input_count() {
        let mut link = ChannelLink::new(2).expect("link");
        let mut ch = IdealChannel::new();
        let input = vec![vec![1.0]]; // only 1, need 2
        let mut output = vec![vec![]; 2];

        assert!(link.transmit_all(&input, &mut output, &mut ch).is_err());
    }

    #[test]
    fn transmit_all_wrong_output_count() {
        let mut link = ChannelLink::new(2).expect("link");
        let mut ch = IdealChannel::new();
        let input = vec![vec![1.0], vec![2.0]];
        let mut output = vec![vec![]]; // only 1, need 2

        assert!(link.transmit_all(&input, &mut output, &mut ch).is_err());
    }

    #[test]
    fn transmit_step_ideal() {
        let mut link = ChannelLink::new(3).expect("link");
        let mut ch = IdealChannel::new();

        let input = vec![1.0, 2.0, 3.0];
        let output = link.transmit_step(&input, &mut ch).expect("step");

        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn transmit_step_wrong_count() {
        let mut link = ChannelLink::new(2).expect("link");
        let mut ch = IdealChannel::new();

        let input = vec![1.0, 2.0, 3.0]; // 3 instead of 2
        assert!(link.transmit_step(&input, &mut ch).is_err());
    }

    #[test]
    fn transmit_step_gaussian_adds_noise() {
        let mut link = ChannelLink::new(1).expect("link");
        let mut ch = GaussianChannel::new(1.0, 42).expect("channel");

        let input = vec![0.0];
        let output = link.transmit_step(&input, &mut ch).expect("step");

        // With σ=1.0, very unlikely to get exactly 0.0
        assert!(output[0].abs() > 1e-15, "noise should have been added");
    }
}
