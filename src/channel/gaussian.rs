use super::error::ChannelError;
use super::traits::ChannelModel;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Noise mode for the Gaussian channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseMode {
    /// Additive noise: output = input + N(0, σ²)
    Additive,
    /// Multiplicative noise: output = input * (1 + N(0, σ²))
    Multiplicative,
}

/// Gaussian noise channel with configurable noise mode.
///
/// In additive mode (default), adds independent N(0, σ²) noise to each sample.
/// In multiplicative mode, scales each sample by (1 + N(0, σ²)).
///
/// Uses a seeded RNG for deterministic, reproducible simulations.
pub struct GaussianChannel {
    /// Standard deviation of the noise.
    sigma: f64,
    /// Noise mode (additive or multiplicative).
    mode: NoiseMode,
    /// Normal distribution N(0, σ²).
    distribution: Normal<f64>,
    /// Seeded RNG for reproducibility.
    rng: rand::rngs::StdRng,
}

impl GaussianChannel {
    /// Create a new Gaussian channel with the given noise standard deviation.
    ///
    /// Uses additive noise mode by default.
    pub fn new(sigma: f64, rng_seed: u64) -> Result<Self, ChannelError> {
        if sigma < 0.0 || !sigma.is_finite() {
            return Err(ChannelError::InvalidParameter {
                name: "sigma".to_string(),
                reason: format!("must be non-negative and finite, got {sigma}"),
            });
        }

        let distribution = Normal::new(0.0, sigma).map_err(|e| ChannelError::InvalidParameter {
            name: "sigma".to_string(),
            reason: format!("failed to create normal distribution: {e}"),
        })?;

        Ok(Self {
            sigma,
            mode: NoiseMode::Additive,
            distribution,
            rng: rand::rngs::StdRng::seed_from_u64(rng_seed),
        })
    }

    /// Create a Gaussian channel with a specific noise mode.
    pub fn with_mode(sigma: f64, rng_seed: u64, mode: NoiseMode) -> Result<Self, ChannelError> {
        let mut ch = Self::new(sigma, rng_seed)?;
        ch.mode = mode;
        Ok(ch)
    }

    /// Get the noise standard deviation.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Get the noise mode.
    pub fn mode(&self) -> NoiseMode {
        self.mode
    }
}

impl ChannelModel for GaussianChannel {
    fn transmit(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), ChannelError> {
        if input.len() != output.len() {
            return Err(ChannelError::LengthMismatch {
                input_len: input.len(),
                output_len: output.len(),
            });
        }

        match self.mode {
            NoiseMode::Additive => {
                for (output_sample, &input_sample) in output.iter_mut().zip(input.iter()) {
                    let noise = self.distribution.sample(&mut self.rng);
                    *output_sample = input_sample + noise;
                }
            }
            NoiseMode::Multiplicative => {
                for (output_sample, &input_sample) in output.iter_mut().zip(input.iter()) {
                    let noise = self.distribution.sample(&mut self.rng);
                    *output_sample = input_sample * (1.0 + noise);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_additive_mean_preserving() {
        let mut ch = GaussianChannel::new(0.1, 42).expect("channel");
        let n = 100_000;
        let input = vec![5.0; n];
        let mut output = vec![0.0; n];

        ch.transmit(&input, &mut output).expect("transmit");

        let mean: f64 = output.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 5.0).abs() < 0.01,
            "mean should be close to input: got {mean}"
        );
    }

    #[test]
    fn gaussian_additive_variance() {
        let sigma = 0.5;
        let mut ch = GaussianChannel::new(sigma, 123).expect("channel");
        let n = 100_000;
        let input = vec![0.0; n];
        let mut output = vec![0.0; n];

        ch.transmit(&input, &mut output).expect("transmit");

        let mean: f64 = output.iter().sum::<f64>() / n as f64;
        let variance: f64 = output.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let expected_var = sigma * sigma;

        assert!(
            (variance - expected_var).abs() < 0.01,
            "variance should be close to σ²={expected_var}: got {variance}"
        );
    }

    #[test]
    fn gaussian_zero_noise_is_passthrough() {
        let mut ch = GaussianChannel::new(0.0, 0).expect("channel");
        let input = vec![1.0, -2.0, 3.5];
        let mut output = vec![0.0; 3];

        ch.transmit(&input, &mut output).expect("transmit");

        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-15, "zero noise should be passthrough");
        }
    }

    #[test]
    fn gaussian_multiplicative_mode() {
        let sigma = 0.1;
        let mut ch =
            GaussianChannel::with_mode(sigma, 42, NoiseMode::Multiplicative).expect("channel");
        let n = 100_000;
        let input = vec![10.0; n];
        let mut output = vec![0.0; n];

        ch.transmit(&input, &mut output).expect("transmit");

        // Mean should be close to input (E[x*(1+N(0,σ²))] = x)
        let mean: f64 = output.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 10.0).abs() < 0.1,
            "multiplicative mean should be close to input: got {mean}"
        );

        // Variance should be approximately x² * σ²
        let variance: f64 = output.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let expected_var = 100.0 * sigma * sigma; // x² * σ²
        assert!(
            (variance - expected_var).abs() < 0.1,
            "multiplicative variance should be close to x²σ²={expected_var}: got {variance}"
        );
    }

    #[test]
    fn gaussian_deterministic_with_seed() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out1 = vec![0.0; 5];
        let mut out2 = vec![0.0; 5];

        GaussianChannel::new(1.0, 999)
            .expect("ch1")
            .transmit(&input, &mut out1)
            .expect("t1");
        GaussianChannel::new(1.0, 999)
            .expect("ch2")
            .transmit(&input, &mut out2)
            .expect("t2");

        assert_eq!(out1, out2, "same seed should produce identical output");
    }

    #[test]
    fn gaussian_different_seeds_differ() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out1 = vec![0.0; 5];
        let mut out2 = vec![0.0; 5];

        GaussianChannel::new(1.0, 1)
            .expect("ch1")
            .transmit(&input, &mut out1)
            .expect("t1");
        GaussianChannel::new(1.0, 2)
            .expect("ch2")
            .transmit(&input, &mut out2)
            .expect("t2");

        assert_ne!(
            out1, out2,
            "different seeds should produce different output"
        );
    }

    #[test]
    fn gaussian_negative_sigma_error() {
        assert!(GaussianChannel::new(-1.0, 0).is_err());
    }

    #[test]
    fn gaussian_nan_sigma_error() {
        assert!(GaussianChannel::new(f64::NAN, 0).is_err());
    }

    #[test]
    fn gaussian_length_mismatch() {
        let mut ch = GaussianChannel::new(0.1, 0).expect("channel");
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 3];
        assert!(ch.transmit(&input, &mut output).is_err());
    }
}
