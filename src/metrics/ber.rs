use super::error::MetricsError;

/// Bit Error Rate (BER) evaluator.
///
/// Computes symbol-level and bit-level error rates between transmitted
/// and received sequences.
pub struct BerEvaluator;

impl BerEvaluator {
    /// Compute symbol error rate (SER) between transmitted and received symbols.
    ///
    /// Returns the fraction of symbols that differ.
    /// Returns 0.0 for empty sequences (both must be empty).
    pub fn evaluate(tx_symbols: &[usize], rx_symbols: &[usize]) -> Result<f64, MetricsError> {
        if tx_symbols.len() != rx_symbols.len() {
            return Err(MetricsError::LengthMismatch {
                tx_len: tx_symbols.len(),
                rx_len: rx_symbols.len(),
            });
        }

        if tx_symbols.is_empty() {
            return Ok(0.0);
        }

        let errors = tx_symbols
            .iter()
            .zip(rx_symbols.iter())
            .filter(|(tx, rx)| tx != rx)
            .count();

        Ok(errors as f64 / tx_symbols.len() as f64)
    }

    /// Compute bit error rate (BER) between transmitted and received bit sequences.
    ///
    /// Each element should be 0 or 1. Returns the fraction of bits that differ.
    pub fn evaluate_bits(tx_bits: &[u8], rx_bits: &[u8]) -> Result<f64, MetricsError> {
        if tx_bits.len() != rx_bits.len() {
            return Err(MetricsError::LengthMismatch {
                tx_len: tx_bits.len(),
                rx_len: rx_bits.len(),
            });
        }

        if tx_bits.is_empty() {
            return Ok(0.0);
        }

        let errors = tx_bits
            .iter()
            .zip(rx_bits.iter())
            .filter(|(tx, rx)| tx != rx)
            .count();

        Ok(errors as f64 / tx_bits.len() as f64)
    }

    /// Compute BER from symbol sequences, converting symbols to bits.
    ///
    /// `bits_per_symbol` determines how many bits each symbol represents.
    /// Symbols are converted to binary using Gray coding for `bits_per_symbol <= 1`,
    /// or natural binary for higher-order modulation.
    pub fn evaluate_symbol_to_bit(
        tx_symbols: &[usize],
        rx_symbols: &[usize],
        bits_per_symbol: usize,
    ) -> Result<f64, MetricsError> {
        if tx_symbols.len() != rx_symbols.len() {
            return Err(MetricsError::LengthMismatch {
                tx_len: tx_symbols.len(),
                rx_len: rx_symbols.len(),
            });
        }

        if tx_symbols.is_empty() {
            return Ok(0.0);
        }

        if bits_per_symbol == 0 {
            return Err(MetricsError::SimulationFailed {
                reason: "bits_per_symbol must be at least 1".to_string(),
            });
        }

        let mut total_bit_errors = 0usize;
        let total_bits = tx_symbols.len() * bits_per_symbol;

        for (&tx, &rx) in tx_symbols.iter().zip(rx_symbols.iter()) {
            // XOR to find differing bits, count them
            let diff = tx ^ rx;
            total_bit_errors += diff.count_ones() as usize;
        }

        Ok(total_bit_errors as f64 / total_bits as f64)
    }
}

/// Result of a single BER trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Number of symbols transmitted.
    pub num_symbols: usize,
    /// Number of symbol errors.
    pub symbol_errors: usize,
    /// Symbol error rate.
    pub ser: f64,
}

impl TrialResult {
    /// Create a new trial result from tx/rx symbol sequences.
    pub fn from_sequences(tx: &[usize], rx: &[usize]) -> Result<Self, MetricsError> {
        let ser = BerEvaluator::evaluate(tx, rx)?;
        let symbol_errors = tx.iter().zip(rx.iter()).filter(|(a, b)| a != b).count();
        Ok(Self {
            num_symbols: tx.len(),
            symbol_errors,
            ser,
        })
    }
}

/// A single point on the BER curve.
#[derive(Debug, Clone)]
pub struct BerPoint {
    /// Noise standard deviation σ.
    pub sigma: f64,
    /// Mean BER across all trials.
    pub mean_ber: f64,
    /// Lower bound of 95% confidence interval.
    pub ci_low: f64,
    /// Upper bound of 95% confidence interval.
    pub ci_high: f64,
    /// Number of trials run.
    pub num_trials: usize,
}

/// A BER curve: collection of BER points across noise levels.
pub type BerCurve = Vec<BerPoint>;

/// Compute 95% confidence interval for a proportion (Wald interval).
///
/// Returns (ci_low, ci_high) clamped to [0, 1].
pub fn confidence_interval_95(mean: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    // z_{0.975} ≈ 1.96
    let z = 1.96;
    let se = (mean * (1.0 - mean) / n as f64).sqrt();
    let low = (mean - z * se).max(0.0);
    let high = (mean + z * se).min(1.0);
    (low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ber_identical_symbols() {
        let tx = vec![0, 1, 0, 1, 0];
        let rx = vec![0, 1, 0, 1, 0];
        let ber = BerEvaluator::evaluate(&tx, &rx).expect("evaluate");
        assert!((ber - 0.0).abs() < 1e-15);
    }

    #[test]
    fn ber_all_wrong() {
        let tx = vec![0, 0, 0, 0];
        let rx = vec![1, 1, 1, 1];
        let ber = BerEvaluator::evaluate(&tx, &rx).expect("evaluate");
        assert!((ber - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ber_half_wrong() {
        let tx = vec![0, 1, 0, 1];
        let rx = vec![0, 0, 0, 0];
        let ber = BerEvaluator::evaluate(&tx, &rx).expect("evaluate");
        assert!((ber - 0.5).abs() < 1e-15);
    }

    #[test]
    fn ber_empty_sequences() {
        let ber = BerEvaluator::evaluate(&[], &[]).expect("evaluate");
        assert!((ber - 0.0).abs() < 1e-15);
    }

    #[test]
    fn ber_length_mismatch() {
        let tx = vec![0, 1];
        let rx = vec![0, 1, 0];
        assert!(BerEvaluator::evaluate(&tx, &rx).is_err());
    }

    #[test]
    fn bit_ber_identical() {
        let tx = vec![0u8, 1, 0, 1, 1, 0];
        let rx = vec![0u8, 1, 0, 1, 1, 0];
        let ber = BerEvaluator::evaluate_bits(&tx, &rx).expect("evaluate");
        assert!((ber - 0.0).abs() < 1e-15);
    }

    #[test]
    fn bit_ber_all_flipped() {
        let tx = vec![0u8, 0, 0, 0];
        let rx = vec![1u8, 1, 1, 1];
        let ber = BerEvaluator::evaluate_bits(&tx, &rx).expect("evaluate");
        assert!((ber - 1.0).abs() < 1e-15);
    }

    #[test]
    fn bit_ber_length_mismatch() {
        assert!(BerEvaluator::evaluate_bits(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn symbol_to_bit_ber_binary() {
        // For binary (1 bit per symbol), symbol errors = bit errors
        let tx = vec![0, 1, 0, 1];
        let rx = vec![0, 0, 0, 0]; // 2 errors
        let ber = BerEvaluator::evaluate_symbol_to_bit(&tx, &rx, 1).expect("evaluate");
        assert!((ber - 0.5).abs() < 1e-15);
    }

    #[test]
    fn symbol_to_bit_ber_quaternary() {
        // Symbol 0=00, Symbol 3=11 → 2 bit errors
        let tx = vec![0];
        let rx = vec![3];
        let ber = BerEvaluator::evaluate_symbol_to_bit(&tx, &rx, 2).expect("evaluate");
        assert!((ber - 1.0).abs() < 1e-15); // 2 bit errors out of 2 bits
    }

    #[test]
    fn symbol_to_bit_zero_bits_per_symbol() {
        assert!(BerEvaluator::evaluate_symbol_to_bit(&[0], &[0], 0).is_err());
    }

    #[test]
    fn trial_result_from_sequences() {
        let tx = vec![0, 1, 0, 1, 0];
        let rx = vec![0, 0, 0, 1, 1]; // 2 errors
        let result = TrialResult::from_sequences(&tx, &rx).expect("trial");
        assert_eq!(result.num_symbols, 5);
        assert_eq!(result.symbol_errors, 2);
        assert!((result.ser - 0.4).abs() < 1e-15);
    }

    #[test]
    fn confidence_interval_basic() {
        let (low, high) = confidence_interval_95(0.5, 100);
        assert!(low > 0.4);
        assert!(high < 0.6);
        assert!(low < 0.5);
        assert!(high > 0.5);
    }

    #[test]
    fn confidence_interval_zero_ber() {
        let (low, _high) = confidence_interval_95(0.0, 1000);
        assert!((low - 0.0).abs() < 1e-15);
    }

    #[test]
    fn confidence_interval_empty() {
        let (low, high) = confidence_interval_95(0.5, 0);
        assert!((low - 0.0).abs() < 1e-15);
        assert!((high - 1.0).abs() < 1e-15);
    }
}
