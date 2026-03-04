use super::error::CodecError;
use super::symbol_map::Symbol;

/// Frame configuration for CLSK transmission.
///
/// Defines the timing structure for symbol transmission including
/// bit period, guard intervals, and preamble sequences.
#[derive(Debug, Clone)]
pub struct FrameConfig {
    /// Bit period T_b in time units (duration per symbol).
    pub bit_period: f64,
    /// Guard interval between symbols (in time units).
    /// Allows transient effects to settle after coupling strength switches.
    pub guard_interval: f64,
    /// Integration time step dt.
    pub dt: f64,
    /// Optional preamble/synchronization sequence.
    /// Transmitted before the data symbols to allow the receiver to lock on.
    pub preamble: Vec<Symbol>,
}

impl FrameConfig {
    /// Create a new frame configuration.
    pub fn new(bit_period: f64, guard_interval: f64, dt: f64) -> Result<Self, CodecError> {
        if bit_period <= 0.0 || !bit_period.is_finite() {
            return Err(CodecError::FramingError {
                reason: format!("bit period must be positive and finite, got {bit_period}"),
            });
        }
        if guard_interval < 0.0 || !guard_interval.is_finite() {
            return Err(CodecError::FramingError {
                reason: format!(
                    "guard interval must be non-negative and finite, got {guard_interval}"
                ),
            });
        }
        if dt <= 0.0 || !dt.is_finite() {
            return Err(CodecError::FramingError {
                reason: format!("dt must be positive and finite, got {dt}"),
            });
        }
        if dt > bit_period {
            return Err(CodecError::FramingError {
                reason: format!("dt ({dt}) must not exceed bit period ({bit_period})"),
            });
        }

        Ok(Self {
            bit_period,
            guard_interval,
            dt,
            preamble: Vec::new(),
        })
    }

    /// Set the preamble sequence.
    pub fn with_preamble(mut self, preamble: Vec<Symbol>) -> Self {
        self.preamble = preamble;
        self
    }

    /// Number of integration steps per bit period.
    pub fn steps_per_bit(&self) -> usize {
        (self.bit_period / self.dt).round() as usize
    }

    /// Number of integration steps for the guard interval.
    pub fn guard_steps(&self) -> usize {
        (self.guard_interval / self.dt).round() as usize
    }

    /// Total steps per symbol (bit period + guard interval).
    pub fn total_steps_per_symbol(&self) -> usize {
        self.steps_per_bit() + self.guard_steps()
    }

    /// Total time per symbol including guard interval.
    pub fn total_time_per_symbol(&self) -> f64 {
        self.bit_period + self.guard_interval
    }

    /// Total number of symbols in a frame (preamble + data).
    pub fn frame_length(&self, data_symbols: usize) -> usize {
        self.preamble.len() + data_symbols
    }
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            bit_period: 10.0,
            guard_interval: 1.0,
            dt: 0.001,
            preamble: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_config_creation() {
        let fc = FrameConfig::new(10.0, 1.0, 0.001).expect("frame config");
        assert!((fc.bit_period - 10.0).abs() < 1e-15);
        assert!((fc.guard_interval - 1.0).abs() < 1e-15);
        assert!((fc.dt - 0.001).abs() < 1e-15);
        assert_eq!(fc.steps_per_bit(), 10000);
        assert_eq!(fc.guard_steps(), 1000);
        assert_eq!(fc.total_steps_per_symbol(), 11000);
    }

    #[test]
    fn frame_config_default() {
        let fc = FrameConfig::default();
        assert!((fc.bit_period - 10.0).abs() < 1e-15);
        assert_eq!(fc.steps_per_bit(), 10000);
    }

    #[test]
    fn frame_config_with_preamble() {
        let fc = FrameConfig::new(5.0, 0.5, 0.001)
            .expect("fc")
            .with_preamble(vec![0, 1, 0, 1]);
        assert_eq!(fc.preamble.len(), 4);
        assert_eq!(fc.frame_length(10), 14);
    }

    #[test]
    fn invalid_bit_period() {
        assert!(FrameConfig::new(0.0, 1.0, 0.001).is_err());
        assert!(FrameConfig::new(-1.0, 1.0, 0.001).is_err());
        assert!(FrameConfig::new(f64::NAN, 1.0, 0.001).is_err());
    }

    #[test]
    fn invalid_guard_interval() {
        assert!(FrameConfig::new(10.0, -1.0, 0.001).is_err());
        assert!(FrameConfig::new(10.0, f64::INFINITY, 0.001).is_err());
    }

    #[test]
    fn invalid_dt() {
        assert!(FrameConfig::new(10.0, 1.0, 0.0).is_err());
        assert!(FrameConfig::new(10.0, 1.0, -0.001).is_err());
    }

    #[test]
    fn dt_exceeds_bit_period() {
        assert!(FrameConfig::new(0.001, 0.0, 0.01).is_err());
    }

    #[test]
    fn zero_guard_interval() {
        let fc = FrameConfig::new(5.0, 0.0, 0.001).expect("no guard");
        assert_eq!(fc.guard_steps(), 0);
        assert_eq!(fc.total_steps_per_symbol(), fc.steps_per_bit());
    }

    #[test]
    fn total_time_calculation() {
        let fc = FrameConfig::new(10.0, 2.0, 0.001).expect("fc");
        assert!((fc.total_time_per_symbol() - 12.0).abs() < 1e-15);
    }
}
