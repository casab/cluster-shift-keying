use super::ber::{confidence_interval_95, BerCurve, BerEvaluator, BerPoint};
use super::error::MetricsError;

/// Configuration for Monte Carlo BER simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of symbols per trial.
    pub symbols_per_trial: usize,
    /// Number of independent trials per noise level.
    pub num_trials: usize,
    /// Noise standard deviations to sweep.
    pub sigma_values: Vec<f64>,
    /// RNG base seed (each trial uses base_seed + trial_index).
    pub base_seed: u64,
}

impl MonteCarloConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), MetricsError> {
        if self.symbols_per_trial == 0 {
            return Err(MetricsError::SimulationFailed {
                reason: "symbols_per_trial must be at least 1".to_string(),
            });
        }
        if self.num_trials == 0 {
            return Err(MetricsError::SimulationFailed {
                reason: "num_trials must be at least 1".to_string(),
            });
        }
        if self.sigma_values.is_empty() {
            return Err(MetricsError::SimulationFailed {
                reason: "sigma_values must not be empty".to_string(),
            });
        }
        for &sigma in &self.sigma_values {
            if sigma < 0.0 || !sigma.is_finite() {
                return Err(MetricsError::SimulationFailed {
                    reason: format!("sigma must be non-negative and finite, got {sigma}"),
                });
            }
        }
        Ok(())
    }
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            symbols_per_trial: 100,
            num_trials: 10,
            sigma_values: vec![0.0, 0.1, 0.5, 1.0, 2.0],
            base_seed: 42,
        }
    }
}

/// Callback type for reporting progress during Monte Carlo simulation.
pub type ProgressCallback = Box<dyn FnMut(usize, usize) + Send>;

/// Trait for generating trial data in a Monte Carlo simulation.
///
/// Implementations produce transmitted symbols and corresponding received symbols
/// for a given noise level and trial index.
pub trait TrialRunner: Send {
    /// Run a single trial at the given noise level.
    ///
    /// Returns (tx_symbols, rx_symbols).
    fn run_trial(
        &mut self,
        sigma: f64,
        trial_index: usize,
        symbols_per_trial: usize,
        base_seed: u64,
    ) -> Result<(Vec<usize>, Vec<usize>), MetricsError>;
}

/// Monte Carlo BER simulation runner.
///
/// Sweeps over noise levels, running multiple independent trials at each level,
/// and computes mean BER with 95% confidence intervals.
pub struct MonteCarloRunner {
    config: MonteCarloConfig,
}

impl MonteCarloRunner {
    /// Create a new Monte Carlo runner with the given configuration.
    pub fn new(config: MonteCarloConfig) -> Result<Self, MetricsError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Run the Monte Carlo simulation using a trial runner.
    ///
    /// Returns a BER curve with one point per noise level.
    pub fn run(
        &self,
        trial_runner: &mut dyn TrialRunner,
        mut progress: Option<&mut ProgressCallback>,
    ) -> Result<BerCurve, MetricsError> {
        let total_points = self.config.sigma_values.len();
        let mut curve = Vec::with_capacity(total_points);

        for (point_idx, &sigma) in self.config.sigma_values.iter().enumerate() {
            let mut trial_bers = Vec::with_capacity(self.config.num_trials);

            for trial_idx in 0..self.config.num_trials {
                let (tx, rx) = trial_runner.run_trial(
                    sigma,
                    trial_idx,
                    self.config.symbols_per_trial,
                    self.config.base_seed,
                )?;

                let ber = BerEvaluator::evaluate(&tx, &rx)?;
                trial_bers.push(ber);
            }

            let mean_ber = trial_bers.iter().sum::<f64>() / trial_bers.len() as f64;
            let (ci_low, ci_high) = confidence_interval_95(mean_ber, self.config.num_trials);

            curve.push(BerPoint {
                sigma,
                mean_ber,
                ci_low,
                ci_high,
                num_trials: self.config.num_trials,
            });

            if let Some(ref mut cb) = progress {
                cb(point_idx + 1, total_points);
            }
        }

        Ok(curve)
    }

    /// Run the simulation from pre-computed trial results.
    ///
    /// Useful when trials are generated externally (e.g., from a full pipeline).
    /// `results[sigma_index][trial_index]` = (tx_symbols, rx_symbols).
    pub fn evaluate_precomputed(
        &self,
        results: &[Vec<(Vec<usize>, Vec<usize>)>],
    ) -> Result<BerCurve, MetricsError> {
        if results.len() != self.config.sigma_values.len() {
            return Err(MetricsError::SimulationFailed {
                reason: format!(
                    "expected {} sigma levels, got {}",
                    self.config.sigma_values.len(),
                    results.len()
                ),
            });
        }

        let mut curve = Vec::with_capacity(results.len());

        for (sigma_idx, trials) in results.iter().enumerate() {
            let sigma = self.config.sigma_values[sigma_idx];
            let mut trial_bers = Vec::with_capacity(trials.len());

            for (tx, rx) in trials {
                let ber = BerEvaluator::evaluate(tx, rx)?;
                trial_bers.push(ber);
            }

            let n = trial_bers.len();
            let mean_ber = if n > 0 {
                trial_bers.iter().sum::<f64>() / n as f64
            } else {
                0.0
            };
            let (ci_low, ci_high) = confidence_interval_95(mean_ber, n);

            curve.push(BerPoint {
                sigma,
                mean_ber,
                ci_low,
                ci_high,
                num_trials: n,
            });
        }

        Ok(curve)
    }

    /// Get the configuration.
    pub fn config(&self) -> &MonteCarloConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial trial runner that returns identical tx/rx (noiseless).
    struct NoiselessTrialRunner;

    impl TrialRunner for NoiselessTrialRunner {
        fn run_trial(
            &mut self,
            _sigma: f64,
            _trial_index: usize,
            symbols_per_trial: usize,
            _base_seed: u64,
        ) -> Result<(Vec<usize>, Vec<usize>), MetricsError> {
            let symbols: Vec<usize> = (0..symbols_per_trial).map(|i| i % 2).collect();
            Ok((symbols.clone(), symbols))
        }
    }

    /// A trial runner that flips all symbols (worst case).
    struct AllErrorTrialRunner;

    impl TrialRunner for AllErrorTrialRunner {
        fn run_trial(
            &mut self,
            _sigma: f64,
            _trial_index: usize,
            symbols_per_trial: usize,
            _base_seed: u64,
        ) -> Result<(Vec<usize>, Vec<usize>), MetricsError> {
            let tx: Vec<usize> = (0..symbols_per_trial).map(|i| i % 2).collect();
            let rx: Vec<usize> = tx.iter().map(|&s| 1 - s).collect();
            Ok((tx, rx))
        }
    }

    /// A trial runner where error rate scales with sigma.
    struct ScalingTrialRunner;

    impl TrialRunner for ScalingTrialRunner {
        fn run_trial(
            &mut self,
            sigma: f64,
            _trial_index: usize,
            symbols_per_trial: usize,
            _base_seed: u64,
        ) -> Result<(Vec<usize>, Vec<usize>), MetricsError> {
            let tx: Vec<usize> = vec![0; symbols_per_trial];
            // Flip approximately sigma fraction of symbols (clamped to [0,1])
            let error_rate = sigma.min(1.0);
            let num_errors = (symbols_per_trial as f64 * error_rate).round() as usize;
            let mut rx = tx.clone();
            for i in 0..num_errors.min(symbols_per_trial) {
                rx[i] = 1;
            }
            Ok((tx, rx))
        }
    }

    #[test]
    fn mc_noiseless_zero_ber() {
        let config = MonteCarloConfig {
            symbols_per_trial: 50,
            num_trials: 5,
            sigma_values: vec![0.0, 0.5, 1.0],
            base_seed: 42,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");
        let mut trial = NoiselessTrialRunner;
        let curve = runner.run(&mut trial, None).expect("run");

        assert_eq!(curve.len(), 3);
        for point in &curve {
            assert!(
                point.mean_ber.abs() < 1e-15,
                "noiseless should have 0 BER at sigma={}",
                point.sigma
            );
        }
    }

    #[test]
    fn mc_all_errors() {
        let config = MonteCarloConfig {
            symbols_per_trial: 50,
            num_trials: 3,
            sigma_values: vec![1.0],
            base_seed: 0,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");
        let mut trial = AllErrorTrialRunner;
        let curve = runner.run(&mut trial, None).expect("run");

        assert_eq!(curve.len(), 1);
        assert!((curve[0].mean_ber - 1.0).abs() < 1e-15);
    }

    #[test]
    fn mc_scaling_ber() {
        let config = MonteCarloConfig {
            symbols_per_trial: 100,
            num_trials: 1,
            sigma_values: vec![0.0, 0.25, 0.5, 1.0],
            base_seed: 0,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");
        let mut trial = ScalingTrialRunner;
        let curve = runner.run(&mut trial, None).expect("run");

        assert_eq!(curve.len(), 4);
        assert!(curve[0].mean_ber.abs() < 1e-15, "sigma=0 → BER=0");
        assert!(
            (curve[1].mean_ber - 0.25).abs() < 0.02,
            "sigma=0.25 → BER≈0.25"
        );
        assert!(
            (curve[2].mean_ber - 0.5).abs() < 0.02,
            "sigma=0.5 → BER≈0.5"
        );
        assert!(
            (curve[3].mean_ber - 1.0).abs() < 0.02,
            "sigma=1.0 → BER≈1.0"
        );
    }

    #[test]
    fn mc_progress_callback() {
        let config = MonteCarloConfig {
            symbols_per_trial: 10,
            num_trials: 2,
            sigma_values: vec![0.0, 1.0],
            base_seed: 0,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");
        let mut trial = NoiselessTrialRunner;

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();
        let mut cb: ProgressCallback = Box::new(move |_current, _total| {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });

        runner.run(&mut trial, Some(&mut cb)).expect("run");
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "should be called once per sigma value"
        );
    }

    #[test]
    fn mc_evaluate_precomputed() {
        let config = MonteCarloConfig {
            symbols_per_trial: 4,
            num_trials: 2,
            sigma_values: vec![0.0, 1.0],
            base_seed: 0,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");

        let results = vec![
            // sigma=0.0: two perfect trials
            vec![
                (vec![0, 1, 0, 1], vec![0, 1, 0, 1]),
                (vec![0, 1, 0, 1], vec![0, 1, 0, 1]),
            ],
            // sigma=1.0: two trials with 50% errors
            vec![
                (vec![0, 1, 0, 1], vec![1, 0, 0, 1]),
                (vec![0, 1, 0, 1], vec![1, 0, 0, 1]),
            ],
        ];

        let curve = runner.evaluate_precomputed(&results).expect("evaluate");
        assert_eq!(curve.len(), 2);
        assert!(curve[0].mean_ber.abs() < 1e-15);
        assert!((curve[1].mean_ber - 0.5).abs() < 1e-15);
    }

    #[test]
    fn mc_invalid_config_zero_symbols() {
        let config = MonteCarloConfig {
            symbols_per_trial: 0,
            num_trials: 1,
            sigma_values: vec![0.0],
            base_seed: 0,
        };
        assert!(MonteCarloRunner::new(config).is_err());
    }

    #[test]
    fn mc_invalid_config_zero_trials() {
        let config = MonteCarloConfig {
            symbols_per_trial: 10,
            num_trials: 0,
            sigma_values: vec![0.0],
            base_seed: 0,
        };
        assert!(MonteCarloRunner::new(config).is_err());
    }

    #[test]
    fn mc_invalid_config_empty_sigma() {
        let config = MonteCarloConfig {
            symbols_per_trial: 10,
            num_trials: 1,
            sigma_values: vec![],
            base_seed: 0,
        };
        assert!(MonteCarloRunner::new(config).is_err());
    }

    #[test]
    fn mc_invalid_config_negative_sigma() {
        let config = MonteCarloConfig {
            symbols_per_trial: 10,
            num_trials: 1,
            sigma_values: vec![-1.0],
            base_seed: 0,
        };
        assert!(MonteCarloRunner::new(config).is_err());
    }

    #[test]
    fn mc_ber_point_has_confidence_interval() {
        let config = MonteCarloConfig {
            symbols_per_trial: 100,
            num_trials: 10,
            sigma_values: vec![0.5],
            base_seed: 0,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");
        let mut trial = ScalingTrialRunner;
        let curve = runner.run(&mut trial, None).expect("run");

        let point = &curve[0];
        assert!(point.ci_low <= point.mean_ber);
        assert!(point.ci_high >= point.mean_ber);
        assert!(point.ci_low >= 0.0);
        assert!(point.ci_high <= 1.0);
    }

    #[test]
    fn mc_precomputed_wrong_sigma_count() {
        let config = MonteCarloConfig {
            symbols_per_trial: 4,
            num_trials: 1,
            sigma_values: vec![0.0, 1.0],
            base_seed: 0,
        };
        let runner = MonteCarloRunner::new(config).expect("runner");

        // Only 1 sigma level worth of results, but config has 2
        let results = vec![vec![(vec![0], vec![0])]];
        assert!(runner.evaluate_precomputed(&results).is_err());
    }
}
