use rand::Rng;
use rand::SeedableRng;

use super::config::SimulationConfig;
use super::error::PipelineError;
use crate::channel::{ChannelLink, ChannelModel, GaussianChannel, IdealChannel, NoiseMode};
use crate::codec::modulator::{Modulator, ModulatorConfig};
use crate::codec::symbol_map::SymbolMap;
use crate::codec::FrameConfig;
use crate::codec::{Demodulator, DemodulatorConfig};
use crate::dynamics::chen::ChenSystem;
use crate::graph::{ClusterPattern, TopologyBuilder};
use crate::metrics::ber::BerEvaluator;
use crate::metrics::sync_energy::RatioScoring;

/// Result of a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Transmitted symbol sequence.
    pub tx_symbols: Vec<usize>,
    /// Received (decoded) symbol sequence.
    pub rx_symbols: Vec<usize>,
    /// Symbol error rate.
    pub ser: f64,
    /// Number of symbol errors.
    pub symbol_errors: usize,
    /// Total symbols transmitted.
    pub total_symbols: usize,
}

/// End-to-end CLSK simulation pipeline.
///
/// Orchestrates: symbol generation → modulation → channel → demodulation → BER.
pub struct Simulation {
    config: SimulationConfig,
}

impl Simulation {
    /// Create a new simulation from configuration.
    pub fn new(config: SimulationConfig) -> Result<Self, PipelineError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Run the simulation and return results.
    pub fn run(&self) -> Result<SimulationResult, PipelineError> {
        // Build system
        let system = self.build_system()?;

        // Build topology
        let coupling = self.build_topology()?;

        // Build symbol map
        let symbol_map = self.build_symbol_map()?;

        // Build modulator
        let mod_config = ModulatorConfig {
            bit_period: self.config.codec.bit_period,
            dt: self.config.codec.dt,
            initial_state: self.config.codec.initial_state.clone(),
        };
        let mut modulator = Modulator::new(&coupling, symbol_map.clone(), &mod_config)?;

        // Build demodulator
        let frame_config =
            FrameConfig::new(self.config.codec.bit_period, 0.0, self.config.codec.dt)?;
        let demod_config = DemodulatorConfig {
            initial_state: self.config.codec.initial_state.clone(),
        };
        let mut demodulator = Demodulator::new(
            &coupling,
            symbol_map.clone(),
            frame_config,
            Box::new(RatioScoring::default()),
            &demod_config,
        )?;

        // Build channel
        let mut channel = self.build_channel()?;
        let num_links = symbol_map.channel_links().len();
        let mut channel_link = ChannelLink::new(num_links)?;

        // Generate random symbols
        let tx_symbols = self.generate_symbols(&symbol_map);

        // Encode all symbols, collecting signals
        let tx_signals = modulator.encode_sequence(&tx_symbols, &system)?;

        // Pass through channel
        let mut rx_signals = vec![vec![]; num_links];
        channel_link.transmit_all(&tx_signals, &mut rx_signals, channel.as_mut())?;

        // Decode
        let rx_symbols = demodulator.decode_sequence(&rx_signals, tx_symbols.len(), &system)?;

        // Compute BER
        let ser = BerEvaluator::evaluate(&tx_symbols, &rx_symbols)?;
        let symbol_errors = tx_symbols
            .iter()
            .zip(rx_symbols.iter())
            .filter(|(a, b)| a != b)
            .count();

        Ok(SimulationResult {
            tx_symbols,
            rx_symbols,
            ser,
            symbol_errors,
            total_symbols: self.config.simulation.num_symbols,
        })
    }

    fn build_system(&self) -> Result<ChenSystem, PipelineError> {
        ChenSystem::new(
            self.config.system.a,
            self.config.system.b,
            self.config.system.c,
        )
        .map_err(|e| PipelineError::ConfigError {
            reason: format!("failed to create Chen system: {e}"),
        })
    }

    fn build_topology(&self) -> Result<crate::graph::CouplingMatrix, PipelineError> {
        match self.config.topology.topology_type.as_str() {
            "octagon" => TopologyBuilder::octagon().map_err(|e| PipelineError::ConfigError {
                reason: format!("failed to build octagon topology: {e}"),
            }),
            "ring" => TopologyBuilder::ring(self.config.topology.node_count).map_err(|e| {
                PipelineError::ConfigError {
                    reason: format!("failed to build ring topology: {e}"),
                }
            }),
            other => Err(PipelineError::ConfigError {
                reason: format!("unsupported topology: {other}"),
            }),
        }
    }

    fn build_symbol_map(&self) -> Result<SymbolMap, PipelineError> {
        let mut entries = Vec::new();
        for (i, sym_cfg) in self.config.coupling.symbols.iter().enumerate() {
            let pattern = ClusterPattern::new(sym_cfg.pattern.clone()).map_err(|e| {
                PipelineError::ConfigError {
                    reason: format!("symbol {i}: invalid pattern: {e}"),
                }
            })?;
            entries.push((i, pattern, sym_cfg.epsilon));
        }

        let sm = SymbolMap::new(entries, self.config.coupling.channel_links.clone())?;
        Ok(sm)
    }

    fn build_channel(&self) -> Result<Box<dyn ChannelModel>, PipelineError> {
        match self.config.channel.channel_type.as_str() {
            "ideal" => Ok(Box::new(IdealChannel::new())),
            "gaussian" => {
                let mode = match self.config.channel.noise_mode.as_str() {
                    "additive" => NoiseMode::Additive,
                    "multiplicative" => NoiseMode::Multiplicative,
                    other => {
                        return Err(PipelineError::ConfigError {
                            reason: format!("unsupported noise mode: {other}"),
                        })
                    }
                };
                let ch = GaussianChannel::with_mode(
                    self.config.channel.sigma,
                    self.config.channel.seed,
                    mode,
                )?;
                Ok(Box::new(ch))
            }
            other => Err(PipelineError::ConfigError {
                reason: format!("unsupported channel: {other}"),
            }),
        }
    }

    fn generate_symbols(&self, symbol_map: &SymbolMap) -> Vec<usize> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.simulation.seed);
        let alphabet_size = symbol_map.alphabet_size();
        (0..self.config.simulation.num_symbols)
            .map(|_| rng.gen_range(0..alphabet_size))
            .collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::traits::DynamicalSystem;

    #[test]
    fn simulation_creation() {
        let config = SimulationConfig::default_paper();
        let sim = Simulation::new(config).expect("create simulation");
        assert_eq!(sim.config().simulation.num_symbols, 100);
    }

    #[test]
    fn build_system_default() {
        let config = SimulationConfig::default_paper();
        let sim = Simulation::new(config).expect("sim");
        let system = sim.build_system().expect("build system");
        assert_eq!(system.dimension(), 3);
    }

    #[test]
    fn build_topology_octagon() {
        let config = SimulationConfig::default_paper();
        let sim = Simulation::new(config).expect("sim");
        let coupling = sim.build_topology().expect("build topology");
        assert_eq!(coupling.node_count(), 8);
    }

    #[test]
    fn build_symbol_map_default() {
        let config = SimulationConfig::default_paper();
        let sim = Simulation::new(config).expect("sim");
        let sm = sim.build_symbol_map().expect("build symbol map");
        assert_eq!(sm.alphabet_size(), 2);
        assert_eq!(sm.channel_links(), &[0, 3]);
    }

    #[test]
    fn build_channel_ideal() {
        let config = SimulationConfig::default_paper();
        let sim = Simulation::new(config).expect("sim");
        let mut ch = sim.build_channel().expect("build channel");
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        ch.transmit(&input, &mut output).expect("transmit");
        assert_eq!(input, output);
    }

    #[test]
    fn build_channel_gaussian() {
        let mut config = SimulationConfig::default_paper();
        config.channel.channel_type = "gaussian".to_string();
        config.channel.sigma = 0.1;
        let sim = Simulation::new(config).expect("sim");
        let ch = sim.build_channel();
        assert!(ch.is_ok());
    }

    #[test]
    fn generate_symbols_deterministic() {
        let config = SimulationConfig::default_paper();
        let sim = Simulation::new(config).expect("sim");
        let sm = sim.build_symbol_map().expect("sm");
        let symbols1 = sim.generate_symbols(&sm);
        let symbols2 = sim.generate_symbols(&sm);
        assert_eq!(symbols1, symbols2, "same seed should produce same symbols");
        assert_eq!(symbols1.len(), 100);
        assert!(symbols1.iter().all(|&s| s < 2));
    }

    #[test]
    fn run_small_simulation() {
        let mut config = SimulationConfig::default_paper();
        // Use small parameters for fast test
        config.codec.bit_period = 1.0;
        config.simulation.num_symbols = 4;
        let sim = Simulation::new(config).expect("sim");
        let result = sim.run().expect("run");

        assert_eq!(result.tx_symbols.len(), 4);
        assert_eq!(result.rx_symbols.len(), 4);
        assert!(result.ser >= 0.0 && result.ser <= 1.0);
        assert_eq!(result.total_symbols, 4);
        // All symbols should be valid
        for &s in &result.rx_symbols {
            assert!(s < 2, "decoded symbol {s} should be valid");
        }
    }
}
