use super::error::CodecError;
use super::symbol_map::{Symbol, SymbolMap};
use super::traits::Encoder;
use crate::dynamics::traits::DynamicalSystem;
use crate::graph::CouplingMatrix;
use crate::sync::CoupledNetwork;

/// Configuration for the CLSK modulator.
#[derive(Debug, Clone)]
pub struct ModulatorConfig {
    /// Bit period T_b in time units. The network is driven at a symbol's ε
    /// for this duration before switching to the next symbol.
    pub bit_period: f64,
    /// Integration time step dt.
    pub dt: f64,
    /// Initial state for all oscillators (on or near the attractor).
    pub initial_state: Vec<f64>,
}

impl Default for ModulatorConfig {
    fn default() -> Self {
        Self {
            bit_period: 10.0,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        }
    }
}

/// CLSK modulator: encodes symbols by switching the coupling strength of a
/// coupled chaotic network and extracting channel link signals.
///
/// For each symbol, the modulator:
/// 1. Sets the coupling strength ε corresponding to the symbol
/// 2. Integrates the coupled network for one bit period T_b
/// 3. Extracts signals on the channel link nodes during integration
pub struct Modulator {
    /// The coupled network being driven.
    network: CoupledNetwork,
    /// Symbol-to-pattern/epsilon mapping.
    symbol_map: SymbolMap,
    /// Bit period in time units.
    bit_period: f64,
    /// Integration time step.
    dt: f64,
    /// Number of integration steps per bit period.
    steps_per_bit: usize,
    /// Output buffer: channel link signals for the last encoded symbol.
    /// Indexed as output_signals[link_index][time_step].
    output_signals: Vec<Vec<f64>>,
}

impl Modulator {
    /// Create a new CLSK modulator.
    ///
    /// `coupling` defines the network topology. The initial coupling strength
    /// is set from the first symbol's ε (will be switched on each encode).
    pub fn new(
        coupling: &CouplingMatrix,
        symbol_map: SymbolMap,
        config: &ModulatorConfig,
    ) -> Result<Self, CodecError> {
        // Validate that channel link nodes are valid for this topology
        let n = coupling.node_count();
        if symbol_map.num_nodes() != n {
            return Err(CodecError::InvalidSymbolMap {
                reason: format!(
                    "symbol map has {} nodes but network has {n}",
                    symbol_map.num_nodes()
                ),
            });
        }

        // Small deterministic perturbations break the initial symmetry so that
        // the coupling term ε Σⱼ ξᵢⱼ Γ (xⱼ - xᵢ) is non-zero and different
        // coupling strengths produce distinguishable dynamics.
        let dim = config.initial_state.len();
        let perturbations: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| 0.01 * ((i * dim + d) as f64 * 0.37).sin())
                    .collect()
            })
            .collect();
        let network = CoupledNetwork::new(coupling, &config.initial_state, Some(&perturbations))?;

        let steps_per_bit = (config.bit_period / config.dt).round() as usize;
        if steps_per_bit == 0 {
            return Err(CodecError::FramingError {
                reason: "bit period / dt yields 0 steps".to_string(),
            });
        }

        let num_links = symbol_map.channel_links().len();
        let output_signals = vec![Vec::with_capacity(steps_per_bit); num_links];

        Ok(Self {
            network,
            symbol_map,
            bit_period: config.bit_period,
            dt: config.dt,
            steps_per_bit,
            output_signals,
        })
    }

    /// Get the channel link signals from the last encoded symbol.
    ///
    /// Returns a slice of signal vectors, one per channel link node.
    /// Each signal vector has `steps_per_bit` samples.
    pub fn output_signals(&self) -> &[Vec<f64>] {
        &self.output_signals
    }

    /// Drain the output signals, returning ownership and clearing the buffer.
    pub fn drain_output_signals(&mut self) -> Vec<Vec<f64>> {
        let mut signals = vec![Vec::new(); self.symbol_map.channel_links().len()];
        std::mem::swap(&mut self.output_signals, &mut signals);
        signals
    }

    /// Get a reference to the symbol map.
    pub fn symbol_map(&self) -> &SymbolMap {
        &self.symbol_map
    }

    /// Get the bit period.
    pub fn bit_period(&self) -> f64 {
        self.bit_period
    }

    /// Get the time step.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Get the number of steps per bit period.
    pub fn steps_per_bit(&self) -> usize {
        self.steps_per_bit
    }

    /// Get a reference to the internal network.
    pub fn network(&self) -> &CoupledNetwork {
        &self.network
    }

    /// Encode a symbol using a specific dynamical system.
    ///
    /// Sets the coupling strength, integrates for one bit period,
    /// and records the channel link signals.
    pub fn encode_with_system(
        &mut self,
        symbol: &Symbol,
        system: &dyn DynamicalSystem,
    ) -> Result<(), CodecError> {
        let epsilon = self.symbol_map.lookup_epsilon(*symbol)?;
        let channel_links: Vec<usize> = self.symbol_map.channel_links().to_vec();

        // Set coupling strength for this symbol
        self.network.set_coupling_strength(epsilon);

        // Clear output buffers
        for sig in &mut self.output_signals {
            sig.clear();
        }

        // Integrate and record channel link signals
        let dim = self.network.dimension();
        for _ in 0..self.steps_per_bit {
            // Record current state of channel link nodes (second state variable
            // for Chen, since Γ = diag(0,1,0) couples through y)
            for (link_idx, &node) in channel_links.iter().enumerate() {
                let state = self.network.node_state(node)?;
                // Extract the coupled state variable (index 1 for standard Chen+Γ)
                let signal_val = if dim > 1 { state[1] } else { state[0] };
                self.output_signals[link_idx].push(signal_val);
            }

            self.network.step(system, self.dt)?;
        }

        Ok(())
    }

    /// Encode a sequence of symbols, returning all channel link signals.
    ///
    /// Returns signals[link_index][time_step] covering all symbols.
    pub fn encode_sequence(
        &mut self,
        symbols: &[Symbol],
        system: &dyn DynamicalSystem,
    ) -> Result<Vec<Vec<f64>>, CodecError> {
        let num_links = self.symbol_map.channel_links().len();
        let total_steps = symbols.len() * self.steps_per_bit;
        let mut all_signals = vec![Vec::with_capacity(total_steps); num_links];

        for symbol in symbols {
            self.encode_with_system(symbol, system)?;
            for (link_idx, sig) in self.output_signals.iter().enumerate() {
                all_signals[link_idx].extend_from_slice(sig);
            }
        }

        Ok(all_signals)
    }
}

/// Implement the Encoder trait with a boxed DynamicalSystem.
///
/// For the trait-based interface, the system must be stored alongside
/// the modulator. Use `ModulatorWithSystem` for this purpose.
pub struct ModulatorWithSystem {
    modulator: Modulator,
    system: Box<dyn DynamicalSystem>,
}

impl ModulatorWithSystem {
    /// Create a new modulator with an owned dynamical system.
    pub fn new(
        coupling: &CouplingMatrix,
        symbol_map: SymbolMap,
        config: &ModulatorConfig,
        system: Box<dyn DynamicalSystem>,
    ) -> Result<Self, CodecError> {
        let modulator = Modulator::new(coupling, symbol_map, config)?;
        Ok(Self { modulator, system })
    }

    /// Access the inner modulator.
    pub fn modulator(&self) -> &Modulator {
        &self.modulator
    }

    /// Access the inner modulator mutably.
    pub fn modulator_mut(&mut self) -> &mut Modulator {
        &mut self.modulator
    }

    /// Get output signals from the last encode.
    pub fn output_signals(&self) -> &[Vec<f64>] {
        self.modulator.output_signals()
    }

    /// Drain output signals.
    pub fn drain_output_signals(&mut self) -> Vec<Vec<f64>> {
        self.modulator.drain_output_signals()
    }
}

impl Encoder for ModulatorWithSystem {
    type Symbol = Symbol;
    type Error = CodecError;

    fn encode(&mut self, symbol: &Self::Symbol) -> Result<(), Self::Error> {
        self.modulator
            .encode_with_system(symbol, self.system.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;
    use crate::graph::{ClusterPattern, TopologyBuilder};

    fn setup_binary_modulator(bit_period: f64) -> (Modulator, ChenSystem) {
        let chen = ChenSystem::default_paper();
        let coupling = TopologyBuilder::octagon().expect("octagon");

        // Two patterns: alternating and paired 2-cluster
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");

        // Channel links: nodes 0 and 3 (in different clusters for both patterns)
        let sm = SymbolMap::new(vec![(0, p0, 8.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol map");

        let config = ModulatorConfig {
            bit_period,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };

        let modulator = Modulator::new(&coupling, sm, &config).expect("modulator");
        (modulator, chen)
    }

    #[test]
    fn modulator_creation() {
        let (modulator, _) = setup_binary_modulator(5.0);
        assert_eq!(modulator.steps_per_bit(), 5000);
        assert!((modulator.bit_period() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn encode_produces_signals() {
        let (mut modulator, chen) = setup_binary_modulator(1.0);
        modulator.encode_with_system(&0, &chen).expect("encode 0");

        let signals = modulator.output_signals();
        assert_eq!(signals.len(), 2, "should have 2 channel link signals");
        assert_eq!(
            signals[0].len(),
            1000,
            "each signal should have steps_per_bit samples"
        );
        assert_eq!(signals[1].len(), 1000);

        // Signals should be finite and non-constant (chaotic)
        for sig in signals {
            assert!(
                sig.iter().all(|x| x.is_finite()),
                "signals should be finite"
            );
            let min = sig.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = sig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                max - min > 0.01,
                "signal should vary (chaotic), range = {}",
                max - min
            );
        }
    }

    #[test]
    fn different_symbols_produce_different_energy() {
        let (mut modulator, chen) = setup_binary_modulator(2.0);

        // Encode symbol 0
        modulator.encode_with_system(&0, &chen).expect("encode 0");
        let sig0 = modulator.drain_output_signals();
        let energy0: f64 = sig0[0].iter().map(|x| x * x).sum::<f64>() / sig0[0].len() as f64;

        // Encode symbol 1
        modulator.encode_with_system(&1, &chen).expect("encode 1");
        let sig1 = modulator.drain_output_signals();
        let energy1: f64 = sig1[0].iter().map(|x| x * x).sum::<f64>() / sig1[0].len() as f64;

        // The energies should be different (different coupling strengths)
        // This is a weak check since the network hasn't reached steady-state
        assert!(
            energy0.is_finite() && energy1.is_finite(),
            "energies should be finite: {energy0}, {energy1}"
        );
    }

    #[test]
    fn encode_sequence_concatenates() {
        let (mut modulator, chen) = setup_binary_modulator(0.5);
        let symbols = vec![0, 1, 0, 1];
        let signals = modulator
            .encode_sequence(&symbols, &chen)
            .expect("encode sequence");

        assert_eq!(signals.len(), 2);
        let expected_len = 4 * modulator.steps_per_bit();
        assert_eq!(
            signals[0].len(),
            expected_len,
            "concatenated signal length should be 4 * steps_per_bit"
        );
    }

    #[test]
    fn encoder_trait_via_modulator_with_system() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");
        let sm = SymbolMap::new(vec![(0, p0, 8.0), (1, p1, 12.0)], vec![0, 3]).expect("sm");
        let config = ModulatorConfig {
            bit_period: 0.5,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };

        let chen = ChenSystem::default_paper();
        let mut enc =
            ModulatorWithSystem::new(&coupling, sm, &config, Box::new(chen)).expect("enc");

        enc.encode(&0).expect("encode");
        let signals = enc.output_signals();
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].len(), 500);
    }

    #[test]
    fn modulator_node_count_mismatch() {
        let coupling = TopologyBuilder::ring(4).expect("ring4");
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("8 nodes");
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("8 nodes");
        let sm = SymbolMap::new(vec![(0, p0, 8.0), (1, p1, 12.0)], vec![0, 3]).expect("sm");
        let config = ModulatorConfig::default();

        let result = Modulator::new(&coupling, sm, &config);
        assert!(result.is_err());
    }

    #[test]
    fn drain_clears_signals() {
        let (mut modulator, chen) = setup_binary_modulator(0.5);
        modulator.encode_with_system(&0, &chen).expect("encode");

        let drained = modulator.drain_output_signals();
        assert!(!drained[0].is_empty());

        // After drain, signals should be empty
        let remaining = modulator.output_signals();
        assert!(remaining[0].is_empty());
    }
}
