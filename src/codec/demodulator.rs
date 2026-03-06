use super::error::CodecError;
use super::framing::FrameConfig;
use super::symbol_map::{Symbol, SymbolMap};
use super::traits::Decoder;
use crate::dynamics::traits::DynamicalSystem;
use crate::graph::CouplingMatrix;
use crate::sync::CoupledNetwork;

/// Configuration for the CLSK demodulator.
#[derive(Debug, Clone)]
pub struct DemodulatorConfig {
    /// Initial state for all receiver oscillators.
    pub initial_state: Vec<f64>,
}

impl Default for DemodulatorConfig {
    fn default() -> Self {
        Self {
            initial_state: vec![1.0, 1.0, 1.0],
        }
    }
}

/// CLSK demodulator: decodes symbols from received channel link signals
/// using matched filter detection.
///
/// The demodulator maintains a **reference network** that mirrors the
/// transmitter's state. For each bit period:
/// 1. For each candidate symbol (ε_m), save reference state, set ε_m,
///    simulate for T_b, and extract predicted channel link signals.
/// 2. Compute MSE between predicted and received signals.
/// 3. Select the candidate with the lowest MSE (best match).
/// 4. Advance the reference network with the winning ε to maintain
///    state continuity for the next symbol.
///
/// This works because the reference network tracks the TX state (assuming
/// correct previous detections). The correct ε produces signals matching
/// the received ones; incorrect ε produces different signals.
/// Default amplitude for inter-symbol perturbation (must match modulator).
const DEFAULT_PERTURBATION_ALPHA: f64 = 0.01;

pub struct Demodulator {
    /// Reference network mirroring the transmitter state.
    network: CoupledNetwork,
    /// Symbol-to-pattern/epsilon mapping.
    symbol_map: SymbolMap,
    /// Frame configuration.
    frame_config: FrameConfig,
    /// Buffer for received channel link signals (one per link, per bit period).
    /// Indexed as received_signals[link_index][time_step].
    received_signals: Vec<Vec<f64>>,
    /// Pre-allocated buffer for saving/restoring network states, avoiding
    /// per-symbol heap allocation in detect_symbol().
    saved_states_buf: Vec<f64>,
    /// Counter for inter-symbol perturbation keying (must match modulator).
    symbol_counter: usize,
}

impl Demodulator {
    /// Create a new CLSK demodulator.
    ///
    /// The reference network uses the same topology and initial conditions
    /// as the transmitter (including deterministic perturbations) so it can
    /// predict channel link signals for each candidate symbol.
    ///
    /// The `_scorer` parameter is accepted for API compatibility but is no
    /// longer used; detection is performed via matched filter (MSE).
    pub fn new(
        coupling: &CouplingMatrix,
        symbol_map: SymbolMap,
        frame_config: FrameConfig,
        _scorer: Box<dyn crate::metrics::sync_energy::ScoringFunction>,
        config: &DemodulatorConfig,
    ) -> Result<Self, CodecError> {
        let n = coupling.node_count();
        if symbol_map.num_nodes() != n {
            return Err(CodecError::InvalidSymbolMap {
                reason: format!(
                    "symbol map has {} nodes but network has {n}",
                    symbol_map.num_nodes()
                ),
            });
        }

        // Same deterministic perturbations as the modulator, so the reference
        // network mirrors the TX state exactly.
        let dim = config.initial_state.len();
        let perturbations: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| 0.01 * ((i * dim + d) as f64 * 0.37).sin())
                    .collect()
            })
            .collect();
        let network = CoupledNetwork::new(coupling, &config.initial_state, Some(&perturbations))?;
        let num_links = symbol_map.channel_links().len();
        let steps = frame_config.steps_per_bit();
        let state_size = n * config.initial_state.len();

        Ok(Self {
            network,
            symbol_map,
            frame_config,
            received_signals: vec![Vec::with_capacity(steps); num_links],
            saved_states_buf: vec![0.0; state_size],
            symbol_counter: 0,
        })
    }

    /// Feed received channel link signals for one bit period.
    ///
    /// `signals[link_index][time_step]` — each inner vec must have
    /// `steps_per_bit` samples.
    pub fn feed_signals(&mut self, signals: &[Vec<f64>]) -> Result<(), CodecError> {
        let num_links = self.symbol_map.channel_links().len();
        if signals.len() != num_links {
            return Err(CodecError::DetectionFailed {
                reason: format!(
                    "expected {num_links} channel link signals, got {}",
                    signals.len()
                ),
            });
        }

        let expected_len = self.frame_config.steps_per_bit();
        for (i, sig) in signals.iter().enumerate() {
            if sig.len() != expected_len {
                return Err(CodecError::DetectionFailed {
                    reason: format!(
                        "channel link {i}: expected {expected_len} samples, got {}",
                        sig.len()
                    ),
                });
            }
        }

        self.received_signals = signals.to_vec();
        Ok(())
    }

    /// Detect the symbol from the currently fed signals using matched filter.
    ///
    /// For each candidate symbol (ε_m):
    /// 1. Save reference network state
    /// 2. Set coupling to ε_m, simulate for T_b
    /// 3. Extract predicted channel link signals
    /// 4. Compute MSE between predicted and received signals
    ///
    /// The candidate with the lowest MSE is selected. After detection,
    /// the reference network is advanced with the winning ε for state
    /// continuity.
    pub fn detect_symbol(&mut self, system: &dyn DynamicalSystem) -> Result<Symbol, CodecError> {
        let steps = self.frame_config.steps_per_bit();
        let dim = self.network.dimension();
        let dt = self.frame_config.dt;
        let channel_links: Vec<usize> = self.symbol_map.channel_links().to_vec();
        let num_links = channel_links.len();
        let normalizer = (steps * num_links) as f64;

        // Apply the same inter-symbol perturbation as the modulator to keep
        // the reference network in sync with the transmitter.
        self.network
            .apply_inter_symbol_perturbation(self.symbol_counter, DEFAULT_PERTURBATION_ALPHA);
        self.symbol_counter += 1;

        // Save reference network state into pre-allocated buffer (no heap alloc)
        self.saved_states_buf
            .copy_from_slice(self.network.states_flat());

        let mut best_symbol: Option<Symbol> = None;
        let mut best_mse = f64::INFINITY;

        for entry in self.symbol_map.entries() {
            // Restore reference network to saved state
            self.network.restore_states(&self.saved_states_buf)?;

            // Set coupling to this candidate's epsilon
            self.network.set_coupling_strength(entry.epsilon);

            // Simulate reference network and extract predicted channel signals
            // with early stopping: if cumulative MSE already exceeds best,
            // skip remaining timesteps for this candidate.
            let mut cumulative_sse = 0.0;
            let best_sse_threshold = best_mse * normalizer;
            let mut early_stopped = false;

            for step in 0..steps {
                for (link_idx, &node) in channel_links.iter().enumerate() {
                    let state = self.network.node_state(node)?;
                    let predicted = if dim > 1 { state[1] } else { state[0] };
                    let received = self.received_signals[link_idx][step];
                    let diff = predicted - received;
                    cumulative_sse += diff * diff;
                }

                // Early stopping: if partial SSE already exceeds best, no need to continue
                if cumulative_sse > best_sse_threshold {
                    early_stopped = true;
                    break;
                }

                self.network.step(system, dt)?;
            }

            if !early_stopped {
                let mse = cumulative_sse / normalizer;
                if mse < best_mse {
                    best_mse = mse;
                    best_symbol = Some(entry.symbol);
                }
            }
        }

        let winner = best_symbol.ok_or(CodecError::DetectionFailed {
            reason: "no symbols in symbol map".to_string(),
        })?;

        // Advance reference network with the winning ε for state continuity
        self.network.restore_states(&self.saved_states_buf)?;
        let winner_epsilon = self.symbol_map.lookup_epsilon(winner)?;
        self.network.set_coupling_strength(winner_epsilon);
        for _t in 0..steps {
            self.network.step(system, dt)?;
        }

        Ok(winner)
    }

    /// Decode a sequence of symbols from concatenated channel link signals.
    ///
    /// `signals[link_index]` contains all samples for all bit periods,
    /// concatenated. The total length must be `num_symbols * steps_per_bit`.
    pub fn decode_sequence(
        &mut self,
        signals: &[Vec<f64>],
        num_symbols: usize,
        system: &dyn DynamicalSystem,
    ) -> Result<Vec<Symbol>, CodecError> {
        let steps = self.frame_config.steps_per_bit();
        let num_links = self.symbol_map.channel_links().len();

        if signals.len() != num_links {
            return Err(CodecError::DetectionFailed {
                reason: format!(
                    "expected {num_links} channel link signals, got {}",
                    signals.len()
                ),
            });
        }

        let expected_total = num_symbols * steps;
        for (i, sig) in signals.iter().enumerate() {
            if sig.len() != expected_total {
                return Err(CodecError::DetectionFailed {
                    reason: format!(
                        "link {i}: expected {} samples ({num_symbols} symbols * {steps} steps), got {}",
                        expected_total,
                        sig.len()
                    ),
                });
            }
        }

        let mut decoded = Vec::with_capacity(num_symbols);

        for sym_idx in 0..num_symbols {
            let start = sym_idx * steps;
            let end = start + steps;

            // Extract one bit period's worth of signals per link
            let bit_signals: Vec<Vec<f64>> =
                signals.iter().map(|sig| sig[start..end].to_vec()).collect();

            self.feed_signals(&bit_signals)?;
            let symbol = self.detect_symbol(system)?;
            decoded.push(symbol);
        }

        Ok(decoded)
    }

    /// Get a reference to the symbol map.
    pub fn symbol_map(&self) -> &SymbolMap {
        &self.symbol_map
    }

    /// Get a reference to the frame config.
    pub fn frame_config(&self) -> &FrameConfig {
        &self.frame_config
    }

    /// Get a reference to the receiver network.
    pub fn network(&self) -> &CoupledNetwork {
        &self.network
    }

    /// Get the detection scores (negative MSE) for each candidate symbol.
    ///
    /// For each candidate, simulates the reference network at that candidate's ε
    /// and computes MSE against received signals. Returns (symbol, -mse) pairs
    /// sorted by score descending (lowest MSE first).
    ///
    /// Does not advance the reference network state.
    pub fn score_all(
        &mut self,
        system: &dyn DynamicalSystem,
    ) -> Result<Vec<(Symbol, f64)>, CodecError> {
        let steps = self.frame_config.steps_per_bit();
        let dim = self.network.dimension();
        let dt = self.frame_config.dt;
        let channel_links: Vec<usize> = self.symbol_map.channel_links().to_vec();
        let num_links = channel_links.len();

        self.saved_states_buf
            .copy_from_slice(self.network.states_flat());

        let mut detection_scores: Vec<(Symbol, f64)> = Vec::new();
        for entry in self.symbol_map.entries() {
            self.network.restore_states(&self.saved_states_buf)?;
            self.network.set_coupling_strength(entry.epsilon);

            let mut mse = 0.0;
            for step in 0..steps {
                for (link_idx, &node) in channel_links.iter().enumerate() {
                    let state = self.network.node_state(node)?;
                    let predicted = if dim > 1 { state[1] } else { state[0] };
                    let received = self.received_signals[link_idx][step];
                    let diff = predicted - received;
                    mse += diff * diff;
                }
                self.network.step(system, dt)?;
            }

            mse /= (steps * num_links) as f64;
            // Return negative MSE so higher = better (consistent with scoring convention)
            detection_scores.push((entry.symbol, -mse));
        }

        // Restore original state (score_all should not advance the network)
        self.network.restore_states(&self.saved_states_buf)?;

        detection_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(detection_scores)
    }
}

/// Wraps a demodulator with an owned dynamical system for the Decoder trait.
pub struct DemodulatorWithSystem {
    demodulator: Demodulator,
    system: Box<dyn DynamicalSystem>,
}

impl DemodulatorWithSystem {
    /// Create a new demodulator with an owned dynamical system.
    pub fn new(
        coupling: &CouplingMatrix,
        symbol_map: SymbolMap,
        frame_config: FrameConfig,
        scorer: Box<dyn crate::metrics::sync_energy::ScoringFunction>,
        config: &DemodulatorConfig,
        system: Box<dyn DynamicalSystem>,
    ) -> Result<Self, CodecError> {
        let demodulator = Demodulator::new(coupling, symbol_map, frame_config, scorer, config)?;
        Ok(Self {
            demodulator,
            system,
        })
    }

    /// Access the inner demodulator.
    pub fn demodulator(&self) -> &Demodulator {
        &self.demodulator
    }

    /// Access the inner demodulator mutably.
    pub fn demodulator_mut(&mut self) -> &mut Demodulator {
        &mut self.demodulator
    }

    /// Feed signals for one bit period.
    pub fn feed_signals(&mut self, signals: &[Vec<f64>]) -> Result<(), CodecError> {
        self.demodulator.feed_signals(signals)
    }
}

impl Decoder for DemodulatorWithSystem {
    type Symbol = Symbol;
    type Error = CodecError;

    fn decode(&mut self) -> Result<Self::Symbol, Self::Error> {
        self.demodulator.detect_symbol(self.system.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;
    use crate::graph::{ClusterPattern, TopologyBuilder};
    use crate::metrics::sync_energy::RatioScoring;

    fn setup_codec(
        bit_period: f64,
    ) -> (crate::codec::modulator::Modulator, Demodulator, ChenSystem) {
        let chen = ChenSystem::default_paper();
        let coupling = TopologyBuilder::octagon().expect("octagon");

        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 8.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol_map");

        let mod_config = crate::codec::modulator::ModulatorConfig {
            bit_period,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };

        let modulator =
            crate::codec::modulator::Modulator::new(&coupling, symbol_map.clone(), &mod_config)
                .expect("modulator");

        let frame_config = FrameConfig::new(bit_period, 0.0, 0.001).expect("frame config");
        let demod_config = DemodulatorConfig::default();

        let demodulator = Demodulator::new(
            &coupling,
            symbol_map,
            frame_config,
            Box::new(RatioScoring::default()),
            &demod_config,
        )
        .expect("demodulator");

        (modulator, demodulator, chen)
    }

    #[test]
    fn demodulator_creation() {
        let (_, demod, _) = setup_codec(5.0);
        assert_eq!(demod.symbol_map().alphabet_size(), 2);
        assert_eq!(demod.network().node_count(), 8);
    }

    #[test]
    fn demodulator_node_count_mismatch() {
        let coupling = TopologyBuilder::ring(4).expect("ring4");
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 8.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol_map");
        let frame_config = FrameConfig::new(5.0, 0.0, 0.001).expect("frame_config");
        let config = DemodulatorConfig::default();

        let result = Demodulator::new(
            &coupling,
            symbol_map,
            frame_config,
            Box::new(RatioScoring::default()),
            &config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn feed_wrong_signal_count() {
        let (_, mut demod, _) = setup_codec(1.0);
        // Only 1 signal instead of 2
        let signals = vec![vec![0.0; 1000]];
        assert!(demod.feed_signals(&signals).is_err());
    }

    #[test]
    fn feed_wrong_signal_length() {
        let (_, mut demod, _) = setup_codec(1.0);
        // Wrong length
        let signals = vec![vec![0.0; 500], vec![0.0; 500]];
        assert!(demod.feed_signals(&signals).is_err());
    }

    #[test]
    fn single_symbol_encode_decode() {
        let (mut modulator, mut demodulator, chen) = setup_codec(5.0);

        // Encode symbol 0
        modulator.encode_with_system(&0, &chen).expect("encode");
        let signals = modulator.output_signals().to_vec();

        // Decode
        demodulator.feed_signals(&signals).expect("feed");
        let detected = demodulator.detect_symbol(&chen).expect("detect");

        // With short bit period, detection might not be perfect,
        // but the scorer should at least return a valid symbol
        assert!(detected < 2, "detected symbol should be valid");
    }

    #[test]
    fn decode_sequence_length() {
        let (mut modulator, mut demodulator, chen) = setup_codec(2.0);

        let symbols = vec![0, 1, 0, 1];
        let signals = modulator
            .encode_sequence(&symbols, &chen)
            .expect("encode seq");

        let decoded = demodulator
            .decode_sequence(&signals, 4, &chen)
            .expect("decode seq");

        assert_eq!(decoded.len(), 4, "should decode 4 symbols");
        // All decoded symbols should be valid
        for &s in &decoded {
            assert!(s < 2, "decoded symbol {s} should be valid");
        }
    }

    #[test]
    fn decode_sequence_wrong_length() {
        let (_, mut demodulator, chen) = setup_codec(1.0);

        // 3 symbols * 1000 steps = 3000, but provide 2000
        let signals = vec![vec![0.0; 2000], vec![0.0; 2000]];
        let result = demodulator.decode_sequence(&signals, 3, &chen);
        assert!(result.is_err());
    }

    #[test]
    fn decoder_trait_via_demodulator_with_system() {
        let coupling = TopologyBuilder::octagon().expect("octagon");
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 8.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol_map");
        let frame_config = FrameConfig::new(2.0, 0.0, 0.001).expect("frame_config");
        let config = DemodulatorConfig::default();

        let chen = ChenSystem::default_paper();
        let mut decoder = DemodulatorWithSystem::new(
            &coupling,
            symbol_map,
            frame_config,
            Box::new(RatioScoring::default()),
            &config,
            Box::new(chen.clone()),
        )
        .expect("decoder");

        // Encode a symbol first to get valid signals
        let sm2 = SymbolMap::new(
            vec![
                (
                    0,
                    ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0"),
                    8.0,
                ),
                (
                    1,
                    ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1"),
                    12.0,
                ),
            ],
            vec![0, 3],
        )
        .expect("sm2");

        let mod_config = crate::codec::modulator::ModulatorConfig {
            bit_period: 2.0,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };

        let mut modulator = crate::codec::modulator::Modulator::new(&coupling, sm2, &mod_config)
            .expect("modulator");
        modulator.encode_with_system(&0, &chen).expect("encode");
        let signals = modulator.output_signals().to_vec();

        decoder.feed_signals(&signals).expect("feed");
        let symbol = decoder.decode().expect("decode");
        assert!(symbol < 2);
    }

    #[test]
    fn score_all_returns_all_symbols() {
        let (mut modulator, mut demodulator, chen) = setup_codec(2.0);

        modulator.encode_with_system(&0, &chen).expect("encode");
        let signals = modulator.output_signals().to_vec();

        demodulator.feed_signals(&signals).expect("feed");
        let scores = demodulator.score_all(&chen).expect("score_all");

        assert_eq!(scores.len(), 2, "should have scores for all symbols");
        // Scores should be sorted descending
        assert!(
            scores[0].1 >= scores[1].1,
            "scores should be sorted descending"
        );
    }

    #[test]
    fn single_symbol_detection_correct() {
        // Verify that each symbol is correctly detected on ideal channel.
        for test_symbol in [0usize, 1] {
            let (mut modulator, mut demodulator, chen) = setup_codec(5.0);

            modulator
                .encode_with_system(&test_symbol, &chen)
                .expect("encode");
            let signals = modulator.output_signals().to_vec();

            demodulator.feed_signals(&signals).expect("feed");
            let detected = demodulator.detect_symbol(&chen).expect("detect");

            assert_eq!(
                detected, test_symbol,
                "symbol {test_symbol} should be correctly detected"
            );
        }
    }

    #[test]
    fn sequence_detection_perfect_on_ideal_channel() {
        // With identical initial states and ideal channel,
        // detection should be 100% correct for any sequence.
        let (mut modulator, mut demodulator, chen) = setup_codec(10.0);

        let tx_symbols = vec![0, 1, 0, 1, 1, 0, 0, 1];
        let signals = modulator
            .encode_sequence(&tx_symbols, &chen)
            .expect("encode");

        let rx_symbols = demodulator
            .decode_sequence(&signals, tx_symbols.len(), &chen)
            .expect("decode");

        assert_eq!(
            tx_symbols, rx_symbols,
            "detection should be perfect on ideal channel\nTX: {:?}\nRX: {:?}",
            tx_symbols, rx_symbols
        );
    }
}
