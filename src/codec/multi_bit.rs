use super::demodulator::{Demodulator, DemodulatorConfig};
use super::error::CodecError;
use super::framing::FrameConfig;
use super::modulator::{Modulator, ModulatorConfig};
use super::symbol_map::SymbolMap;
use crate::graph::ring_patterns::{build_ring_clsk, RingClskConfig};
use crate::metrics::sync_energy::RatioScoring;

/// High-level configuration for M-ary CLSK on a ring topology.
///
/// This is the simplest way to set up multi-bit CLSK. Specify the ring size,
/// bits per symbol, coupling range, and timing — the builder handles
/// partition generation, epsilon spacing, channel link selection, and
/// modulator/demodulator construction.
#[derive(Debug, Clone)]
pub struct MaryClskConfig {
    /// Number of nodes in the ring.
    pub num_nodes: usize,
    /// Bits encoded per symbol (M = 2^bits_per_symbol symbols).
    pub bits_per_symbol: usize,
    /// Minimum coupling strength ε.
    pub eps_min: f64,
    /// Maximum coupling strength ε.
    pub eps_max: f64,
    /// Bit period T_b in time units.
    pub bit_period: f64,
    /// Integration time step dt.
    pub dt: f64,
    /// Initial state for all oscillators.
    pub initial_state: Vec<f64>,
}

impl Default for MaryClskConfig {
    fn default() -> Self {
        Self {
            num_nodes: 512,
            bits_per_symbol: 2,
            eps_min: 5.0,
            eps_max: 17.0,
            bit_period: 10.0,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        }
    }
}

/// A fully configured M-ary CLSK system ready for modulation/demodulation.
pub struct MaryClskSystem {
    /// The modulator (transmitter side).
    pub modulator: Modulator,
    /// The demodulator (receiver side).
    pub demodulator: Demodulator,
    /// Bits per symbol (log₂ M).
    pub bits_per_symbol: usize,
    /// Number of symbols in the alphabet (M).
    pub alphabet_size: usize,
}

/// Build a complete M-ary CLSK modulator and demodulator from a high-level config.
///
/// This is the recommended way to set up CLSK for ring topologies. It:
/// 1. Generates the ring topology and equitable partition
/// 2. Computes M evenly-spaced epsilon values
/// 3. Selects channel link nodes satisfying covertness
/// 4. Constructs both modulator and demodulator with matching parameters
pub fn build_mary_clsk(config: &MaryClskConfig) -> Result<MaryClskSystem, CodecError> {
    let ring_config = build_ring_clsk(
        config.num_nodes,
        config.bits_per_symbol,
        config.eps_min,
        config.eps_max,
    )
    .map_err(|e| CodecError::InvalidSymbolMap {
        reason: format!("ring CLSK setup failed: {e}"),
    })?;

    build_from_ring_config(ring_config, config)
}

/// Build modulator and demodulator from an existing `RingClskConfig`.
///
/// Use this when you need more control over the ring configuration
/// (e.g., custom epsilon spacing or channel links).
pub fn build_from_ring_config(
    ring_config: RingClskConfig,
    config: &MaryClskConfig,
) -> Result<MaryClskSystem, CodecError> {
    let symbol_map = SymbolMap::new(ring_config.entries, ring_config.channel_links)?;
    let alphabet_size = symbol_map.alphabet_size();

    let mod_config = ModulatorConfig {
        bit_period: config.bit_period,
        dt: config.dt,
        initial_state: config.initial_state.clone(),
    };

    let demod_config = DemodulatorConfig {
        initial_state: config.initial_state.clone(),
    };

    let frame_config = FrameConfig {
        bit_period: config.bit_period,
        guard_interval: 0.0,
        dt: config.dt,
        preamble: Vec::new(),
    };

    let modulator = Modulator::new(&ring_config.coupling, symbol_map.clone(), &mod_config)?;
    let demodulator = Demodulator::new(
        &ring_config.coupling,
        symbol_map,
        frame_config,
        Box::new(RatioScoring { epsilon: 1e-12 }),
        &demod_config,
    )?;

    Ok(MaryClskSystem {
        modulator,
        demodulator,
        bits_per_symbol: ring_config.bits_per_symbol,
        alphabet_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::chen::ChenSystem;

    #[test]
    fn build_mary_clsk_default() {
        let config = MaryClskConfig::default();
        let system = build_mary_clsk(&config).expect("default config");
        assert_eq!(system.alphabet_size, 4); // 2^2
        assert_eq!(system.bits_per_symbol, 2);
    }

    #[test]
    fn build_mary_clsk_ring16_2bit() {
        let config = MaryClskConfig {
            num_nodes: 16,
            bits_per_symbol: 2,
            eps_min: 5.0,
            eps_max: 17.0,
            bit_period: 1.0,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };
        let system = build_mary_clsk(&config).expect("ring16 2-bit");
        assert_eq!(system.alphabet_size, 4);
    }

    #[test]
    fn build_mary_clsk_encodes() {
        let config = MaryClskConfig {
            num_nodes: 8,
            bits_per_symbol: 1,
            eps_min: 5.0,
            eps_max: 17.0,
            bit_period: 0.5,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };
        let mut system = build_mary_clsk(&config).expect("ring8 1-bit");
        let chen = ChenSystem::default_paper();
        system
            .modulator
            .encode_with_system(&0, &chen)
            .expect("encode");
        let signals = system.modulator.output_signals();
        assert_eq!(signals.len(), 2); // 2 channel links
        assert_eq!(signals[0].len(), 500); // 0.5/0.001
    }

    #[test]
    fn build_mary_clsk_odd_n_error() {
        let config = MaryClskConfig {
            num_nodes: 7,
            ..MaryClskConfig::default()
        };
        assert!(build_mary_clsk(&config).is_err());
    }
}
