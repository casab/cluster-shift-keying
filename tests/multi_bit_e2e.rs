//! End-to-end integration tests for M-ary CLSK on ring topologies.

use cluster_shift_keying::channel::{ChannelLink, IdealChannel};
use cluster_shift_keying::codec::multi_bit::{build_mary_clsk, MaryClskConfig};
use cluster_shift_keying::dynamics::chen::ChenSystem;

/// Helper: run a full encode-channel-decode roundtrip on an ideal channel.
fn roundtrip(
    config: &MaryClskConfig,
    tx_symbols: &[usize],
) -> (Vec<usize>, f64) {
    let mut system = build_mary_clsk(config).expect("build system");
    let chen = ChenSystem::default_paper();

    // Encode
    let tx_signals = system
        .modulator
        .encode_sequence(tx_symbols, &chen)
        .expect("encode");

    // Ideal channel
    let num_links = tx_signals.len();
    let mut channel_link = ChannelLink::new(num_links).expect("channel link");
    let mut rx_signals = vec![vec![]; num_links];
    let mut channel = IdealChannel::new();
    channel_link
        .transmit_all(&tx_signals, &mut rx_signals, &mut channel)
        .expect("transmit");

    // Decode
    let rx_symbols = system
        .demodulator
        .decode_sequence(&rx_signals, tx_symbols.len(), &chen)
        .expect("decode");

    let errors = tx_symbols
        .iter()
        .zip(rx_symbols.iter())
        .filter(|(a, b)| a != b)
        .count();
    let ser = errors as f64 / tx_symbols.len() as f64;

    (rx_symbols, ser)
}

/// Binary (M=2) on ring(8), ideal channel, should achieve SER < 50%.
#[test]
fn binary_ring8_ideal_channel() {
    let config = MaryClskConfig {
        num_nodes: 8,
        bits_per_symbol: 1,
        eps_min: 5.0,
        eps_max: 17.0,
        bit_period: 10.0,
        dt: 0.001,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let tx_symbols = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let (rx_symbols, ser) = roundtrip(&config, &tx_symbols);

    assert!(
        ser < 0.5,
        "SER = {ser:.3} too high for binary ideal channel. TX: {tx_symbols:?}, RX: {rx_symbols:?}"
    );
}

/// 4-ary (M=4, 2 bits/symbol) on ring(16), ideal channel.
#[test]
fn quaternary_ring16_ideal_channel() {
    let config = MaryClskConfig {
        num_nodes: 16,
        bits_per_symbol: 2,
        eps_min: 5.0,
        eps_max: 17.0,
        bit_period: 10.0,
        dt: 0.001,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let tx_symbols = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let (rx_symbols, ser) = roundtrip(&config, &tx_symbols);

    // For M=4 random guessing gives SER=75%, so < 75% shows the detector works.
    assert!(
        ser < 0.75,
        "SER = {ser:.3} too high for 4-ary ideal channel. TX: {tx_symbols:?}, RX: {rx_symbols:?}"
    );
}

/// Verify that the convenience builder correctly sets alphabet size.
#[test]
fn mary_clsk_alphabet_sizes() {
    for bits in 1..=3 {
        let config = MaryClskConfig {
            num_nodes: 16,
            bits_per_symbol: bits,
            eps_min: 5.0,
            eps_max: 17.0,
            bit_period: 1.0,
            dt: 0.001,
            initial_state: vec![1.0, 1.0, 1.0],
        };
        let system = build_mary_clsk(&config).expect("build");
        assert_eq!(system.alphabet_size, 1 << bits);
        assert_eq!(system.bits_per_symbol, bits);
    }
}

/// Verify the receiver doesn't output all-zero symbols for M-ary.
#[test]
fn receiver_detects_nonzero_symbols() {
    let config = MaryClskConfig {
        num_nodes: 16,
        bits_per_symbol: 2,
        eps_min: 5.0,
        eps_max: 17.0,
        bit_period: 10.0,
        dt: 0.001,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let tx_symbols = vec![1, 2, 3, 1, 2, 3];
    let (rx_symbols, _) = roundtrip(&config, &tx_symbols);

    let has_nonzero = rx_symbols.iter().any(|&s| s != 0);
    assert!(
        has_nonzero,
        "receiver should detect some non-zero symbols, got: {rx_symbols:?}"
    );
}
