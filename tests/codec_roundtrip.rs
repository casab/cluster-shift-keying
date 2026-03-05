use cluster_shift_keying::channel::{ChannelLink, IdealChannel};
use cluster_shift_keying::codec::modulator::{Modulator, ModulatorConfig};
use cluster_shift_keying::codec::symbol_map::SymbolMap;
use cluster_shift_keying::codec::{Demodulator, DemodulatorConfig, FrameConfig};
use cluster_shift_keying::dynamics::chen::ChenSystem;
use cluster_shift_keying::graph::{ClusterPattern, TopologyBuilder};
use cluster_shift_keying::metrics::sync_energy::RatioScoring;

fn setup_binary_codec(bit_period: f64) -> (Modulator, Demodulator, ChenSystem, SymbolMap) {
    let chen = ChenSystem::default_paper();
    let coupling = TopologyBuilder::octagon().expect("octagon");

    let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
    let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");
    let sm = SymbolMap::binary(p0, 8.0, p1, 12.0, vec![0, 3]).expect("sm");

    let mod_config = ModulatorConfig {
        bit_period,
        dt: 0.001,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let modulator = Modulator::new(&coupling, sm.clone(), &mod_config).expect("modulator");

    let frame_config = FrameConfig::new(bit_period, 0.0, 0.001).expect("frame config");
    let demod_config = DemodulatorConfig::default();
    let demodulator = Demodulator::new(
        &coupling,
        sm.clone(),
        frame_config,
        Box::new(RatioScoring::default()),
        &demod_config,
    )
    .expect("demodulator");

    (modulator, demodulator, chen, sm)
}

/// Verify that the receiver doesn't produce all-zeros.
#[test]
fn receiver_does_not_output_all_zeros() {
    let (mut modulator, mut demodulator, chen, sm) = setup_binary_codec(10.0);

    // Transmit a mix of 0s and 1s
    let tx_symbols = vec![0, 1, 0, 1, 0, 1, 0, 1];
    let tx_signals = modulator
        .encode_sequence(&tx_symbols, &chen)
        .expect("encode");

    // Pass through ideal channel
    let num_links = sm.channel_links().len();
    let mut channel_link = ChannelLink::new(num_links).expect("channel link");
    let mut rx_signals = vec![vec![]; num_links];
    let mut channel = IdealChannel::new();
    channel_link
        .transmit_all(&tx_signals, &mut rx_signals, &mut channel)
        .expect("transmit");

    // Decode
    let rx_symbols = demodulator
        .decode_sequence(&rx_signals, tx_symbols.len(), &chen)
        .expect("decode");

    // The received symbols should not ALL be zeros
    let has_ones = rx_symbols.iter().any(|&s| s == 1);
    assert!(
        has_ones,
        "receiver should detect some symbol-1s, got all zeros: {:?}",
        rx_symbols
    );
}

/// Verify that noiseless roundtrip achieves reasonable SER.
#[test]
fn noiseless_roundtrip_reasonable_ser() {
    let (mut modulator, mut demodulator, chen, sm) = setup_binary_codec(10.0);

    let tx_symbols = vec![0, 1, 1, 0, 1, 0, 0, 1];
    let tx_signals = modulator
        .encode_sequence(&tx_symbols, &chen)
        .expect("encode");

    let num_links = sm.channel_links().len();
    let mut channel_link = ChannelLink::new(num_links).expect("channel link");
    let mut rx_signals = vec![vec![]; num_links];
    let mut channel = IdealChannel::new();
    channel_link
        .transmit_all(&tx_signals, &mut rx_signals, &mut channel)
        .expect("transmit");

    let rx_symbols = demodulator
        .decode_sequence(&rx_signals, tx_symbols.len(), &chen)
        .expect("decode");

    let errors = tx_symbols
        .iter()
        .zip(rx_symbols.iter())
        .filter(|(a, b)| a != b)
        .count();
    let ser = errors as f64 / tx_symbols.len() as f64;

    // With bit_period=10.0 on ideal channel, SER should be well below 50%
    // (random guessing is 50% for binary)
    assert!(
        ser < 0.5,
        "SER = {ser:.3} ({errors}/{}) is too high for noiseless channel. \
         TX: {:?}, RX: {:?}",
        tx_symbols.len(),
        tx_symbols,
        rx_symbols
    );
}
