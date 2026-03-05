//! Basic CLSK transmission example.
//!
//! Transmits a short ASCII message ("Hello") through the Cluster Shift Keying
//! system and recovers it at the receiver. Uses the lower-level modulator/
//! demodulator API to transmit specific bits.
//!
//! Usage:
//!   cargo run --example basic_transmission
//!   cargo run --example basic_transmission -- --message "Hi" --bit-period 10.0

use clap::Parser;
use cluster_shift_keying::channel::{ChannelLink, IdealChannel};
use cluster_shift_keying::codec::modulator::{Modulator, ModulatorConfig};
use cluster_shift_keying::codec::symbol_map::SymbolMap;
use cluster_shift_keying::codec::{Demodulator, DemodulatorConfig, FrameConfig};
use cluster_shift_keying::dynamics::chen::ChenSystem;
use cluster_shift_keying::graph::{ClusterPattern, TopologyBuilder};
use cluster_shift_keying::metrics::ber::BerEvaluator;
use cluster_shift_keying::metrics::sync_energy::RatioScoring;

#[derive(Parser)]
#[command(name = "basic_transmission", about = "CLSK basic transmission demo")]
struct Args {
    /// Message to transmit (ASCII).
    #[arg(short, long, default_value = "Hello")]
    message: String,

    /// Bit period T_b (time units per symbol).
    #[arg(long, default_value_t = 10.0)]
    bit_period: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Convert message to binary symbols (one bit = one CLSK symbol)
    let tx_symbols: Vec<usize> = args
        .message
        .as_bytes()
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| ((byte >> i) & 1) as usize))
        .collect();

    println!("=== Cluster Shift Keying: Basic Transmission ===");
    println!("Message:    \"{}\"", args.message);
    println!(
        "Bits:       {} ({} bits)",
        format_bits(&tx_symbols),
        tx_symbols.len()
    );
    println!("Bit period: {} time units", args.bit_period);
    println!();

    // Build system and topology
    let system = ChenSystem::default_paper();
    let coupling = TopologyBuilder::octagon()?;

    // Paper's two cluster patterns for binary CLSK
    let pattern_0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1])?;
    let pattern_1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1])?;
    let channel_links = vec![0, 3];
    let symbol_map = SymbolMap::binary(pattern_0, 8.0, pattern_1, 12.0, channel_links)?;

    // Create modulator
    let mod_config = ModulatorConfig {
        bit_period: args.bit_period,
        dt: 0.001,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let mut modulator = Modulator::new(&coupling, symbol_map.clone(), &mod_config)?;

    // Create demodulator
    let frame_config = FrameConfig::new(args.bit_period, 0.0, 0.001)?;
    let demod_config = DemodulatorConfig {
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let mut demodulator = Demodulator::new(
        &coupling,
        symbol_map.clone(),
        frame_config,
        Box::new(RatioScoring::default()),
        &demod_config,
    )?;

    // Encode the specific message bits
    println!("Encoding {} symbols...", tx_symbols.len());
    let tx_signals = modulator.encode_sequence(&tx_symbols, &system)?;

    // Pass through ideal (noiseless) channel
    let num_links = symbol_map.channel_links().len();
    let mut channel_link = ChannelLink::new(num_links)?;
    let mut rx_signals = vec![vec![]; num_links];
    let mut channel = IdealChannel::new();
    channel_link.transmit_all(&tx_signals, &mut rx_signals, &mut channel)?;

    // Decode
    println!("Decoding...");
    let rx_symbols = demodulator.decode_sequence(&rx_signals, tx_symbols.len(), &system)?;

    // Compute SER
    let ser = BerEvaluator::evaluate(&tx_symbols, &rx_symbols)?;
    let symbol_errors = tx_symbols
        .iter()
        .zip(rx_symbols.iter())
        .filter(|(a, b)| a != b)
        .count();

    // Reconstruct received message
    let rx_message = bits_to_string(&rx_symbols);

    println!();
    println!("--- Results ---");
    println!("TX bits: {}", format_bits(&tx_symbols));
    println!("RX bits: {}", format_bits(&rx_symbols));
    println!();
    println!("TX message: \"{}\"", args.message);
    println!("RX message: \"{}\"", rx_message);
    println!();
    println!("Symbol errors: {} / {}", symbol_errors, tx_symbols.len());
    println!("SER:           {:.6}", ser);

    if ser == 0.0 {
        println!("\nPerfect transmission — all symbols recovered correctly!");
    } else {
        println!("\n{} symbol error(s) detected.", symbol_errors);
    }

    Ok(())
}

fn format_bits(bits: &[usize]) -> String {
    bits.iter()
        .enumerate()
        .map(|(i, &b)| {
            if i > 0 && i % 8 == 0 {
                format!(" {b}")
            } else {
                format!("{b}")
            }
        })
        .collect()
}

fn bits_to_string(bits: &[usize]) -> String {
    bits.chunks(8)
        .filter(|chunk| chunk.len() == 8)
        .map(|chunk| {
            let byte = chunk
                .iter()
                .fold(0u8, |acc, &bit| (acc << 1) | (bit as u8 & 1));
            byte as char
        })
        .collect()
}
