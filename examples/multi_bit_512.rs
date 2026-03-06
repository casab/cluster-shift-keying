//! 512-node M-ary CLSK example with multi-bit symbols.
//!
//! Demonstrates M-ary Cluster Shift Keying on a ring(512) topology,
//! encoding 2 bits per symbol (M=4). Uses the `build_mary_clsk` convenience
//! builder for automatic partition generation and system setup.
//!
//! Usage:
//!   cargo run --example multi_bit_512
//!   cargo run --example multi_bit_512 -- --bits-per-symbol 3 --num-symbols 20

use clap::Parser;
use cluster_shift_keying::channel::{ChannelLink, IdealChannel};
use cluster_shift_keying::codec::multi_bit::{build_mary_clsk, MaryClskConfig};
use cluster_shift_keying::dynamics::chen::ChenSystem;
use cluster_shift_keying::metrics::ber::BerEvaluator;

#[derive(Parser)]
#[command(
    name = "multi_bit_512",
    about = "M-ary CLSK on ring(512) — multi-bit per symbol demo"
)]
struct Args {
    /// Number of nodes in the ring.
    #[arg(long, default_value_t = 512)]
    num_nodes: usize,

    /// Bits per symbol (M = 2^bits_per_symbol).
    #[arg(long, default_value_t = 2)]
    bits_per_symbol: usize,

    /// Number of symbols to transmit.
    #[arg(long, default_value_t = 10)]
    num_symbols: usize,

    /// Minimum coupling strength epsilon.
    #[arg(long, default_value_t = 5.0)]
    eps_min: f64,

    /// Maximum coupling strength epsilon.
    #[arg(long, default_value_t = 17.0)]
    eps_max: f64,

    /// Bit period T_b (time units per symbol).
    #[arg(long, default_value_t = 10.0)]
    bit_period: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let m = 1usize << args.bits_per_symbol;

    println!("=== M-ary Cluster Shift Keying: Ring({}) Demo ===", args.num_nodes);
    println!("Bits/symbol:  {} (M = {} symbols)", args.bits_per_symbol, m);
    println!("Epsilon range: [{}, {}]", args.eps_min, args.eps_max);
    println!("Bit period:   {} time units", args.bit_period);
    println!("Symbols:      {}", args.num_symbols);
    println!();

    // Build M-ary CLSK system using the convenience builder
    println!("Building ring({}) topology and CLSK system...", args.num_nodes);
    let config = MaryClskConfig {
        num_nodes: args.num_nodes,
        bits_per_symbol: args.bits_per_symbol,
        eps_min: args.eps_min,
        eps_max: args.eps_max,
        bit_period: args.bit_period,
        dt: 0.001,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let mut system = build_mary_clsk(&config)?;

    // Generate test symbols cycling through all M values
    let tx_symbols: Vec<usize> = (0..args.num_symbols).map(|i| i % m).collect();

    println!("TX symbols:   {:?}", tx_symbols);
    println!();

    // Encode
    let chen = ChenSystem::default_paper();
    println!("Encoding {} symbols...", tx_symbols.len());
    let tx_signals = system
        .modulator
        .encode_sequence(&tx_symbols, &chen)?;
    println!(
        "  Signal length: {} samples per link ({} links)",
        tx_signals[0].len(),
        tx_signals.len()
    );

    // Pass through ideal channel
    let num_links = tx_signals.len();
    let mut channel_link = ChannelLink::new(num_links)?;
    let mut rx_signals = vec![vec![]; num_links];
    let mut channel = IdealChannel::new();
    channel_link.transmit_all(&tx_signals, &mut rx_signals, &mut channel)?;

    // Decode
    println!("Decoding...");
    let rx_symbols = system
        .demodulator
        .decode_sequence(&rx_signals, tx_symbols.len(), &chen)?;

    println!("RX symbols:   {:?}", rx_symbols);
    println!();

    // Compute SER
    let ser = BerEvaluator::evaluate(&tx_symbols, &rx_symbols)?;
    let symbol_errors = tx_symbols
        .iter()
        .zip(rx_symbols.iter())
        .filter(|(a, b)| a != b)
        .count();

    println!("--- Results ---");
    println!("Symbol errors: {} / {}", symbol_errors, tx_symbols.len());
    println!("SER:           {:.6}", ser);
    println!(
        "Throughput:    {} bits/symbol ({} total info bits)",
        args.bits_per_symbol,
        args.bits_per_symbol * tx_symbols.len()
    );

    if ser == 0.0 {
        println!("\nPerfect transmission — all {} symbols recovered correctly!", tx_symbols.len());
    } else {
        println!("\n{} symbol error(s) detected.", symbol_errors);
    }

    Ok(())
}
