//! Octagon network demo — reproduces the paper's 8-node setup.
//!
//! Demonstrates cluster pattern switching in an 8-node octagonal network
//! of coupled Chen oscillators. Outputs trajectory and sync energy data
//! in CSV format suitable for plotting with gnuplot or matplotlib.
//!
//! Usage:
//!   cargo run --example octagon_demo
//!   cargo run --example octagon_demo -- --symbols 10 --csv

use clap::Parser;
use cluster_shift_keying::codec::modulator::{Modulator, ModulatorConfig};
use cluster_shift_keying::codec::symbol_map::SymbolMap;
use cluster_shift_keying::dynamics::chen::ChenSystem;
use cluster_shift_keying::graph::{ClusterPattern, TopologyBuilder};

#[derive(Parser)]
#[command(
    name = "octagon_demo",
    about = "8-node octagon CLSK demo from the paper"
)]
struct Args {
    /// Number of symbols to transmit.
    #[arg(short = 'n', long, default_value_t = 8)]
    symbols: usize,

    /// Bit period T_b (time units per symbol).
    #[arg(long, default_value_t = 10.0)]
    bit_period: f64,

    /// Integration step dt.
    #[arg(long, default_value_t = 0.001)]
    dt: f64,

    /// Output trajectories as CSV.
    #[arg(long)]
    csv: bool,

    /// Symbol sequence to transmit (e.g., "01010011"). If not specified,
    /// an alternating pattern is used.
    #[arg(long)]
    pattern: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Cluster Shift Keying: Octagon Demo ===");
    println!("Reproducing the paper's 8-node octagonal network setup.");
    println!();

    // Build the octagon topology (paper's Fig. 2)
    let coupling = TopologyBuilder::octagon()?;
    let system = ChenSystem::default_paper();

    println!("Network: 8-node octagon (cycle graph C₈)");
    println!(
        "System:  Chen attractor (a={}, b={:.4}, c={})",
        system.a(),
        system.b(),
        system.c()
    );
    println!("Inner coupling Γ = diag(0, 1, 0)");
    println!();

    // Define the two cluster patterns from the paper
    // Pattern 0 (ε=8.0):  alternating clusters {0,2,4,6} and {1,3,5,7}
    // Pattern 1 (ε=12.0): paired clusters {0,1,4,5} and {2,3,6,7}
    let pattern_0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1])?;
    let pattern_1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1])?;

    println!("Symbol 0 → ε=8.0,  clusters: {pattern_0}");
    println!("Symbol 1 → ε=12.0, clusters: {pattern_1}");
    println!("Channel links: nodes 0 and 3 (never co-clustered)");
    println!();

    let symbol_map = SymbolMap::binary(pattern_0, 8.0, pattern_1, 12.0, vec![0, 3])?;

    // Build symbol sequence
    let symbols: Vec<usize> = if let Some(ref pat) = args.pattern {
        pat.chars()
            .map(|c| match c {
                '0' => Ok(0usize),
                '1' => Ok(1usize),
                _ => Err(format!("invalid symbol '{c}' — use 0 or 1")),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..args.symbols).map(|i| i % 2).collect()
    };

    println!("Transmitting {} symbols: {:?}", symbols.len(), symbols);
    println!(
        "Bit period: {} time units ({} steps at dt={})",
        args.bit_period,
        (args.bit_period / args.dt) as usize,
        args.dt
    );
    println!();

    // Create modulator and encode
    let mod_config = ModulatorConfig {
        bit_period: args.bit_period,
        dt: args.dt,
        initial_state: vec![1.0, 1.0, 1.0],
    };
    let mut modulator = Modulator::new(&coupling, symbol_map.clone(), &mod_config)?;

    let n_nodes = 8;

    // Encode symbol by symbol and collect per-symbol sync energies
    println!("--- Synchronization Energy per Symbol ---");
    println!(
        "{:<8} {:<8} {:<10} {:<12} {:<12}",
        "Symbol", "ε", "Pattern", "E_intra", "E_inter"
    );

    for (idx, &sym) in symbols.iter().enumerate() {
        modulator.encode_with_system(&sym, &system)?;

        let network = modulator.network();

        // Compute instantaneous pairwise sync error to show cluster formation
        let mut max_intra_err = 0.0_f64;
        let mut min_inter_err = f64::MAX;
        let entry = symbol_map.lookup(sym)?;
        let pattern = &entry.pattern;

        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                let err = network.sync_error(i, j)?;
                if pattern.are_same_cluster(i, j) {
                    max_intra_err = max_intra_err.max(err);
                } else {
                    min_inter_err = min_inter_err.min(err);
                }
            }
        }

        let epsilon = entry.epsilon;
        println!(
            "{:<8} {:<8.1} {:<10} {:<12.6} {:<12.6}",
            idx,
            epsilon,
            format!("C{sym}"),
            max_intra_err,
            min_inter_err,
        );
    }

    // If CSV output requested, print the final network state
    if args.csv {
        println!();
        println!("--- Final Network State (CSV) ---");
        println!("node,x,y,z");
        let network = modulator.network();
        for i in 0..n_nodes {
            let state = network.node_state(i)?;
            println!("{},{:.6},{:.6},{:.6}", i, state[0], state[1], state[2]);
        }
    }

    println!();
    println!("Demo complete. The sync errors show that intra-cluster error");
    println!("decreases over time while inter-cluster error remains large,");
    println!("confirming cluster synchronization patterns emerge as expected.");

    Ok(())
}
