//! BER sweep — symbol error rate vs. noise level.
//!
//! Sweeps the noise standard deviation σ from 0 to a maximum value and
//! computes the symbol error rate (SER) at each level. Outputs CSV for
//! plotting with gnuplot or matplotlib.
//!
//! Usage:
//!   cargo run --example ber_sweep
//!   cargo run --example ber_sweep -- --sigma-max 2.0 --steps 10 --symbols 50
//!   cargo run --example ber_sweep -- --csv > ber_curve.csv

use clap::Parser;
use cluster_shift_keying::pipeline::config::SimulationConfig;
use cluster_shift_keying::pipeline::simulation::Simulation;

#[derive(Parser)]
#[command(name = "ber_sweep", about = "CLSK BER vs. noise sweep")]
struct Args {
    /// Maximum noise σ.
    #[arg(long, default_value_t = 1.0)]
    sigma_max: f64,

    /// Number of σ steps (including σ=0).
    #[arg(long, default_value_t = 6)]
    steps: usize,

    /// Number of symbols per trial.
    #[arg(short = 'n', long, default_value_t = 50)]
    symbols: usize,

    /// Bit period T_b.
    #[arg(long, default_value_t = 10.0)]
    bit_period: f64,

    /// Base RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output raw CSV only (no header text).
    #[arg(long)]
    csv: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let sigma_values: Vec<f64> = (0..args.steps)
        .map(|i| {
            if args.steps <= 1 {
                0.0
            } else {
                args.sigma_max * (i as f64) / ((args.steps - 1) as f64)
            }
        })
        .collect();

    if !args.csv {
        println!("=== Cluster Shift Keying: BER Sweep ===");
        println!("Symbols per trial: {}", args.symbols);
        println!("Bit period:        {} time units", args.bit_period);
        println!(
            "σ range:           [0.0, {:.3}] in {} steps",
            args.sigma_max, args.steps
        );
        println!("Seed:              {}", args.seed);
        println!();
    }

    // CSV header
    if args.csv {
        println!("sigma,ser,symbol_errors,total_symbols");
    } else {
        println!(
            "{:<12} {:<12} {:<15} {:<10}",
            "sigma", "SER", "errors", "total"
        );
        println!("{}", "-".repeat(49));
    }

    for &sigma in &sigma_values {
        let mut config = SimulationConfig::default_paper();
        config.codec.bit_period = args.bit_period;
        config.simulation.num_symbols = args.symbols;
        config.simulation.seed = args.seed;

        if sigma > 0.0 {
            config.channel.channel_type = "gaussian".to_string();
            config.channel.sigma = sigma;
            config.channel.noise_mode = "additive".to_string();
            config.channel.seed = args.seed;
        }

        let sim = Simulation::new(config)?;
        let result = sim.run()?;

        if args.csv {
            println!(
                "{:.6},{:.6},{},{}",
                sigma, result.ser, result.symbol_errors, result.total_symbols
            );
        } else {
            println!(
                "{:<12.6} {:<12.6} {:<15} {:<10}",
                sigma, result.ser, result.symbol_errors, result.total_symbols
            );
        }
    }

    if !args.csv {
        println!();
        println!("Use --csv flag to output raw CSV for plotting.");
        println!("Example: cargo run --example ber_sweep -- --csv > ber_curve.csv");
    }

    Ok(())
}
