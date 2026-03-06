//! BER sweep — symbol error rate vs. noise level.
//!
//! Sweeps the noise standard deviation σ from 0 to a maximum value and
//! computes the symbol error rate (SER) at each level. Outputs CSV for
//! plotting with gnuplot or matplotlib.
//!
//! When built with `--features parallel`, sigma values are evaluated
//! concurrently using rayon for significant speedup on multi-core systems.
//!
//! Usage:
//!   cargo run --example ber_sweep
//!   cargo run --example ber_sweep -- --sigma-max 2.0 --steps 10 --symbols 50
//!   cargo run --example ber_sweep -- --csv > ber_curve.csv
//!   cargo run --features parallel --example ber_sweep -- --steps 20

use clap::Parser;
use cluster_shift_keying::pipeline::config::SimulationConfig;
use cluster_shift_keying::pipeline::simulation::Simulation;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

/// Result of a single sigma trial.
struct TrialResult {
    sigma: f64,
    ser: f64,
    symbol_errors: usize,
    total_symbols: usize,
}

fn run_trial(
    sigma: f64,
    bit_period: f64,
    symbols: usize,
    seed: u64,
) -> Result<TrialResult, String> {
    let mut config = SimulationConfig::default_paper();
    config.codec.bit_period = bit_period;
    config.simulation.num_symbols = symbols;
    config.simulation.seed = seed;

    if sigma > 0.0 {
        config.channel.channel_type = "gaussian".to_string();
        config.channel.sigma = sigma;
        config.channel.noise_mode = "additive".to_string();
        config.channel.seed = seed;
    }

    let sim = Simulation::new(config).map_err(|e| e.to_string())?;
    let result = sim.run().map_err(|e| e.to_string())?;

    Ok(TrialResult {
        sigma,
        ser: result.ser,
        symbol_errors: result.symbol_errors,
        total_symbols: result.total_symbols,
    })
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
        #[cfg(feature = "parallel")]
        println!("Mode:              parallel (rayon)");
        #[cfg(not(feature = "parallel"))]
        println!("Mode:              sequential");
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

    // Run trials — parallel when feature is enabled, sequential otherwise.
    #[cfg(feature = "parallel")]
    let results: Vec<TrialResult> = {
        let collected: Result<Vec<_>, String> = sigma_values
            .par_iter()
            .map(|&sigma| run_trial(sigma, args.bit_period, args.symbols, args.seed))
            .collect();
        collected.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<TrialResult> = {
        let collected: Result<Vec<_>, String> = sigma_values
            .iter()
            .map(|&sigma| run_trial(sigma, args.bit_period, args.symbols, args.seed))
            .collect();
        collected.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
    };

    // Print results in sigma order (parallel may return out of order)
    for result in &results {
        if args.csv {
            println!(
                "{:.6},{:.6},{},{}",
                result.sigma, result.ser, result.symbol_errors, result.total_symbols
            );
        } else {
            println!(
                "{:<12.6} {:<12.6} {:<15} {:<10}",
                result.sigma, result.ser, result.symbol_errors, result.total_symbols
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
