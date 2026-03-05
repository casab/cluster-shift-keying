# Cluster Shift Keying (CLSK)

Rust implementation of **Cluster Shift Keying**, a chaos-based communication scheme that encodes information into spatio-temporal synchronization patterns of coupled chaotic networks.

Based on the paper:

> **Cluster Shift Keying: Covert Transmission of Information via Cluster Synchronization in Chaotic Networks**
> Zekeriya Sari, Serkan Gunel
> *Physica Scripta*, Volume 99, 035204 (2024)
> Preprint: https://arxiv.org/abs/2312.04593

## How It Works

CLSK encodes symbols by switching between different cluster synchronization patterns in a network of coupled chaotic oscillators. Each symbol in an M-ary alphabet maps to a specific coupling strength, which produces a distinct cluster pattern in the network.

1. **Transmitter** drives a coupled chaotic network (e.g., 8 Chen oscillators on an octagon topology) and switches coupling strength per symbol
2. **Channel links** carry signals between transmitter and receiver subnetworks -- these signals remain chaotic and covert regardless of the symbol being transmitted
3. **Receiver** detects the cluster synchronization pattern via pairwise sync energy and decodes the symbol

The covertness condition ensures that channel link nodes are never co-clustered for any symbol, making the transmitted signals indistinguishable from noise to an eavesdropper without knowledge of the network topology.

## Project Structure

```
src/
  dynamics/    Chaotic systems (Chen attractor), ODE solvers (RK4)
  linalg/      Matrix operations, eigendecomposition (via nalgebra)
  graph/       Network topology, coupling matrices, symmetry, cluster partitions
  sync/        Master stability function, coupled network simulation, cluster detection
  codec/       Symbol mapping, CLSK modulator/demodulator, framing
  channel/     Channel models (ideal, Gaussian with additive/multiplicative noise)
  metrics/     Sync energy detector, BER evaluator, Monte Carlo simulation framework
  pipeline/    End-to-end simulation config (TOML) and orchestration
  utils/       Seeded RNG, shared helpers
```

## Build & Test

```bash
cargo build                    # Build the library
cargo test                     # Run all tests (unit + integration)
cargo clippy -- -D warnings    # Lint (must pass with zero warnings)
cargo fmt --check              # Format check
```

## Configuration

Simulations are configured via TOML. Example:

```toml
[system]
system_type = "chen"
a = 35.0
b = 2.6666666666666665
c = 28.0

[topology]
topology_type = "octagon"
node_count = 8

[coupling]
channel_links = [0, 3]

[[coupling.symbols]]
epsilon = 8.0
pattern = [0, 1, 0, 1, 0, 1, 0, 1]

[[coupling.symbols]]
epsilon = 12.0
pattern = [0, 0, 1, 1, 0, 0, 1, 1]

[codec]
bit_period = 10.0
dt = 0.001
initial_state = [1.0, 1.0, 1.0]

[channel]
channel_type = "ideal"

[simulation]
num_symbols = 100
seed = 12345
```

## Quick Start

```rust
use cluster_shift_keying::pipeline::{SimulationConfig, Simulation};

let config = SimulationConfig::default_paper();
let sim = Simulation::new(config).expect("create simulation");
let result = sim.run().expect("run simulation");

println!("Transmitted: {:?}", &result.tx_symbols[..5]);
println!("Received:    {:?}", &result.rx_symbols[..5]);
println!("SER: {:.4}", result.ser);
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Chen (a, b, c) | (35.0, 8/3, 28.0) | Standard Chen attractor parameters |
| Inner coupling | diag(0, 1, 0) | Coupling through y-component |
| Octagon coupling range | [5.15, 17.46] | Valid coupling strengths for cluster sync |
| MSF threshold | -4.2 | Master stability function zero crossing |
| dt | 0.001 | ODE integration step size |
| Bit period | 10.0 | Symbol duration in time units |

## License

GPL-2.0
