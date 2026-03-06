# Plan: 512-Node Multi-Bit CLSK Example

## Goal

Create a working 512-node CLSK example that transmits multiple bits per symbol
(M-ary alphabet, log₂(M) bits/symbol). Provide helper functions to automatically
generate equitable partitions, epsilon values, and channel links for any ring(N).

---

## Phase 1: Ring Pattern Generator (`src/graph/ring_patterns.rs`)

**Goal:** Auto-generate M equitable partitions for ring(N).

For ring(N), equitable 2-cluster partitions follow a simple rule: assign nodes
to clusters with period P where P divides N. Each divisor of N gives a distinct
equitable partition:

| Divisor P | Pattern (first 8 nodes of ring(8))  | Description |
|-----------|--------------------------------------|-------------|
| 1         | [0,1,0,1,0,1,0,1]                   | Alternating |
| 2         | [0,0,1,1,0,0,1,1]                   | Pairs       |
| 4         | [0,0,0,0,1,1,1,1]                   | Quadruples  |

**Functions:**

```rust
/// Generate M distinct equitable 2-cluster partitions for ring(n).
/// Uses periodic assignment with different divisors of n.
pub fn generate_ring_partitions(n: usize, m: usize) -> Result<Vec<ClusterPattern>, GraphError>

/// Compute valid coupling strength range [ε_min, ε_max] for ring(n)
/// using MSF threshold and Laplacian eigenvalues.
pub fn coupling_range(n: usize) -> Result<(f64, f64), GraphError>

/// Generate M evenly-spaced epsilon values within the valid coupling range.
pub fn generate_epsilon_values(n: usize, m: usize) -> Result<Vec<f64>, GraphError>

/// Select channel link nodes that satisfy the covertness condition:
/// no two channel links share a cluster in ANY of the M patterns.
pub fn select_channel_links(patterns: &[ClusterPattern], count: usize) -> Result<Vec<usize>, GraphError>
```

**Key design decisions:**
- For ring(512), divisors include: 1, 2, 4, 8, 16, 32, 64, 128, 256
  → up to 9 distinct 2-cluster equitable partitions → supports up to M=9 (3+ bits)
- For M=4 (2 bits/symbol): use divisors 1, 2, 4, 8 → 4 patterns
- For M=8 (3 bits/symbol): use divisors 1, 2, 4, 8, 16, 32, 64, 128 → 8 patterns
- Epsilon values spaced evenly in the valid range

---

## Phase 2: Convenience Builder (`src/codec/multi_bit.rs`)

**Goal:** One-call setup for M-ary CLSK with automatic configuration.

```rust
/// Build a complete M-ary CLSK configuration for ring(n).
/// Returns (CouplingMatrix, SymbolMap, ModulatorConfig, FrameConfig, DemodulatorConfig).
pub fn build_mary_clsk(
    n: usize,           // number of nodes (e.g., 512)
    bits_per_symbol: usize,  // log₂(M), e.g., 2 for 4-ary
    bit_period: f64,    // T_b per symbol
    dt: f64,            // integration step
) -> Result<MaryClskConfig, CodecError>

pub struct MaryClskConfig {
    pub coupling: CouplingMatrix,
    pub symbol_map: SymbolMap,
    pub mod_config: ModulatorConfig,
    pub frame_config: FrameConfig,
    pub demod_config: DemodulatorConfig,
    pub bits_per_symbol: usize,
}
```

This internally calls Phase 1 functions to:
1. Generate M = 2^bits_per_symbol equitable partitions for ring(n)
2. Compute epsilon range and generate M epsilon values
3. Select channel link nodes
4. Build all config structs

---

## Phase 3: 512-Node Example Binary (`examples/multi_bit_512.rs`)

**Goal:** Runnable example demonstrating multi-bit transmission on a 512-node ring.

```
$ cargo run --example multi_bit_512

Multi-bit CLSK on ring(512)
Alphabet size: M=4 (2 bits/symbol)
Coupling range: ε ∈ [X.XX, Y.YY]
Epsilon values: [ε₀, ε₁, ε₂, ε₃]
Channel links: [a, b]

Transmitting 50 symbols (100 bits)...
TX: [2, 0, 3, 1, ...]
RX: [2, 0, 3, 1, ...]
SER: 0.0000
```

The example:
1. Calls `build_mary_clsk(512, 2, 10.0, 0.001)` for 4-ary / 2 bits per symbol
2. Creates Modulator + Demodulator
3. Generates random symbols, encodes, transmits (ideal channel), decodes
4. Reports SER

---

## Phase 4: Tests

1. **Unit tests in `ring_patterns.rs`:**
   - `generate_ring_partitions` returns valid equitable partitions
   - `coupling_range` computes correct range for known topologies (ring(8) should match [5.15, 17.46])
   - `select_channel_links` satisfies covertness condition
   - Error cases: M too large (not enough divisors), N too small

2. **Unit tests in `multi_bit.rs`:**
   - `build_mary_clsk` produces valid config for various (N, bits_per_symbol)
   - Round-trip: encode → decode with ideal channel gives 0% SER for M=4

3. **Integration test `tests/multi_bit_e2e.rs`:**
   - 512-node, M=4, 10-symbol sequence → 0% SER on ideal channel

---

## Ground Rules

- No changes to existing public APIs
- All new code follows CLAUDE.md rules (no unwrap in lib, thiserror, tests)
- `cargo test && cargo clippy -- -D warnings && cargo fmt --check` must pass
- One commit per phase
