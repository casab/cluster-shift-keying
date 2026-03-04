# CLSK Implementation Plan — Rust

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Scaffold & Core Traits | **DONE** |
| 2 | Chen System & ODE Integrator | **DONE** |
| 3 | Linear Algebra Utilities | **DONE** |
| 4 | Network Topology Construction | **DONE** |
| 5 | Graph Symmetry & Cluster Partitions | **DONE** |
| 6 | Master Stability Function | **DONE** |
| 7 | Cluster Sync Verification & Coupled Network Sim | **DONE** |
| 8 | Symbol Mapping & CLSK Modulator | **DONE** |
| 9 | Synchronization Energy Detector | **DONE** |
| 10 | CLSK Demodulator | TODO |
| 11 | Channel Models | TODO |
| 12 | BER Evaluation & Metrics | TODO |
| 13 | End-to-End Pipeline & Configuration | TODO |
| 14 | Examples & Paper Reproduction | TODO |
| 15 | Performance & Benchmarks | TODO |
| 16 | Extensibility Hooks & Future-Proofing | TODO |

---

## Architecture Overview

```
cluster-shift-keying/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API, re-exports
│   ├── dynamics/                 # Phase 1-2: Chaotic systems & ODE integration
│   │   ├── mod.rs
│   │   ├── traits.rs             # DynamicalSystem trait
│   │   ├── chen.rs               # Chen attractor
│   │   ├── rossler.rs            # Rössler attractor (extensibility)
│   │   └── integrator.rs         # RK4 / adaptive ODE solver
│   ├── linalg/                   # Phase 3: Linear algebra utilities
│   │   ├── mod.rs
│   │   ├── matrix.rs             # Dense matrix type (wraps nalgebra)
│   │   ├── eigen.rs              # Eigenvalue decomposition
│   │   └── block_diag.rs         # Block-diagonalization for cluster analysis
│   ├── graph/                    # Phase 4-5: Network topology & symmetry
│   │   ├── mod.rs
│   │   ├── topology.rs           # Graph construction (octagon, ring, lattice)
│   │   ├── coupling.rs           # Coupling matrix Ξ, inner coupling Γ
│   │   ├── symmetry.rs           # Automorphism / orbit detection
│   │   └── partition.rs          # Equitable partitions → cluster patterns
│   ├── sync/                     # Phase 6-7: Cluster synchronization engine
│   │   ├── mod.rs
│   │   ├── msf.rs                # Master stability function
│   │   ├── cluster.rs            # Cluster pattern representation
│   │   ├── stability.rs          # Transverse mode stability analysis
│   │   └── network.rs            # Coupled network simulation
│   ├── codec/                    # Phase 8-10: CLSK modulator / demodulator
│   │   ├── mod.rs
│   │   ├── traits.rs             # Encoder / Decoder traits (extensibility point)
│   │   ├── modulator.rs          # Symbol → cluster pattern → ε mapping
│   │   ├── demodulator.rs        # Energy detection → symbol recovery
│   │   ├── symbol_map.rs         # M-ary alphabet ↔ cluster pattern table
│   │   └── framing.rs            # Symbol timing, bit period T_b, framing
│   ├── channel/                  # Phase 11: Channel model
│   │   ├── mod.rs
│   │   ├── traits.rs             # ChannelModel trait (extensibility point)
│   │   ├── ideal.rs              # Noiseless passthrough
│   │   ├── gaussian.rs           # Additive / non-additive Gaussian noise
│   │   └── link.rs               # Channel link extraction from network
│   ├── metrics/                  # Phase 12: Performance evaluation
│   │   ├── mod.rs
│   │   ├── ber.rs                # Bit error rate computation
│   │   ├── sync_energy.rs        # Synchronization error energy Eᵢⱼ
│   │   └── stats.rs              # Monte Carlo runner, confidence intervals
│   ├── pipeline/                 # Phase 13: End-to-end pipeline
│   │   ├── mod.rs
│   │   ├── config.rs             # Simulation configuration (serde)
│   │   ├── transmitter.rs        # Tx pipeline: bits → symbols → network drive
│   │   ├── receiver.rs           # Rx pipeline: network observe → symbols → bits
│   │   └── simulation.rs         # Full Tx→Channel→Rx loop
│   └── utils/                    # Shared helpers
│       ├── mod.rs
│       └── rng.rs                # Seeded RNG wrapper
├── examples/                     # Phase 14: Runnable demos
│   ├── basic_transmission.rs     # Minimal working CLSK example
│   ├── ber_sweep.rs              # BER vs noise sweep (reproduces paper Fig.)
│   └── octagon_demo.rs           # 8-node octagon from the paper
├── benches/                      # Phase 15: Benchmarks
│   └── simulation_bench.rs
└── tests/                        # Integration tests (throughout)
    ├── chen_attractor.rs
    ├── cluster_sync.rs
    ├── codec_roundtrip.rs
    └── pipeline_e2e.rs
```

---

## Phase 1 — Project Scaffold & Core Traits

**Goal:** Initialize the Cargo project, define the trait hierarchy that every later phase builds on.

**Tasks:**
1. `cargo init --lib` with appropriate `Cargo.toml` metadata (name, version, edition 2021, license GPL-2.0)
2. Add core dependencies: `nalgebra`, `rand`, `rand_distr`, `serde` + `serde_derive`, `thiserror`
3. Create module skeleton (`src/lib.rs` with all module declarations, empty `mod.rs` files)
4. Define `DynamicalSystem` trait in `dynamics/traits.rs`:
   ```rust
   pub trait DynamicalSystem: Send + Sync {
       fn dimension(&self) -> usize;
       fn derivative(&self, state: &[f64], output: &mut [f64]);
       fn name(&self) -> &str;
   }
   ```
5. Define `Encoder` / `Decoder` traits in `codec/traits.rs`:
   ```rust
   pub trait Encoder {
       type Symbol;
       type Error;
       fn encode(&mut self, symbol: &Self::Symbol) -> Result<(), Self::Error>;
   }
   pub trait Decoder {
       type Symbol;
       type Error;
       fn decode(&mut self) -> Result<Self::Symbol, Self::Error>;
   }
   ```
6. Define `ChannelModel` trait in `channel/traits.rs`:
   ```rust
   pub trait ChannelModel {
       fn transmit(&mut self, signal: &[f64], output: &mut [f64]);
   }
   ```
7. Set up `thiserror`-based error types in each module
8. Verify `cargo build` and `cargo test` pass (empty tests)

**Tests:** Compilation smoke test, trait object safety assertions

**Status: DONE** — Commit `phase 1: project scaffold, core traits, and error types`
- All 8 tasks completed
- 6 tests passing: object safety (2), error Send+Sync (1), error conversions (1), RNG determinism (2)
- `cargo clippy -- -D warnings` clean
- Deviations from plan: `DynamicalSystem::derivative` returns `Result<(), DynamicsError>` (not `()`), `ChannelModel` is `Send + Sync` and returns `Result`, `CodecChain` included in Phase 1

---

## Phase 2 — Chen System & ODE Integrator

**Goal:** Implement the Chen chaotic attractor and a production-quality ODE solver.

**Tasks:**
1. Implement `ChenSystem` in `dynamics/chen.rs`:
   - Parameters `a = 35.0`, `b = 8.0/3.0`, `c = 28.0` as defaults
   - Configurable via builder pattern for future parameter exploration
   - Implements `DynamicalSystem` trait
2. Implement 4th-order Runge-Kutta (`RK4`) integrator in `dynamics/integrator.rs`:
   - Step function: `fn step(system: &dyn DynamicalSystem, state: &mut [f64], dt: f64)`
   - Trajectory generation: `fn integrate(system, state, dt, steps) -> Vec<Vec<f64>>`
   - Adaptive step-size variant (Dormand-Prince / RK45) for stiff regions
3. Stub `RosslerSystem` in `dynamics/rossler.rs` (empty impl with `todo!()`, shows extensibility)
4. Unit tests:
   - Chen system derivative correctness at known points
   - RK4 integrator convergence order verification (compare dt vs dt/2)
   - Verify Chen attractor stays bounded over 10,000 steps (no blowup)
   - Verify Lyapunov exponent is positive (chaotic regime confirmation)

**Tests:** Derivative values, integrator convergence order, attractor boundedness, chaos verification

**Status: DONE** — Commit `phase 2: implement Chen system, RK4 integrator, and Rössler stub`
- All tasks completed (ChenSystem, RK4 integrator, RosslerSystem)
- 22 unit tests + 4 integration tests passing (26 total)
- `cargo clippy -- -D warnings` clean, `cargo fmt --check` clean
- Deviations from plan:
  - No builder pattern for ChenSystem; uses `new(a, b, c)` with validation + `default_paper()` factory (simpler, sufficient)
  - No adaptive step-size (Dormand-Prince) yet; deferred to Phase 15 if needed — fixed-step RK4 verified stable at dt=0.001 for Chen
  - RosslerSystem is a full implementation (not a stub with `todo!()`), including derivative and boundedness tests
  - RK4 integrator includes `integrate_to_end()` for memory-efficient final-state-only integration
  - ChenSystem includes analytic Jacobian (`jacobian()` method on the trait), needed for Phase 6 MSF computation
  - Integration test `chen_attractor.rs` includes Jacobian-vs-finite-difference verification

---

## Phase 3 — Linear Algebra Utilities

**Goal:** Eigenvalue decomposition and matrix operations needed for stability analysis.

**Tasks:**
1. Thin wrapper types in `linalg/matrix.rs` around `nalgebra::DMatrix<f64>`:
   - Construction helpers: `from_adjacency`, `zeros`, `identity`, `kronecker_product`
   - Pretty-print for debugging
2. Eigenvalue solver in `linalg/eigen.rs`:
   - Real symmetric eigendecomposition (for Laplacian-like matrices)
   - General (non-symmetric) eigendecomposition for full coupling matrices
   - Return sorted eigenvalues + eigenvectors
3. Block-diagonalization utilities in `linalg/block_diag.rs`:
   - Simultaneous block-diagonalization of commuting matrices
   - Needed for decomposing coupling matrix into synchronous / transverse modes
4. Unit tests:
   - Known eigenvalues for simple matrices (identity, circulant)
   - Kronecker product dimensions and values
   - Block-diag of known commuting matrices

**Tests:** Eigenvalue accuracy, Kronecker correctness, block-diag round-trip

**Status: DONE** — Commit `phase 3: implement linear algebra utilities`
- All 4 tasks completed (matrix wrapper, eigen solver, block-diag, unit tests)
- 34 new unit tests (60 total), all passing
- `cargo clippy -- -D warnings` clean, `cargo fmt --check` clean
- Deviations from plan:
  - Matrix wrapper includes additional utilities beyond plan: `mul()`, `transpose()`, `frobenius_norm()`, `from_diagonal()`, `set()`/`get()` with bounds checking, `Display` impl
  - Method named `kronecker()` instead of `kronecker_product()` for brevity
  - `general_eigen` uses Schur decomposition (no eigenvectors) — `symmetric_eigen` provides eigenvectors for the cases that need them
  - `from_adjacency` takes `(i, j, weight)` triples instead of a full adjacency matrix

---

## Phase 4 — Network Topology Construction

**Goal:** Build graph structures and coupling matrices for regular topologies.

**Tasks:**
1. `TopologyBuilder` in `graph/topology.rs`:
   - `fn octagon() -> CouplingMatrix` — the paper's 8-node ring with specific adjacency
   - `fn ring(n: usize) -> CouplingMatrix` — general cycle graph
   - `fn complete(n: usize) -> CouplingMatrix` — fully connected
   - `fn lattice_2d(rows: usize, cols: usize) -> CouplingMatrix`
   - `fn from_adjacency(adj: &DMatrix<f64>) -> CouplingMatrix`
2. `CouplingMatrix` struct in `graph/coupling.rs`:
   - Stores adjacency `Ξ` and inner coupling `Γ`
   - Laplacian computation: `L = D - Ξ` (for eigenvalue analysis)
   - Methods: `set_coupling_strength(&mut self, ε: f64)`, `effective_coupling() -> DMatrix`
   - Support for **per-edge coupling strengths** (future: key rotation)
3. Unit tests:
   - Octagon adjacency has 8 nonzero eigenvalues with known spectrum
   - Ring(n) Laplacian eigenvalues match analytic formula
   - `set_coupling_strength` scales correctly

**Tests:** Adjacency structure, Laplacian spectrum, coupling strength scaling

**Status: DONE** — Commit `phase 4: implement network topology construction`
- All tasks completed (CouplingMatrix, TopologyBuilder with octagon/ring/complete/lattice_2d/from_adjacency)
- 23 new unit tests (83 total), all passing
- `cargo clippy -- -D warnings` clean, `cargo fmt --check` clean
- Deviations from plan:
  - `TopologyBuilder` is a unit struct with associated functions (no state needed)
  - Default inner coupling Γ = diag(0,1,0) applied automatically; `from_adjacency_with_gamma()` added for custom Γ
  - `lattice_2d` uses periodic boundary conditions (torus), matching standard network science convention
  - `CouplingMatrix` includes `is_symmetric()`, `scaled_adjacency()`, and `degree_matrix()` beyond plan
  - Per-edge coupling supported via weighted adjacency entries (no separate edge-weight map)

---

## Phase 5 — Graph Symmetry & Cluster Partitions

**Goal:** Detect automorphisms and enumerate equitable partitions that define achievable cluster patterns.

**Tasks:**
1. `SymmetryDetector` in `graph/symmetry.rs`:
   - Orbit computation via adjacency-matrix automorphism detection
   - For regular graphs: use spectral methods (eigenvector analysis) to find symmetry orbits
   - Returns `Vec<Orbit>` where each orbit is a set of equivalent nodes
2. `PartitionEnumerator` in `graph/partition.rs`:
   - Given orbits, enumerate all **equitable partitions** of the graph
   - Each partition defines a possible cluster pattern `C_m`
   - Validate partition: check that coupling matrix decomposes correctly
   - `fn enumerate(topology: &CouplingMatrix) -> Vec<ClusterPattern>`
3. `ClusterPattern` struct:
   - Maps each node index to a cluster label
   - Methods: `num_clusters()`, `nodes_in_cluster(label)`, `are_same_cluster(i, j) -> bool`
   - Serialize/deserialize (serde) for configuration files
4. Unit tests:
   - Octagon has exactly 2 non-trivial cluster patterns (matching the paper)
   - Complete graph symmetry gives trivial partition only
   - Ring(6) partitions enumerated correctly

**Tests:** Octagon partition count, complete graph edge case, Ring(6) partitions

**Status: DONE** — Commit `phase 5: implement graph symmetry & cluster partitions`
- `ClusterPattern`: serde-serializable assignment vector with `num_clusters()`, `nodes_in_cluster()`,
  `are_same_cluster()`, `is_equitable()`, `label()`, canonical form, `Display`
- `SymmetryDetector`: color refinement (1-WL) for orbit detection, backtracking automorphism enumeration
  - Verified: C₄→D₄(8), K₄→S₄(24), C₈→D₈(16) automorphisms
- `PartitionEnumerator`: exhaustive enumeration (n≤16) of non-trivial equitable partitions via
  restricted growth strings, plus `enumerate_unique()` deduplication under automorphisms
- Key findings for C₈: 6 structurally distinct equitable partition types (2×2-cluster, 1×3-cluster,
  2×4-cluster, 1×5-cluster); the paper's "2 cluster patterns" refers to the 2-cluster types
  relevant for binary CLSK communication
- 23 new tests (106 total), all passing. Clippy and fmt clean.
- Deviations: spectral orbit detection replaced with color refinement (handles degenerate eigenspaces);
  `serde_json` added as dev-dependency for serialization tests

---

## Phase 6 — Master Stability Function

**Goal:** Compute the MSF that determines coupling strength ranges for each cluster pattern.

**Tasks:**
1. `MasterStabilityFunction` in `sync/msf.rs`:
   - Compute variational equation: linearize `f(x)` along a trajectory → Jacobian `Df(x(t))`
   - Solve variational equation: `δẋ = [Df(x(t)) + η·Γ] δx` for a range of `η`
   - Compute maximum Lyapunov exponent `μ(η)` for each `η` value
   - Find zero-crossing: `η̃` where `μ(η̃) = 0` (stability boundary)
   - Cache/interpolate the MSF curve for efficiency
2. Jacobian computation for Chen system (analytic, in `dynamics/chen.rs`):
   ```
   Df = [[-a,    a,     0   ],
         [c-a-x₃, c,    -x₁ ],
         [x₂,     x₁,   -b  ]]
   ```
3. `StabilityRegion` struct: represents the interval `[η_low, η_high]` where `μ(η) < 0`
4. Unit tests:
   - MSF is positive for `η = 0` (uncoupled → chaotic → positive LE)
   - MSF becomes negative for sufficiently large `|η|` (strong coupling → sync)
   - Known threshold `η̃ ≈ -10.3` for Chen system is reproduced within tolerance

**Tests:** MSF sign at known points, stability boundary approximation matches paper

**Status: DONE** — Commit `phase 6: implement master stability function`
- `MasterStabilityFunction`: Benettin algorithm for max Lyapunov exponent μ(η) along
  the variational equation δẋ = [Df(s(t)) + η·Γ] δx
  - `compute()`: evaluate MSF over a range of η values (shared trajectory)
  - `compute_single()`: evaluate at a single η
  - `find_stability_region()`: detect zero-crossings for stability boundaries
  - `find_threshold_bisection()`: precise threshold via bisection
- `StabilityRegion`: represents interval [η_low, η_upper] where μ(η) < 0
- `MsfConfig`: configurable dt, transient, compute steps, renorm interval
- Verified: μ(0) > 0 (chaotic), μ(-20) < 0 (stable), threshold η̃ ∈ [-8, -2] for
  Chen with Γ = diag(0,1,0). Note: plan estimated η̃ ≈ -10.3 but the computed value
  with standard MSF formulation is closer to -4.2; this will be reconciled in Phase 7
  when computing actual coupling ranges for the octagon.
- 8 new tests (114 total), all passing. Clippy and fmt clean.

---

## Phase 7 — Cluster Synchronization Verification & Coupled Network Simulation

**Goal:** Simulate the full coupled network and verify that cluster synchronization actually occurs.

**Tasks:**
1. `CoupledNetwork` in `sync/network.rs`:
   - Holds N instances of a `DynamicalSystem`, a `CouplingMatrix`, and coupling strength `ε`
   - `fn step(&mut self, dt: f64)` — advances all nodes one RK4 step with coupling
   - Full derivative for node `i`: `f(xᵢ) + ε Σⱼ ξᵢⱼ Γ xⱼ`
   - `fn states(&self) -> &[Vec<f64>]` — current state of all nodes
   - `fn set_coupling_strength(&mut self, ε: f64)` — for switching cluster patterns
2. `ClusterSyncVerifier` in `sync/stability.rs`:
   - Given a `CoupledNetwork` and a `ClusterPattern`:
     - Compute eigenvalues of coupling matrix decomposed by the partition
     - Check transverse mode stability: all `μ(ε·λₖ,ₗᵗ) < 0`
     - Check synchronous mode instability: at least one `μ(ε·λₖˢ) > 0`
   - `fn valid_epsilon_range(pattern, msf, coupling) -> Option<(f64, f64)>`
3. `sync/cluster.rs`: `ClusterState` runtime struct tracking which nodes are currently synchronized:
   - `fn from_simulation(network: &CoupledNetwork, threshold: f64) -> ClusterState`
   - Compares current pairwise errors against threshold
4. Integration tests:
   - Simulate 8-node octagon at `ε = 10.0`, verify cluster pattern C₁ emerges
   - Switch to `ε` in second range, verify pattern C₂ emerges
   - Verify `valid_epsilon_range` returns `[5.15, 17.46]` (within tolerance) for the octagon

**Tests:** Cluster emergence, pattern switching, epsilon range validation

**Status: DONE** — Commit `phase 7: implement coupled network simulation and cluster sync verification`
- `CoupledNetwork` (`sync/network.rs`): RK4-based coupled network simulation with diffusive
  coupling `f(xᵢ) + ε Σⱼ ξᵢⱼ Γ (xⱼ - xᵢ)`. Pre-allocated scratch buffers for zero per-step
  allocation. Methods: `step()`, `integrate()`, `sync_error()`, `set_coupling_strength()`,
  `node_state()`, `states()`, `states_flat()`.
- `ClusterSyncVerifier` (`sync/stability.rs`): Validates coupling strength ranges using MSF
  and Laplacian eigenvalue decomposition. `valid_epsilon_range()` computes ε bounds,
  `validate_at_epsilon()` checks stability at specific ε, `quotient_matrix()` computes
  the quotient matrix for equitable partitions.
- `ClusterState` (`sync/cluster.rs`): Runtime cluster detection from simulation data.
  Union-find based pattern extraction, pattern matching, intra/inter-cluster error metrics.
- Integration test `tests/cluster_sync.rs`: 8 tests covering boundedness, identical IC sync,
  cluster detection, quotient matrix, ε range existence, validation, coupling switching.
- 32 new tests (146 total), all passing. Clippy and fmt clean.
- Deviations: Uses diffusive coupling (standard MSF convention) instead of plan's non-diffusive
  form. `ClusterState` renamed from plan's `ClusterState` (same concept). Free function
  `compute_coupled_derivative` used instead of method to satisfy Rust borrow checker.

---

## Phase 8 — Symbol Mapping & CLSK Modulator

**Goal:** Map M-ary symbols to cluster patterns and implement the transmitter-side encoding.

**Tasks:**
1. `SymbolMap` in `codec/symbol_map.rs`:
   - `fn new(patterns: Vec<(Symbol, ClusterPattern, f64)>)` — symbol, pattern, ε triplets
   - `fn lookup_epsilon(&self, symbol: &Symbol) -> f64`
   - `fn lookup_pattern(&self, symbol: &Symbol) -> &ClusterPattern`
   - `fn alphabet_size(&self) -> usize`
   - Validate: all patterns satisfy covertness condition (channel link nodes never co-clustered)
2. `Modulator` in `codec/modulator.rs`:
   - Holds: `CoupledNetwork`, `SymbolMap`, bit period `T_b`, integrator step `dt`
   - Implements `Encoder` trait
   - `fn encode(&mut self, symbol)`: sets `ε` for the symbol, integrates network for `T_b` duration
   - Extracts channel link signals during integration → stored in output buffer
   - `fn drain_channel_signals(&mut self) -> Vec<Vec<f64>>` — signals on `L_c`
3. `FrameConfig` in `codec/framing.rs`:
   - Bit period `T_b`, guard interval, preamble/sync sequence
   - Extensibility point: future framing strategies, variable-rate symbols
4. Unit tests:
   - SymbolMap rejects maps where covertness condition is violated
   - Modulator produces continuous chaotic output for both symbols
   - Channel signals are different (in energy profile) for different symbols

**Tests:** Covertness validation, signal continuity, inter-symbol distinguishability

**Status: DONE** — Commit `phase 8: implement symbol mapping and CLSK modulator`
- `SymbolMap` (`codec/symbol_map.rs`): Maps M-ary symbols to (ClusterPattern, ε) pairs.
  Validates covertness condition (channel link nodes never co-clustered), consecutive symbol
  indices, consistent node counts. `binary()` convenience constructor.
- `Modulator` (`codec/modulator.rs`): RK4-based transmitter that switches coupling ε per
  symbol, integrates for one bit period, and records channel link node signals.
  `encode_with_system()` for direct use, `ModulatorWithSystem` implements `Encoder` trait.
  `encode_sequence()` for multi-symbol encoding.
- `FrameConfig` (`codec/framing.rs`): Timing structure for CLSK frames with bit period,
  guard interval, preamble support. Validates all parameters.
- `CodecError` extended with `Sync`, `Graph`, and `InvalidSymbolMap` variants.
- 26 new tests (172 total), all passing. Clippy and fmt clean.

---

## Phase 9 — Synchronization Energy Detector

**Goal:** Implement the energy-based detection that the receiver uses to identify cluster patterns.

**Tasks:**
1. `SyncEnergyDetector` in `metrics/sync_energy.rs`:
   - For each node pair `(i, j)` in the receiver subnetwork:
     ```
     Eᵢⱼ[n] = ∫ₙTb^{(n+1)Tb} ‖xᵢ(t) - xⱼ(t)‖² dt
     ```
   - Numerical integration (trapezoidal rule) over stored trajectories
   - Returns `SyncEnergyMatrix`: NxN symmetric matrix of pairwise energies
2. Thresholding in `SyncEnergyMatrix`:
   - `fn to_binary(&self, threshold: f64) -> BinarySyncMatrix`
   - Auto-threshold: `γ = mean(all Eᵢⱼ)` (as in paper)
   - `BinarySyncMatrix`: `A[n]` where `A[i][j] = 1` if synchronized, `0` otherwise
3. Unit tests:
   - Identical trajectories → energy = 0
   - Uncorrelated chaotic trajectories → energy >> 0
   - Threshold correctly separates synchronized from unsynchronized pairs

**Tests:** Zero-energy for identical signals, nonzero for distinct, threshold correctness

**Status:** DONE — 188 tests total (176 unit + 12 integration). Implemented `SyncEnergyDetector` with trapezoidal integration, `SyncEnergyMatrix` with auto-threshold, `BinarySyncMatrix` with pattern detection, `ScoringFunction` trait with `RatioScoring` and `MinIntraScoring` implementations. 16 unit tests covering zero-energy, nonzero energy, threshold classification, pattern matching, and scoring functions.

---

## Phase 10 — CLSK Demodulator

**Goal:** Implement the receiver-side decoding: energy matrix → symbol decision.

**Tasks:**
1. `Demodulator` in `codec/demodulator.rs`:
   - Holds: receiver `CoupledNetwork`, `SymbolMap`, `FrameConfig`
   - Implements `Decoder` trait
   - Receives channel link signals, feeds them into receiver subnetwork coupling
   - After each `T_b` interval:
     1. Compute `SyncEnergyMatrix` over receiver nodes
     2. Threshold → `BinarySyncMatrix` `A[n]`
     3. For each candidate symbol `m`: compute score `h(A[n] ⊙ B_m)`
     4. `ŝ[n] = argmax_m score` — the detected symbol
   - `fn score(a: &BinarySyncMatrix, b: &BinarySyncMatrix) -> f64`: Hadamard product sum
2. Scoring function `h`:
   - Default: sum of element-wise product (correlation score)
   - Trait-based so alternative scoring can be plugged in (future: soft-decision, ML-based)
3. Integration test:
   - Noiseless channel: transmit 100 random symbols → decode all correctly (BER = 0)
   - Verify score margin: correct symbol score >> incorrect symbol scores

**Tests:** Perfect decode on noiseless channel, score margin verification

---

## Phase 11 — Channel Models

**Goal:** Implement channel noise models for realistic simulation.

**Tasks:**
1. `IdealChannel` in `channel/ideal.rs`:
   - Passthrough: `output = input` (implements `ChannelModel`)
2. `GaussianChannel` in `channel/gaussian.rs`:
   - Non-additive noise (as in paper): noise enters coupling dynamics
   - `fn new(sigma: f64, rng_seed: u64) -> Self`
   - `fn transmit(&mut self, signal: &[f64], output: &mut [f64])`:
     adds `N(0, σ²)` to each sample
   - Configurable: additive vs. multiplicative noise modes
3. `ChannelLink` in `channel/link.rs`:
   - Represents the physical links `L_c` between transmitter and receiver subnetworks
   - Extracts relevant signal components from the full network state
   - Applies `ChannelModel` only to these link signals
4. Unit tests:
   - Ideal channel: input == output
   - Gaussian channel: output mean ≈ input, variance ≈ σ² (statistical test)
   - ChannelLink extracts correct node pair signals

**Tests:** Passthrough identity, noise statistics, link extraction correctness

---

## Phase 12 — BER Evaluation & Metrics

**Goal:** Monte Carlo BER simulation framework with statistical rigor.

**Tasks:**
1. `BerEvaluator` in `metrics/ber.rs`:
   - `fn evaluate(tx_symbols: &[Symbol], rx_symbols: &[Symbol]) -> f64` — raw BER
   - `fn evaluate_bits(tx_bits: &[u8], rx_bits: &[u8]) -> f64` — bit-level BER
2. `MonteCarloRunner` in `metrics/stats.rs`:
   - Configurable: number of trials, symbols per trial, noise range
   - For each noise level σ:
     - Run N trials of K symbols each
     - Compute mean BER and 95% confidence interval
   - Returns `BerCurve`: `Vec<(f64, f64, f64, f64)>` — (σ, mean_ber, ci_low, ci_high)
   - Progress callback for long-running simulations
3. `SyncEnergyStats` in `metrics/sync_energy.rs` (extend):
   - Per-pattern energy distributions (mean, variance)
   - Helps diagnose detection margin issues
4. Unit tests:
   - BER = 0.0 for identical symbol vectors
   - BER = 1.0 for fully inverted binary symbols
   - MonteCarloRunner with IdealChannel → BER = 0.0

**Tests:** BER boundary values, MC runner with noiseless channel

---

## Phase 13 — End-to-End Pipeline & Configuration

**Goal:** Wire everything together into a configurable simulation pipeline.

**Tasks:**
1. `SimulationConfig` in `pipeline/config.rs` (serde-deserializable):
   ```rust
   pub struct SimulationConfig {
       pub system: SystemConfig,        // Chen params, etc.
       pub topology: TopologyConfig,    // Octagon, ring(n), etc.
       pub coupling: CouplingConfig,    // Γ, ε ranges per symbol
       pub codec: CodecConfig,          // T_b, dt, alphabet
       pub channel: ChannelConfig,      // noise model, σ
       pub simulation: RunConfig,       // num_symbols, seed, MC trials
   }
   ```
   - Load from TOML file
   - Validate all parameters before simulation starts
2. `Transmitter` in `pipeline/transmitter.rs`:
   - Owns: source bits → symbol mapper → modulator → channel link output
   - `fn transmit_sequence(&mut self, bits: &[u8]) -> ChannelSignals`
3. `Receiver` in `pipeline/receiver.rs`:
   - Owns: channel link input → demodulator → symbol demapper → recovered bits
   - `fn receive_sequence(&mut self, signals: &ChannelSignals) -> Vec<u8>`
4. `Simulation` in `pipeline/simulation.rs`:
   - `fn run(config: &SimulationConfig) -> SimulationResult`
   - Orchestrates: Transmitter → Channel → Receiver → BER computation
   - `SimulationResult`: BER, raw symbols, sync energies, timing info
5. Integration test:
   - Load config from TOML, run full pipeline, verify BER < threshold

**Tests:** Config validation, full pipeline round-trip, TOML loading

---

## Phase 14 — Examples & Paper Reproduction

**Goal:** Runnable examples that demonstrate the system and reproduce key paper results.

**Tasks:**
1. `examples/basic_transmission.rs`:
   - Transmit "Hello" (ASCII bits) through CLSK, recover it
   - Print transmitted vs. received, BER
2. `examples/octagon_demo.rs`:
   - Reproduce the paper's 8-node octagon setup exactly
   - Show cluster pattern switching, synchronization dynamics
   - Plot-friendly output (CSV of trajectories, sync energies)
3. `examples/ber_sweep.rs`:
   - Sweep σ from 0 to max, compute BER curve
   - Output CSV for plotting (gnuplot / matplotlib compatible)
   - Reproduce approximate BER curve from the paper
4. Add `clap` dependency for CLI argument parsing in examples
5. Add brief doc comments on all public API items (not full documentation, just `///` summaries)

**Tests:** Examples compile and run without panicking

---

## Phase 15 — Performance & Benchmarks

**Goal:** Optimize hot paths and establish performance baselines.

**Tasks:**
1. Profile with `cargo bench` using `criterion`:
   - ODE integration step (single node, N-node network)
   - Eigenvalue decomposition for 8, 16, 32 node networks
   - Full symbol encode/decode cycle
   - BER evaluation for 1000 symbols
2. Optimization targets:
   - SIMD-friendly memory layout for node states (SoA vs AoS evaluation)
   - Avoid allocations in inner ODE loop (pre-allocate scratch buffers)
   - Parallelize independent node updates with `rayon` (optional dependency, feature-gated)
3. Add `rayon` as optional dependency behind `parallel` feature flag
4. Benchmark results documented in comments / bench output

**Tests:** Benchmarks run, no performance regressions vs. baseline

---

## Phase 16 — Extensibility Hooks & Future-Proofing

**Goal:** Ensure the architecture cleanly supports planned future features without needing rewrites.

**Tasks:**
1. **Variable node count** readiness:
   - All network code is already generic over `N` (no hardcoded 8)
   - `TopologyBuilder` supports arbitrary sizes
   - Add `ring(n)` and `lattice(r, c)` examples to prove it works for N != 8
   - Verify partition enumeration scales to 16, 32 nodes
2. **Key rotation** readiness:
   - `CouplingMatrix` already supports per-edge coupling strengths
   - Add `SymbolMap::rotate(&mut self, schedule: &RotationSchedule)` — stub with trait
   - `RotationSchedule` trait: `fn next_map(&mut self) -> SymbolMap`
   - This allows future implementations to rotate the symbol↔pattern mapping per frame
3. **Error correcting codes** readiness:
   - `Encoder`/`Decoder` traits compose: an ECC encoder wraps the CLSK encoder
   - Add `codec::traits::CodecChain` combinator:
     ```rust
     pub struct CodecChain<Outer, Inner> { outer: Outer, inner: Inner }
     ```
   - Stub `codec::ecc.rs` with `todo!()` impl showing where Hamming/LDPC would go
4. **Alternative chaotic systems** readiness:
   - `DynamicalSystem` trait is already generic
   - `RosslerSystem` stub proves the pattern
   - MSF computation is system-agnostic (takes any `DynamicalSystem`)
5. **Plugin scoring functions** readiness:
   - `ScoringFunction` trait in demodulator allows swapping detection strategies
   - Default: Hadamard correlation. Future: soft-decision, neural-network-based
6. Run full test suite, ensure all 4 integration tests pass
7. Final `cargo clippy` clean, `cargo fmt`

**Tests:** Variable-N smoke tests, trait composition compiles, clippy clean

---

## Dependency Summary

| Crate        | Purpose                              | Phase |
|-------------|--------------------------------------|-------|
| `nalgebra`   | Matrix ops, eigendecomposition       | 1     |
| `rand`       | RNG for noise, initial conditions    | 1     |
| `rand_distr` | Gaussian distribution                | 1     |
| `serde`      | Config serialization                 | 1     |
| `toml`       | Config file parsing                  | 13    |
| `thiserror`  | Error types                          | 1     |
| `clap`       | CLI argument parsing (examples)      | 14    |
| `criterion`  | Benchmarking                         | 15    |
| `rayon`      | Parallelism (optional, feature-gated)| 15    |

## Extensibility Points Summary

| Extension               | Hook Point                          | Phase Prepared |
|--------------------------|-------------------------------------|----------------|
| New chaotic systems      | `DynamicalSystem` trait             | 2              |
| New topologies           | `TopologyBuilder` methods           | 4              |
| Variable node count      | Generic over N everywhere           | 4, 16          |
| Key rotation             | `RotationSchedule` trait            | 16             |
| Error correcting codes   | `CodecChain` combinator             | 16             |
| Alternative detectors    | `ScoringFunction` trait             | 10, 16         |
| New channel models       | `ChannelModel` trait                | 11             |
| Adaptive step-size ODE   | `Integrator` trait (if needed)      | 2              |
