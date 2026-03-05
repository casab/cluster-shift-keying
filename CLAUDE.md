# CLAUDE.md — Cluster Shift Keying (Rust)

## Project Overview

Rust implementation of Cluster Shift Keying (CLSK), a chaos-based communication scheme that encodes information into spatio-temporal synchronization patterns of coupled chaotic networks. Based on the paper by Sarı & Günel (Physica Scripta, 2024). See `RESEARCH.md` for the full research summary and `PLAN.md` for the 16-phase implementation plan.

## Build & Test

```bash
cargo build                    # Build the library
cargo test                     # Run all tests (unit + integration)
cargo test --lib               # Unit tests only
cargo test --test <name>       # Single integration test (chen_attractor, cluster_sync, codec_roundtrip, pipeline_e2e)
cargo clippy -- -D warnings    # Lint — must pass with zero warnings
cargo fmt --check              # Format check — must pass
cargo bench                    # Run benchmarks (requires criterion, Phase 15+)
```

## Mandatory Rules

### Type Safety — Zero Tolerance for Runtime Panics

- **No `.unwrap()` anywhere in library code (`src/`).** Use `?` with proper error types or `expect()` with a message only in tests/examples.
- **No `.expect()` in library code** unless the invariant is provably unreachable and documented with a comment explaining why.
- All fallible operations return `Result<T, E>` with `thiserror`-derived error types.
- Use newtypes and enums to make invalid states unrepresentable (e.g., `NodeIndex(usize)`, `CouplingStrength(f64)` with validated constructors).
- Prefer `TryFrom`/`TryInto` over unchecked conversions.
- No `unsafe` code without a `// SAFETY:` comment and a compelling reason.
- All public API functions must have typed errors — never `Box<dyn Error>` in public signatures.

### Error Handling Hierarchy

```
src/ library code   → Result<T, ModuleError> with thiserror — NEVER unwrap/expect
tests/              → .expect("reason") is acceptable for test clarity
examples/           → .expect("reason") is acceptable, anyhow::Result for main
benches/            → .unwrap() acceptable only in setup code
```

### Testing — Every Phase Ships with Tests

- **Unit tests**: Live in the same file as the code (`#[cfg(test)] mod tests`).
- **Integration tests**: Live in `tests/` directory, one per major subsystem.
- After completing any phase, run `cargo test` and `cargo clippy -- -D warnings`. Both must pass before committing.
- Tests must be deterministic: use seeded RNG (`rand::SeedableRng`) for anything stochastic.
- Numerical tests use tolerance-based assertions (e.g., `assert!((a - b).abs() < 1e-6)`), never exact float equality.
- Every public function must have at least one test exercising its happy path and one test for each error variant it can return.

### Code Style

- Edition 2021, stable Rust only — no nightly features.
- `cargo fmt` enforced — no custom formatting overrides.
- `cargo clippy -- -D warnings` enforced — treat all warnings as errors.
- Modules organized by domain: `dynamics/`, `linalg/`, `graph/`, `sync/`, `codec/`, `channel/`, `metrics/`, `pipeline/`, `utils/`.
- Prefer returning iterators or slices over allocating `Vec` when the caller doesn't need ownership.
- Pre-allocate scratch buffers for hot loops (ODE integration, energy computation). Never allocate inside a per-timestep function.

### Naming Conventions

- Types: `PascalCase` — `ChenSystem`, `CoupledNetwork`, `ClusterPattern`
- Functions/methods: `snake_case` — `set_coupling_strength`, `compute_energy`
- Constants: `SCREAMING_SNAKE_CASE` — `DEFAULT_CHEN_A`, `DEFAULT_DT`
- Mathematical parameters keep paper notation in comments: `ε` (epsilon), `Ξ` (xi), `Γ` (gamma), `η` (eta), `μ` (mu)
- Type aliases for clarity: `type NodeIndex = usize;` etc. — but prefer newtypes where validation is needed.

### Architecture Constraints

- All core abstractions are trait-based for extensibility:
  - `DynamicalSystem` — chaotic oscillator implementations
  - `Encoder` / `Decoder` — codec implementations (composable via `CodecChain`)
  - `ChannelModel` — noise models
  - `ScoringFunction` — detection strategies
- No hardcoded node counts — all network code is generic over N.
- Coupling matrices support per-edge strengths (future key rotation).
- Configuration is serde-deserializable from TOML.
- Feature flags: `parallel` (rayon), keep the default build dependency-light.

### Numerical Considerations

- Default ODE step size: `dt = 0.001` for Chen system (verified stable in Phase 2).
- Chen system parameters: `a = 35.0`, `b = 8/3`, `c = 28.0`.
- Inner coupling matrix: `Γ = diag(0, 1, 0)` (coupling through second state variable).
- Master stability threshold for Chen: `η̃ ≈ -4.2` (with Γ=diag(0,1,0), standard MSF formulation).
- Octagon coupling range: `ε ∈ [5.15, 17.46]`.
- All floating-point comparisons use epsilon-based tolerance, never `==`.

### Git Workflow

- One commit per completed phase (or logical sub-unit within a phase).
- Commit message format: `phase N: short description` (e.g., `phase 2: implement Chen system and RK4 integrator`).
- All commits must pass `cargo test` and `cargo clippy -- -D warnings`.
- Do not commit generated files, build artifacts, or IDE configs.

### Dependencies

| Crate | Purpose | Phase |
|-------|---------|-------|
| `nalgebra` | Matrix ops, eigendecomposition | 1 |
| `rand` + `rand_distr` | RNG, distributions | 1 |
| `serde` + `serde_derive` | Serialization | 1 |
| `thiserror` | Error types | 1 |
| `toml` | Config parsing | 13 |
| `clap` | CLI for examples | 14 |
| `criterion` | Benchmarks | 15 |
| `rayon` (optional) | Parallelism | 15 |

### File Layout

```
src/
├── lib.rs              # Re-exports, top-level docs
├── dynamics/           # Chaotic systems, ODE solvers
├── linalg/             # Matrix wrappers, eigendecomposition
├── graph/              # Topology, coupling, symmetry, partitions
├── sync/               # MSF, cluster sync, coupled network sim
├── codec/              # Modulator, demodulator, symbol mapping
├── channel/            # Channel models (ideal, Gaussian)
├── metrics/            # BER, sync energy, Monte Carlo
├── pipeline/           # End-to-end simulation config & orchestration
└── utils/              # Seeded RNG, shared helpers
```
