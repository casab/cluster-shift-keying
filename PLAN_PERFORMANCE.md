# Performance Optimization Plan

## Profiling Summary

The CLSK pipeline is: **Encode → Channel → Decode**. Encoding and decoding both call the RK4 ODE integrator in a tight loop. The critical path for a single simulation:

```
decode_sequence()              — O(S) symbols
  └─ detect_symbol()           — per symbol, M candidates
       └─ network.step()       — per candidate, T steps
            └─ compute_coupled_derivative()  — 4× per step (RK4)
                 └─ system.derivative()      — N nodes × D=3 dims
                 └─ coupling term             — N nodes × deg neighbors × D² ops
```

**Total inner ops per simulation:** `S × (M+1) × T × 4 × N × (D + deg·D²)`

For 512-node ring, M=4, 10 symbols, T=10000 steps:
- `10 × 5 × 10000 × 4 × 512 × (3 + 2·9)` ≈ **21.5 billion FLOPs**

---

## Phase 1: Rayon Feature Flag + Parallel Demodulator Candidates

**Impact: HIGH — 2-4× decode speedup**
**Files:** `Cargo.toml`, `src/codec/demodulator.rs`

The demodulator's `detect_symbol()` evaluates M candidate epsilons **sequentially** (line 173). Each candidate independently:
1. Restores saved state
2. Simulates network for T steps
3. Computes MSE

These are **fully independent** — no data dependencies between candidates. With M=4 or M=8, this is the easiest parallelism win.

**Plan:**
1. Add `rayon = { version = "1.10", optional = true }` to `Cargo.toml`
2. Add `[features] parallel = ["rayon"]` section
3. In `detect_symbol()`, when `parallel` feature is enabled:
   - Pre-allocate M `CoupledNetwork` clones (one per candidate)
   - Use `rayon::iter::ParallelIterator` to evaluate candidates in parallel
   - Each thread: clone saved state → set epsilon → simulate → compute MSE
   - Collect results, pick minimum MSE
4. After winner is found, advance the primary network with the winner (sequential, as now)

**Note:** Early stopping is incompatible with full parallelism (no shared `best_mse` across threads), but the parallel gain outweighs early stopping savings for M ≥ 4.

---

## Phase 2: Batch Node Derivatives with Rayon

**Impact: HIGH for large N — 2-8× per-step speedup (N=512)**
**Files:** `src/sync/network.rs`

`compute_coupled_derivative()` (line 365) loops over all N nodes sequentially. Each node's computation is **independent**: it reads (immutably) neighbor states and writes to its own slice of `output[offset_i..offset_i+dim]`.

**Plan:**
1. Gate behind `#[cfg(feature = "parallel")]`
2. Split the output buffer into per-node chunks using `chunks_mut(dim)`
3. Use `rayon::iter::IndexedParallelIterator` to process nodes in parallel
4. Each thread needs its own `derivative_scratch` and `coupling_scratch` (currently shared) — use thread-local buffers or allocate per-chunk
5. The `system.derivative()` trait call requires `&self` (shared ref) — verify `DynamicalSystem` is `Sync` (it should be, pure function of state)

**Key concern:** The `DynamicalSystem` trait must be `Send + Sync` for rayon. Currently it's `dyn DynamicalSystem` with no bounds. Need to add `Send + Sync` supertrait bounds (or use a generic parameter).

**Memory layout note:** Current layout is SoA-friendly: `states[i*dim + d]` = node i, dimension d. Chunks of `dim` are contiguous — good for cache lines.

---

## Phase 3: SIMD-Friendly Inner Loops

**Impact: MODERATE — 1.5-2× for coupling computation**
**Files:** `src/sync/network.rs`, `src/dynamics/chen.rs`

The innermost coupling loop (line 387-394) computes Γ(xⱼ - xᵢ) for each neighbor:
```rust
for d in 0..dim {           // dim = 3
    let mut gamma_diff = 0.0;
    for k in 0..dim {       // dim = 3
        gamma_diff += gamma[d*dim + k] * (state[offset_j+k] - state[offset_i+k]);
    }
    coupling_scratch[d] += xi_ij * gamma_diff;
}
```

For the Chen system with Γ = diag(0,1,0), most of these multiplications are by zero. The compiler can't know this at compile time.

**Plan:**
1. Add a `DiagonalGamma` fast path: when Γ is diagonal (common case), skip the D² inner loop and use D multiplications instead
2. Check if Γ is diagonal at construction time, store a flag
3. Specialized loop: `coupling_scratch[d] += xi_ij * gamma[d*dim+d] * (state[offset_j+d] - state[offset_i+d])`
4. For Γ=diag(0,1,0) specifically, only `d=1` contributes — reduce to a single multiply-add per neighbor

The RK4 combination loop (line 307-310) is a textbook SIMD candidate:
```rust
states[i] += dt/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
```
Use `#[target_feature(enable = "avx2")]` or let the compiler auto-vectorize with `-C target-cpu=native`.

---

## Phase 4: Parallel BER Sweep / Monte Carlo

**Impact: LINEAR with cores — 4-16× for parameter sweeps**
**Files:** `examples/ber_sweep.rs`, `src/pipeline/simulation.rs`

The BER sweep iterates over sigma values sequentially. Each sigma trial is **fully independent** (separate config, simulation, RNG seed).

**Plan:**
1. Use `rayon::iter::IntoParallelIterator` over sigma values
2. Each thread: build config → create simulation → run → collect result
3. Collect results, sort by sigma, print
4. Add `num_trials` parameter to `SimulationConfig` for Monte Carlo averaging
5. Parallelize both across sigma values AND across trials (2D parallelism)

---

## Phase 5: Memory & Allocation Optimization

**Impact: LOW-MODERATE — reduces GC pressure, improves cache behavior**
**Files:** `src/codec/demodulator.rs`, `src/sync/network.rs`

Current allocation patterns to optimize:
1. **`detect_symbol()` line 156**: `channel_links.to_vec()` allocates every call — store as `Vec<usize>` field instead
2. **Demodulator candidate evaluation**: if using Phase 1's parallel approach, pre-allocate M network clones at construction time rather than per-symbol
3. **RK4 scratch buffers**: already pre-allocated (good) — verify no hidden allocations in `system.derivative()` calls

---

## Phase 6: Profile-Guided Optimization (PGO)

**Impact: 10-20% across the board**

Not a code change but a build workflow:
```bash
# Collect profile
cargo build --release
./target/release/examples/multi_bit_512 --num-symbols 100
# Rebuild with PGO
RUSTFLAGS="-Cprofile-use=..." cargo build --release
```

Also: build with `-C target-cpu=native` to enable AVX2/FMA auto-vectorization on the RK4 and coupling loops.

---

## Priority Order

| Phase | Area | Speedup | Effort | Dependencies |
|-------|------|---------|--------|--------------|
| 1 | Parallel demod candidates | 2-4× decode | Medium | None |
| 2 | Parallel node derivatives | 2-8× per step | Medium | Phase 1 (rayon) |
| 3 | Diagonal Γ fast path + SIMD | 1.5-2× coupling | Low | None |
| 4 | Parallel BER sweep | Linear w/ cores | Low | Phase 1 (rayon) |
| 5 | Allocation cleanup | ~10% | Low | None |
| 6 | PGO + target-cpu=native | 10-20% | Low | None |

**Phases 1+3 together** give the best ROI: ~3-6× total speedup with moderate effort.
