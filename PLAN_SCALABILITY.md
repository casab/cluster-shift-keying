# Scalability Improvement Plan

## Bottleneck Summary

| # | Bottleneck | Severity | Location | Current Limit |
|---|---|---|---|---|
| 1 | Partition enumeration is exhaustive O(B(n)) | CRITICAL | `graph/partition.rs:220` | Hard cap N≤16 |
| 2 | Dense adjacency in coupling loop O(N²·D²)/step | SEVERE | `sync/network.rs:350-379` | N>100 slow |
| 3 | Demodulator runs M full sims per symbol | SEVERE | `codec/demodulator.rs:154-182` | M×T× cost of #2 |
| 4 | Dense N×N matrix storage everywhere | MODERATE | `sync/network.rs:21`, `graph/coupling.rs:13` | N=10K → 800MB |
| 5 | `effective_coupling()` materializes N·D×N·D Kronecker | MODERATE | `graph/coupling.rs:124-127` | Never needed in hot path |
| 6 | No parallelism in candidate evaluation | MODERATE | `codec/demodulator.rs:154` | Single-threaded |
| 7 | `states()` allocates Vec<Vec<f64>> on every call | LOW | `sync/network.rs:162-169` | Unnecessary heap churn |

---

## Phase 1: Sparse Adjacency & Neighbor Lists (network hot path)

**Goal:** Replace O(N²) inner loop in `compute_coupled_derivative` with O(N·deg) using
precomputed neighbor lists, and store adjacency sparsely.

### Changes

**`src/sync/network.rs`**
- Add a `neighbors: Vec<Vec<(usize, f64)>>` field — for each node i, a list of
  `(j, ξ_ij)` pairs where ξ_ij ≠ 0.
- Build the neighbor list in `CoupledNetwork::new()` from the dense adjacency,
  skipping zero entries (already done inline at line 360, but scanning all N).
- Rewrite `compute_coupled_derivative()` to iterate `neighbors[i]` instead of
  `0..n`. This changes the inner loop from O(N) to O(deg_i).
- Keep the dense `adjacency: Vec<f64>` for now (removed in Phase 2), but stop
  using it in the hot path.

**Complexity change:**
- Before: O(N² · D²) per RK4 stage → O(4 · N² · D²) per step
- After: O(N · deg · D²) per stage → O(4 · N · deg · D²) per step
- For ring/octagon (deg=2): **N/2 speedup** (8→ trivial, 1000→ 500× faster)

**Tests:**
- All existing tests must still pass (behavior is identical).
- Add a test with N=64 ring to verify correctness at larger scale.
- Add a benchmark comparing dense vs sparse for ring(100).

---

## Phase 2: Sparse Matrix Storage & Lazy Kronecker

**Goal:** Stop storing full N×N dense matrices when the graph is sparse. Make
`effective_coupling()` lazy or remove it from critical paths.

### Changes

**`src/linalg/sparse.rs`** (new module)
- Add a CSR (Compressed Sparse Row) matrix type: `SparseMatrix { row_ptrs, col_indices, values }`.
- Implement `from_dense(Matrix, tol)` and `to_dense() -> Matrix` for interop.
- Implement sparse mat-vec multiply for use in eigendecomposition wrappers.

**`src/graph/coupling.rs`**
- Store `adjacency` as both `Matrix` (for eigendecomposition, which needs dense)
  and a sparse neighbor-list representation (for simulation).
- Make `effective_coupling()` return a lazy wrapper or mark it `#[cfg(test)]`
  since it's only used in tests and diagnostics — the hot path already avoids it
  by caching `adjacency` and `gamma` separately in `CoupledNetwork`.

**`src/sync/network.rs`**
- Drop the dense `adjacency: Vec<f64>` field entirely. Use only the neighbor list
  from Phase 1.
- Memory for N=10K drops from 800MB (dense) to ~160KB (sparse, deg=2).

**Tests:**
- Round-trip: `SparseMatrix::from_dense(m).to_dense() == m`.
- Verify `CoupledNetwork` results are identical with sparse-only storage.

---

## Phase 3: Scalable Partition Discovery

**Goal:** Remove the N≤16 hard cap. Support user-provided partitions for any N,
and add heuristic partition discovery for moderate N.

### Changes

**`src/graph/partition.rs`**
- Keep exhaustive `enumerate()` but raise the cap to N≤20 (still exponential,
  but 20-node graphs are useful for research).
- Add `ClusterPattern::from_user(assignment: Vec<usize>) -> Result<Self, GraphError>`
  that accepts any valid partition without enumeration — this is the primary
  path for N>20 networks where the user knows the desired patterns.
- Add `enumerate_by_orbit()` that uses symmetry orbits to prune the search
  space. For vertex-transitive graphs (rings, complete), this reduces the search
  from B(n) to B(n/|orbit|) by fixing one representative per orbit.

**`src/graph/heuristic_partition.rs`** (new module)
- Implement spectral partitioning: use the Fiedler vector (second eigenvector of
  the Laplacian) to bisect the graph into two clusters, then verify equitability.
- Implement recursive spectral k-way partitioning for k>2 clusters.
- These heuristics work for any N but may miss some equitable partitions. They
  complement (don't replace) exhaustive enumeration.

**`src/pipeline/config.rs`**
- Add `patterns: Option<Vec<Vec<usize>>>` to `CouplingConfig`. When provided,
  skip enumeration entirely and use the given patterns.

**Tests:**
- Verify spectral bisection finds the bipartite partition of C₈.
- Verify `from_user()` works for N=100 ring with manually specified patterns.
- Verify pipeline runs end-to-end with user-provided patterns at N=32.

---

## Phase 4: Demodulator Optimization (early stopping + parallelism)

**Goal:** Reduce demodulator cost from O(M · T · N · deg · D²) to something
practical for large networks.

### Changes

**`src/codec/demodulator.rs` — Early stopping**
- In `detect_symbol()`, track cumulative MSE during the inner loop. If the
  current candidate's partial MSE already exceeds `best_mse`, break out of the
  timestep loop early. For M=2 binary CLSK, this roughly halves the work for
  the losing candidate.
- Extract the early-stopping threshold as a configurable tolerance in
  `DemodulatorConfig`.

**`src/codec/demodulator.rs` — Parallel candidate evaluation** (feature-gated)
- Under `#[cfg(feature = "parallel")]`, use `rayon` to evaluate M candidates in
  parallel. Each candidate needs its own `CoupledNetwork` clone (already saved/
  restored per candidate — just clone once instead of save/restore).
- Pre-allocate M network clones at `Demodulator::new()` time to avoid
  per-symbol allocation.

**`src/codec/demodulator.rs` — State checkpoint optimization**
- Replace `states_flat().to_vec()` (heap alloc) with a pre-allocated
  `saved_states: Vec<f64>` field on `Demodulator`. Copy into it via
  `copy_from_slice` instead of allocating.

**Tests:**
- Verify early stopping produces identical detection results.
- Verify parallel detection matches sequential detection (deterministic with
  same seed).
- Benchmark: compare sequential vs parallel for M=4, N=32.

---

## Ground Rules (all phases)

1. All existing tests must pass after each phase.
2. `cargo clippy -- -D warnings` and `cargo fmt --check` must pass.
3. No public API removals — only additions. Existing constructors keep working.
4. New features behind `#[cfg(feature = "parallel")]` where they add dependencies.
5. One commit per phase.
6. Benchmark before/after for phases 1 and 4 to verify speedup.
