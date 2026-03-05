# Naming Improvement Plan

Improve variable and function naming across the codebase for readability,
without changing any public API signatures or behavior.

---

## Phase 1: Core Math & Simulation Internals

**Scope:** `src/dynamics/`, `src/linalg/`, `src/sync/network.rs`

These modules contain the hottest loops and most cryptic names. Focus on
scratch buffers, loop variables in nested contexts, and tuple bindings.

### Changes

**`src/dynamics/integrator.rs`**
- Rename RK4 zip bindings: `(t, (s, k))` → `(tmp_val, (state_val, k_val))`
- `tmp` field → `scratch`

**`src/linalg/eigen.rs`**
- 2×2 block entries `a, b, c, d` → `top_left, top_right, bot_left, bot_right`
- `evec_data` → `eigenvector_data`

**`src/linalg/block_diag.rs`**
- `pt` → `p_transpose`

**`src/sync/network.rs`**
- `deriv_buf` → `derivative_scratch`
- `coupling_buf` → `coupling_scratch`
- `tmp` → `scratch`
- In `compute_coupled_derivative`: `gamma_diff_d` → `gamma_diff_component`
- Loop variable `d` in nested dim loops → keep `d` (standard for dimension index, always has `dim` bound nearby)

**`src/sync/msf.rs`**
- Variational RK4 buffers: add comments clarifying `k1`–`k4` are standard RK4 stage names (no rename needed — these are conventional)
- `eta_upper` / `eta_lower` → `eta_threshold_upper` / `eta_threshold_lower`

---

## Phase 2: Graph & Partition Naming

**Scope:** `src/graph/partition.rs`, `src/graph/coupling.rs`, `src/graph/symmetry.rs`, `src/graph/topology.rs`

Focus on cluster index abbreviations and vague loop variables in
partition enumeration.

### Changes

**`src/graph/partition.rs`**
- `ci`, `cj` → `cluster_i`, `cluster_j`
- `nodes_ci` → `nodes_in_cluster`
- `l` / `next_label` consolidation — remove the intermediate `let l = next_label;` pattern, use `next_label` directly
- `mapping` in `canonical()` → `label_remap`

**`src/graph/coupling.rs`**
- `deg` → `degree` in degree matrix computation
- No rename for `i`, `j` in adjacency loops (standard matrix convention)

**`src/graph/symmetry.rs`**
- `perm` → `permutation` where used as a full variable (keep `perm` in slice parameter names where it's a standard abbreviation)
- `adj` → `adjacency` in local bindings

---

## Phase 3: Codec & Channel Naming

**Scope:** `src/codec/`, `src/channel/`

Focus on signal naming consistency, `t` loop variable, and generic names
like `scores`.

### Changes

**`src/codec/demodulator.rs`**
- `t` in time-step loops → `step` (or `step_idx`)
- `mse` → `mean_sq_err` (keep concise; `mse` is universally understood but there are two separate accumulators that benefit from distinction)
- `saved_states` → `saved_network_states`
- `scores` in `score_all` → `detection_scores`

**`src/codec/modulator.rs`**
- `sig` in drain loop → `signal`

**`src/codec/symbol_map.rs`**
- `sm` in tests → `symbol_map` (tests only, no API change)

**`src/channel/gaussian.rs`**
- `inp` / `out` in transmit loop → `input_sample` / `output_sample`

**Cross-module consistency:**
- Standardize on `tx_signals` / `rx_signals` in tests and examples (already mostly consistent)

---

## Phase 4: Metrics, Pipeline & Test Cleanup

**Scope:** `src/metrics/`, `src/pipeline/`, `tests/`, `examples/`

Focus on generic names in functional pipelines, tuple destructuring,
and test helper abbreviations.

### Changes

**`src/metrics/ber.rs`**
- `(a, b)` in filter closures → `(tx_sym, rx_sym)`
- `diff` → `bit_diff`

**`src/metrics/sync_energy.rs`**
- No major renames needed (already clear)

**`src/pipeline/simulation.rs`**
- `(a, b)` in SER filter → `(tx_sym, rx_sym)`

**`tests/` and `examples/`**
- `sm` → `symbol_map`
- `fc` → `frame_config`
- `dec` → `decoder`
- `enc` → `encoder`
- These are test-only changes; no API impact.

---

## Ground Rules (all phases)

1. **No public API changes** — only rename local variables, private fields,
   and test helpers.
2. **Keep standard math conventions** — `i`, `j` for matrix indices,
   `n` for count, `dt` for time step, `epsilon` for coupling strength.
   These are domain-standard and renaming them would hurt readability
   for anyone familiar with the math.
3. **Run `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`**
   after each phase before committing.
4. **One commit per phase.**
