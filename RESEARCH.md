# Cluster Shift Keying: Research Summary

## Paper Details

- **Title:** Cluster Shift Keying: Covert Transmission of Information via Cluster Synchronization in Chaotic Networks
- **Authors:** Zekeriya Sarı, Serkan Günel
- **Published:** Physica Scripta, Volume 99, 035204 (2024)
- **Preprint:** arXiv:2312.04593 (December 5, 2023)
- **Categories:** Information Theory (cs.IT), Chaotic Dynamics (nlin.CD)
- **License:** CC BY-NC-ND 4.0

## Overview

The paper introduces **Cluster Shift Keying (CLSK)**, a novel chaos-based communication scheme that encodes information into the spatio-temporal synchronization patterns of a network of coupled chaotic dynamical systems, rather than directly modulating chaotic signals. This approach addresses the fundamental weakness of prior chaotic communication methods — their direct modulation of information onto chaotic waveforms — which makes them detectable and vulnerable to eavesdropping.

## Motivation and Background

### Prior Chaotic Communication Methods

Traditional chaos-based communication schemes fall into three categories:
1. **Chaotic masking** — information is hidden within a chaotic carrier signal
2. **Chaos shift keying (CSK)** — switching between different chaotic attractors to encode bits
3. **Bifurcation parameter modulation** — modulating parameters that alter the chaotic dynamics

All of these methods share a common weakness: they **directly alter the chaotic signal** based on the transmitted information, which can be detected by statistical analysis of the transmitted waveform.

### Cluster Synchronization

In networks of coupled dynamical systems:
- **Full synchronization**: all nodes converge to the same trajectory
- **Cluster synchronization**: nodes synchronize within groups (clusters) but remain unsynchronized across groups

The specific cluster pattern that emerges depends on the network topology and coupling strengths. Crucially, for a given network, **different coupling parameter values can produce different cluster patterns** — this is the key insight exploited by CLSK.

## Core Concept

CLSK encodes symbols by switching between different cluster synchronization patterns in a chaotic network. Each symbol `m` (from an M-ary alphabet) maps to a specific cluster pattern `C_m`, which is selected by adjusting coupling parameters in the transmitter subnetwork.

### Key Design Principles

1. **Network partitioning**: The network is split into:
   - **Transmitter subnetwork** (`N_T`) — contains controllable coupling parameters
   - **Receiver subnetwork** (`N_R`) — detects cluster patterns for decoding
   - **Channel links** (`L_c`) — connect transmitter to receiver

2. **Covertness condition**: The nodes at each end of every channel link must **never** be in the same cluster for any symbol. This ensures all transmitted signals remain continuous and chaotic regardless of the symbol being sent.

3. **Topology-based security**: Without knowledge of the full network topology, an eavesdropper cannot decipher the information from the channel links alone.

## Mathematical Framework

### Network Dynamics

The coupled network is governed by:

```
ẋᵢ = f(xᵢ) + ε Σⱼ ξᵢⱼ Γ xⱼ
```

where:
- `xᵢ` is the state of node `i`
- `f(·)` is the uncoupled chaotic dynamics
- `ε > 0` is the coupling strength
- `Ξ = [ξᵢⱼ]` is the coupling topology matrix
- `Γ` is the inner coupling matrix

### Cluster Synchronization Conditions

Using the **master stability function** framework:
- Stability function: `μ = g(η)` where `η = ε·λ` for eigenvalues `λ` of the coupling matrix
- All transverse modes must satisfy: `μₖ,ₗᵗ = g(ε·λₖ,ₗᵗ) ≤ 0`
- At least one nontrivial synchronous mode must be unstable (to prevent full synchronization)

### Coupling Strength Range

For a valid cluster pattern, the coupling strength must lie in:

```
ε ∈ [η̃ / λ_min, η̃ / λ₂ˢ]
```

where `η̃ < 0` is the master stability threshold.

## Illustrative Example: Chen System on Octagon Network

### Chaotic System

The paper uses the **Chen system** as the node dynamics:

```
ẋ₁ = a(x₂ - x₁)
ẋ₂ = x₁(c - a - x₃) + c·x₂
ẋ₃ = x₁·x₂ - b·x₃
```

Parameters: `a = 35`, `b = 8/3`, `c = 28`

Inner coupling matrix: `Γ = diag(0, 1, 0)` (coupling through the second state variable only)

### Network Topology

- **N = 8 nodes** arranged in an octagon topology
- Two distinct cluster patterns achievable by varying `ε`
- Guaranteed clustering range: `ε ∈ [5.15, 17.46]`
- Master stability threshold: `η̄ ≈ -10.3`

## Detection Algorithm

### Synchronization Energy Computation

For each pair of nodes `(i, j)`, the synchronization error energy over bit period `T_b` is:

```
Eᵢⱼ[n] = ∫₍ₙTb₎^₍₍ₙ₊₁₎Tb₎ ‖eᵢⱼ(t)‖² dt
```

### Decision Rule

A binary synchronization matrix `A[n]` is constructed by thresholding energies. Detection is then:

```
ŝ[n] = argmax_m h(A[n] ⊙ B_m)
```

where:
- `A[n]` is the computed synchronization matrix for the n-th symbol
- `B_m` is the reference pattern matrix for symbol `m`
- `⊙` denotes Hadamard (element-wise) product
- `h(·)` is a scoring function

## Performance

### Bit Error Rate (BER) Results

- Performance evaluated under **Gaussian noise** with varying strength `σ`
- The noise is **non-additive** (enters through the coupling dynamics), so direct comparison with conventional SNR-based methods is nuanced
- The system demonstrates robustness to channel noise at moderate levels
- Outperforms some well-known chaotic communication schemes for certain noise thresholds

### Advantages

1. **Covertness**: Transmitted signals are always chaotic — no statistical signature of the information
2. **Security**: Requires knowledge of full network topology to decode
3. **Distributed encoding**: Information is spread across the network, not concentrated in a single link
4. **No drive-response**: Transmitter and receiver are not in classical drive-response mode

### Limitations

1. **Symbol capacity**: Limited by the number of achievable cluster patterns, which depends on the network's topological symmetries
2. **Topology complexity**: Determining all symmetries in complex topologies is computationally involved
3. **Regular topologies preferred**: Authors use regular (symmetric) network topologies to ensure tractable analysis
4. **Synchronization transients**: Switching between cluster patterns requires time for the network to re-synchronize

## Key Figures in the Paper

| Figure | Description |
|--------|-------------|
| Fig. 1 | Two cluster patterns `C_i` and `C_j` with controlled coupling links marked |
| Fig. 2 | 8-node octagon network topologies showing two distinct cluster patterns |
| Fig. 3 | Eigenvalue distributions for different symmetries of the coupling matrix |
| Fig. 4 | Spatiotemporal network status changes with coupling strength `ε` |

## Relation to This Repository

This repository is intended to implement the CLSK scheme described in the paper. Based on the `.gitignore` configuration, the implementation will be in **Rust**. Key components to implement would include:

1. **Chaotic system integration** (Chen system ODE solver)
2. **Network topology construction** (coupling matrices for octagon and other regular graphs)
3. **Cluster synchronization engine** (eigenvalue analysis, master stability function)
4. **CLSK modulator** (symbol-to-cluster-pattern mapping, coupling parameter control)
5. **CLSK demodulator** (synchronization energy detection, decision rule)
6. **Channel simulation** (Gaussian noise injection)
7. **BER evaluation** (Monte Carlo simulation framework)

## References

- arXiv preprint: https://arxiv.org/abs/2312.04593
- Full HTML version: https://arxiv.org/html/2312.04593
- Physica Scripta publication: Phys. Scr. 99 (2024) 035204
