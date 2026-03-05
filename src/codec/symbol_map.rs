use super::error::CodecError;
use crate::graph::ClusterPattern;

/// A symbol in the CLSK alphabet, represented as a usize index.
pub type Symbol = usize;

/// Entry in the symbol map: a symbol mapped to a cluster pattern and coupling strength.
#[derive(Debug, Clone)]
pub struct SymbolEntry {
    /// The symbol index.
    pub symbol: Symbol,
    /// The cluster pattern for this symbol.
    pub pattern: ClusterPattern,
    /// The coupling strength ε for this symbol.
    pub epsilon: f64,
}

/// Maps M-ary symbols to cluster patterns and coupling strengths.
///
/// Each symbol is associated with a unique cluster pattern and coupling
/// strength ε. The symbol map validates the covertness condition: channel
/// link nodes must never be co-clustered for any symbol.
#[derive(Debug, Clone)]
pub struct SymbolMap {
    entries: Vec<SymbolEntry>,
    /// Channel link node indices (the nodes whose signals are transmitted).
    channel_links: Vec<usize>,
}

impl SymbolMap {
    /// Create a new symbol map from (symbol, pattern, epsilon) triplets.
    ///
    /// `channel_links` specifies the node indices whose signals are sent
    /// over the physical channel. The covertness condition requires that
    /// these nodes are never in the same cluster for any symbol.
    ///
    /// Requires at least 2 symbols. Symbols must be consecutive starting from 0.
    pub fn new(
        entries: Vec<(Symbol, ClusterPattern, f64)>,
        channel_links: Vec<usize>,
    ) -> Result<Self, CodecError> {
        if entries.len() < 2 {
            return Err(CodecError::EmptySymbolMap);
        }

        // Check that all patterns have the same number of nodes
        let n = entries[0].1.num_nodes();
        for (sym, pat, _) in &entries {
            if pat.num_nodes() != n {
                return Err(CodecError::InvalidSymbolMap {
                    reason: format!(
                        "symbol {sym}: pattern has {} nodes but expected {n}",
                        pat.num_nodes()
                    ),
                });
            }
        }

        // Validate channel link indices
        for &node in &channel_links {
            if node >= n {
                return Err(CodecError::InvalidSymbolMap {
                    reason: format!("channel link node {node} out of range (n={n})"),
                });
            }
        }

        // Check covertness condition: channel link nodes must not share a cluster
        for (sym, pat, _) in &entries {
            for (i, &a) in channel_links.iter().enumerate() {
                for &b in channel_links.iter().skip(i + 1) {
                    if pat.are_same_cluster(a, b) {
                        return Err(CodecError::CovertnessViolation {
                            node_a: a,
                            node_b: b,
                            symbol: *sym,
                        });
                    }
                }
            }
        }

        // Validate symbols are consecutive from 0
        let mut symbols: Vec<Symbol> = entries.iter().map(|(s, _, _)| *s).collect();
        symbols.sort();
        for (i, &s) in symbols.iter().enumerate() {
            if s != i {
                return Err(CodecError::InvalidSymbolMap {
                    reason: format!("symbols must be consecutive from 0, gap at index {i}"),
                });
            }
        }

        // Check for duplicate symbols
        symbols.dedup();
        if symbols.len() != entries.len() {
            return Err(CodecError::InvalidSymbolMap {
                reason: "duplicate symbols".to_string(),
            });
        }

        // Build sorted entries
        let mut sorted_entries: Vec<SymbolEntry> = entries
            .into_iter()
            .map(|(symbol, pattern, epsilon)| SymbolEntry {
                symbol,
                pattern,
                epsilon,
            })
            .collect();
        sorted_entries.sort_by_key(|e| e.symbol);

        Ok(Self {
            entries: sorted_entries,
            channel_links,
        })
    }

    /// Create a binary CLSK symbol map for the octagon topology.
    ///
    /// Symbol 0 → pattern P₀ with ε₀
    /// Symbol 1 → pattern P₁ with ε₁
    ///
    /// Channel links are the nodes connecting the two clusters.
    pub fn binary(
        pattern_0: ClusterPattern,
        epsilon_0: f64,
        pattern_1: ClusterPattern,
        epsilon_1: f64,
        channel_links: Vec<usize>,
    ) -> Result<Self, CodecError> {
        Self::new(
            vec![(0, pattern_0, epsilon_0), (1, pattern_1, epsilon_1)],
            channel_links,
        )
    }

    /// Number of symbols in the alphabet.
    pub fn alphabet_size(&self) -> usize {
        self.entries.len()
    }

    /// Look up the coupling strength ε for a symbol.
    pub fn lookup_epsilon(&self, symbol: Symbol) -> Result<f64, CodecError> {
        self.entries
            .get(symbol)
            .map(|e| e.epsilon)
            .ok_or(CodecError::UnknownSymbol { symbol })
    }

    /// Look up the cluster pattern for a symbol.
    pub fn lookup_pattern(&self, symbol: Symbol) -> Result<&ClusterPattern, CodecError> {
        self.entries
            .get(symbol)
            .map(|e| &e.pattern)
            .ok_or(CodecError::UnknownSymbol { symbol })
    }

    /// Get the full entry for a symbol.
    pub fn lookup(&self, symbol: Symbol) -> Result<&SymbolEntry, CodecError> {
        self.entries
            .get(symbol)
            .ok_or(CodecError::UnknownSymbol { symbol })
    }

    /// Get the channel link node indices.
    pub fn channel_links(&self) -> &[usize] {
        &self.channel_links
    }

    /// Get all entries.
    pub fn entries(&self) -> &[SymbolEntry] {
        &self.entries
    }

    /// Number of nodes in the network.
    pub fn num_nodes(&self) -> usize {
        self.entries[0].pattern.num_nodes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_patterns() -> (ClusterPattern, ClusterPattern) {
        // Pattern 0: {0,2,4,6} and {1,3,5,7}
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        // Pattern 1: {0,1,4,5} and {2,3,6,7}
        let p1 = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p1");
        (p0, p1)
    }

    #[test]
    fn symbol_map_creation() {
        let (p0, p1) = two_cluster_patterns();
        // Channel links: nodes 0 and 4 (in different clusters for both patterns)
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 10.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol map");

        assert_eq!(symbol_map.alphabet_size(), 2);
        assert!((symbol_map.lookup_epsilon(0).expect("eps0") - 10.0).abs() < 1e-15);
        assert!((symbol_map.lookup_epsilon(1).expect("eps1") - 12.0).abs() < 1e-15);
    }

    #[test]
    fn symbol_map_binary_helper() {
        let (p0, p1) = two_cluster_patterns();
        let symbol_map = SymbolMap::binary(p0, 10.0, p1, 12.0, vec![0, 3]).expect("binary map");
        assert_eq!(symbol_map.alphabet_size(), 2);
    }

    #[test]
    fn unknown_symbol_error() {
        let (p0, p1) = two_cluster_patterns();
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 10.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol map");

        assert!(symbol_map.lookup_epsilon(2).is_err());
        assert!(symbol_map.lookup_pattern(5).is_err());
    }

    #[test]
    fn too_few_symbols() {
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p0");
        let result = SymbolMap::new(vec![(0, p0, 10.0)], vec![0, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn covertness_violation_detected() {
        // Pattern where nodes 0 and 1 are in the same cluster
        let p_bad = ClusterPattern::new(vec![0, 0, 1, 1, 0, 0, 1, 1]).expect("p_bad");
        let p_good = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("p_good");

        // Channel links: 0 and 1. In p_bad, they're in the same cluster → violation
        let result = SymbolMap::new(vec![(0, p_bad, 10.0), (1, p_good, 12.0)], vec![0, 1]);
        assert!(result.is_err());
        if let Err(CodecError::CovertnessViolation {
            node_a,
            node_b,
            symbol,
        }) = result
        {
            assert_eq!(node_a, 0);
            assert_eq!(node_b, 1);
            assert_eq!(symbol, 0);
        } else {
            panic!("expected CovertnessViolation");
        }
    }

    #[test]
    fn non_consecutive_symbols_error() {
        let (p0, p1) = two_cluster_patterns();
        let result = SymbolMap::new(
            vec![(0, p0, 10.0), (2, p1, 12.0)], // gap at 1
            vec![0, 3],
        );
        assert!(result.is_err());
    }

    #[test]
    fn channel_link_out_of_range() {
        let (p0, p1) = two_cluster_patterns();
        let result = SymbolMap::new(
            vec![(0, p0, 10.0), (1, p1, 12.0)],
            vec![0, 99], // node 99 out of range
        );
        assert!(result.is_err());
    }

    #[test]
    fn lookup_pattern() {
        let (p0, p1) = two_cluster_patterns();
        let p0_clone = p0.clone();
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 10.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol map");

        let pattern = symbol_map.lookup_pattern(0).expect("pattern 0");
        assert_eq!(*pattern, p0_clone);
    }

    #[test]
    fn channel_links_accessor() {
        let (p0, p1) = two_cluster_patterns();
        let symbol_map =
            SymbolMap::new(vec![(0, p0, 10.0), (1, p1, 12.0)], vec![0, 3]).expect("symbol map");

        assert_eq!(symbol_map.channel_links(), &[0, 3]);
    }

    #[test]
    fn node_count_mismatch() {
        let p0 = ClusterPattern::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).expect("8 nodes");
        let p1 = ClusterPattern::new(vec![0, 1, 0, 1]).expect("4 nodes");
        let result = SymbolMap::new(vec![(0, p0, 10.0), (1, p1, 12.0)], vec![0, 3]);
        assert!(result.is_err());
    }
}
