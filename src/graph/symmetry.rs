use super::error::GraphError;
use crate::linalg::Matrix;
use std::collections::HashMap;

/// Detects symmetries (automorphisms) of a graph from its adjacency matrix.
///
/// Uses color refinement (1-WL algorithm) to identify candidate orbits
/// and backtracking to enumerate all automorphisms.
pub struct SymmetryDetector;

impl SymmetryDetector {
    /// Find symmetry orbits of the graph via color refinement.
    ///
    /// Nodes in the same orbit are candidates for equivalence under the
    /// automorphism group. The color refinement algorithm (1-dimensional
    /// Weisfeiler-Leman) produces a partition that is at least as fine as
    /// the true orbit partition and is exact for almost all graphs.
    ///
    /// Returns orbits as groups of node indices, sorted by smallest member.
    pub fn find_orbits(adjacency: &Matrix) -> Result<Vec<Vec<usize>>, GraphError> {
        let n = adjacency.nrows();
        if n == 0 {
            return Ok(vec![]);
        }

        let neighbors = Self::neighbor_lists(adjacency, n)?;
        let colors = Self::color_refinement(n, &neighbors);

        // Group nodes by final color
        let mut orbit_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &c) in colors.iter().enumerate() {
            orbit_map.entry(c).or_default().push(i);
        }

        let mut orbits: Vec<Vec<usize>> = orbit_map.into_values().collect();
        for orbit in &mut orbits {
            orbit.sort();
        }
        orbits.sort_by_key(|o| o[0]);

        Ok(orbits)
    }

    /// Find all automorphisms of the graph (node permutations preserving adjacency).
    ///
    /// Uses backtracking with orbit-based pruning. For small graphs (n ≤ 16),
    /// this is efficient. Always includes the identity permutation.
    pub fn find_automorphisms(adjacency: &Matrix) -> Result<Vec<Vec<usize>>, GraphError> {
        let n = adjacency.nrows();
        if n == 0 {
            return Ok(vec![vec![]]);
        }

        let orbits = Self::find_orbits(adjacency)?;

        // Build orbit membership: which orbit each node belongs to
        let mut orbit_of = vec![0usize; n];
        for (oi, orbit) in orbits.iter().enumerate() {
            for &node in orbit {
                orbit_of[node] = oi;
            }
        }

        // Precompute adjacency as boolean matrix for fast comparison
        let adj_bool: Vec<Vec<bool>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| adjacency.get(i, j).is_ok_and(|v| v.abs() > 1e-12))
                    .collect()
            })
            .collect();

        let mut ctx = AutoSearchCtx {
            adj: &adj_bool,
            orbits: &orbits,
            orbit_of: &orbit_of,
            perm: vec![0usize; n],
            used: vec![false; n],
            results: Vec::new(),
        };

        ctx.backtrack(n, 0);

        Ok(ctx.results)
    }

    /// Compute neighbor lists from adjacency matrix.
    fn neighbor_lists(adjacency: &Matrix, n: usize) -> Result<Vec<Vec<usize>>, GraphError> {
        let mut neighbors = vec![Vec::new(); n];
        for (i, nbrs) in neighbors.iter_mut().enumerate() {
            for j in 0..n {
                if i != j && adjacency.get(i, j)?.abs() > 1e-12 {
                    nbrs.push(j);
                }
            }
        }
        Ok(neighbors)
    }

    /// Color refinement (1-WL) algorithm.
    ///
    /// Starts with degree-based coloring, then iteratively refines: each node's
    /// new color encodes (old_color, sorted_multiset_of_neighbor_colors). Stops
    /// when the coloring stabilizes.
    fn color_refinement(n: usize, neighbors: &[Vec<usize>]) -> Vec<usize> {
        // Initial coloring by degree
        let mut colors = vec![0usize; n];
        let mut degree_map: HashMap<usize, usize> = HashMap::new();
        let mut next_color = 0;
        for (i, nbrs) in neighbors.iter().enumerate() {
            let deg = nbrs.len();
            let c = *degree_map.entry(deg).or_insert_with(|| {
                let c = next_color;
                next_color += 1;
                c
            });
            colors[i] = c;
        }

        // Iterative refinement
        loop {
            let mut new_colors = vec![0usize; n];
            let mut sig_map: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
            let mut nc = 0;

            for i in 0..n {
                let mut nbr_colors: Vec<usize> = neighbors[i].iter().map(|&j| colors[j]).collect();
                nbr_colors.sort();
                let sig = (colors[i], nbr_colors);
                let c = *sig_map.entry(sig).or_insert_with(|| {
                    let c = nc;
                    nc += 1;
                    c
                });
                new_colors[i] = c;
            }

            if new_colors == colors {
                break;
            }
            colors = new_colors;
        }

        colors
    }
}

/// Internal context for automorphism backtracking search,
/// bundling related state to reduce parameter count.
struct AutoSearchCtx<'a> {
    adj: &'a [Vec<bool>],
    orbits: &'a [Vec<usize>],
    orbit_of: &'a [usize],
    perm: Vec<usize>,
    used: Vec<bool>,
    results: Vec<Vec<usize>>,
}

impl AutoSearchCtx<'_> {
    /// Backtracking search: at position `pos`, assign perm[pos] to a node
    /// in the same orbit that preserves adjacency with all already-assigned nodes.
    fn backtrack(&mut self, n: usize, pos: usize) {
        if pos == n {
            self.results.push(self.perm.clone());
            return;
        }

        let orbit_idx = self.orbit_of[pos];
        let candidates: Vec<usize> = self.orbits[orbit_idx].clone();

        for candidate in candidates {
            if self.used[candidate] {
                continue;
            }

            // Check adjacency consistency with all already-assigned nodes
            let mut consistent = true;
            for i in 0..pos {
                if self.adj[pos][i] != self.adj[candidate][self.perm[i]] {
                    consistent = false;
                    break;
                }
            }

            if consistent {
                self.perm[pos] = candidate;
                self.used[candidate] = true;
                self.backtrack(n, pos + 1);
                self.used[candidate] = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::topology::TopologyBuilder;

    #[test]
    fn ring_all_nodes_equivalent() {
        let cm = TopologyBuilder::ring(6).expect("ring6");
        let orbits = SymmetryDetector::find_orbits(cm.adjacency()).expect("orbits");
        // C₆ is vertex-transitive: all nodes in one orbit
        assert_eq!(orbits.len(), 1, "ring6 should have 1 orbit: {orbits:?}");
        assert_eq!(orbits[0].len(), 6);
    }

    #[test]
    fn complete_all_nodes_equivalent() {
        let cm = TopologyBuilder::complete(5).expect("K5");
        let orbits = SymmetryDetector::find_orbits(cm.adjacency()).expect("orbits");
        // K₅ is vertex-transitive
        assert_eq!(orbits.len(), 1, "K5 should have 1 orbit");
    }

    #[test]
    fn octagon_all_nodes_equivalent() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let orbits = SymmetryDetector::find_orbits(cm.adjacency()).expect("orbits");
        // C₈ is vertex-transitive
        assert_eq!(orbits.len(), 1, "octagon should have 1 orbit");
    }

    #[test]
    fn star_graph_two_orbits() {
        // Star graph S₄: node 0 connected to 1,2,3
        let n = 4;
        let mut edges = Vec::new();
        for i in 1..n {
            edges.push((0, i, 1.0));
            edges.push((i, 0, 1.0));
        }
        let adj = Matrix::from_adjacency(n, &edges).expect("adj");
        let orbits = SymmetryDetector::find_orbits(&adj).expect("orbits");
        // Node 0 (degree 3) vs nodes 1,2,3 (degree 1): 2 orbits
        assert_eq!(orbits.len(), 2, "star4 should have 2 orbits: {orbits:?}");
    }

    #[test]
    fn automorphisms_ring4() {
        let cm = TopologyBuilder::ring(4).expect("ring4");
        let autos = SymmetryDetector::find_automorphisms(cm.adjacency()).expect("autos");
        // C₄ has dihedral group D₄ with 8 elements
        assert_eq!(
            autos.len(),
            8,
            "ring4 should have 8 automorphisms (D₄), got {}",
            autos.len()
        );
        // Identity must be present
        assert!(autos.contains(&vec![0, 1, 2, 3]));
    }

    #[test]
    fn automorphisms_complete4() {
        let cm = TopologyBuilder::complete(4).expect("K4");
        let autos = SymmetryDetector::find_automorphisms(cm.adjacency()).expect("autos");
        // K₄ has S₄ with 24 elements
        assert_eq!(
            autos.len(),
            24,
            "K4 should have 24 automorphisms (S₄), got {}",
            autos.len()
        );
    }

    #[test]
    fn automorphisms_preserve_adjacency() {
        let cm = TopologyBuilder::ring(5).expect("ring5");
        let adj = cm.adjacency();
        let autos = SymmetryDetector::find_automorphisms(adj).expect("autos");

        for perm in &autos {
            // Check A[i][j] == A[perm[i]][perm[j]] for all i,j
            for i in 0..5 {
                for j in 0..5 {
                    let aij = adj.get(i, j).unwrap();
                    let apipj = adj.get(perm[i], perm[j]).unwrap();
                    assert!(
                        (aij - apipj).abs() < 1e-12,
                        "automorphism {:?} doesn't preserve edge ({i},{j})",
                        perm
                    );
                }
            }
        }
    }

    #[test]
    fn automorphisms_octagon() {
        let cm = TopologyBuilder::octagon().expect("octagon");
        let autos = SymmetryDetector::find_automorphisms(cm.adjacency()).expect("autos");
        // C₈ has dihedral group D₈ with 16 elements
        assert_eq!(
            autos.len(),
            16,
            "octagon should have 16 automorphisms (D₈), got {}",
            autos.len()
        );
    }
}
