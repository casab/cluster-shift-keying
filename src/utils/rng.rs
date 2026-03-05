use rand::SeedableRng;

/// Seeded RNG wrapper for deterministic simulations.
///
/// Uses `SmallRng` for performance in tight loops.
/// All stochastic components (noise, initial conditions) should
/// derive their RNG from this to ensure reproducibility.
pub type SeededRng = rand::rngs::SmallRng;

/// Create a deterministic RNG from a seed value.
pub fn create_rng(seed: u64) -> SeededRng {
    SeededRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_rng_produces_same_sequence() {
        use rand::Rng;

        let mut rng1 = create_rng(42);
        let mut rng2 = create_rng(42);

        let values1: Vec<f64> = (0..100).map(|_| rng1.gen()).collect();
        let values2: Vec<f64> = (0..100).map(|_| rng2.gen()).collect();

        assert_eq!(values1, values2);
    }

    #[test]
    fn different_seeds_produce_different_sequences() {
        use rand::Rng;

        let mut rng1 = create_rng(42);
        let mut rng2 = create_rng(43);

        let values1: Vec<f64> = (0..100).map(|_| rng1.gen()).collect();
        let values2: Vec<f64> = (0..100).map(|_| rng2.gen()).collect();

        assert_ne!(values1, values2);
    }
}
