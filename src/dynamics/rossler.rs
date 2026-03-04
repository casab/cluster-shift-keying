use super::error::DynamicsError;
use super::traits::DynamicalSystem;

/// Rössler attractor — stub implementation demonstrating extensibility.
///
/// ẋ₁ = -x₂ - x₃
/// ẋ₂ = x₁ + a·x₂
/// ẋ₃ = b + x₃(x₁ - c)
///
/// Classic chaotic parameters: a = 0.2, b = 0.2, c = 5.7
#[derive(Debug, Clone)]
pub struct RosslerSystem {
    a: f64,
    b: f64,
    c: f64,
}

impl RosslerSystem {
    /// Create a Rössler system with the given parameters.
    pub fn new(a: f64, b: f64, c: f64) -> Result<Self, DynamicsError> {
        for (name, val) in [("a", a), ("b", b), ("c", c)] {
            if !val.is_finite() {
                return Err(DynamicsError::InvalidParameter {
                    name: name.to_string(),
                    reason: format!("must be finite, got {val}"),
                });
            }
        }
        Ok(Self { a, b, c })
    }

    /// Classic chaotic parameters.
    pub fn default_chaotic() -> Self {
        Self {
            a: 0.2,
            b: 0.2,
            c: 5.7,
        }
    }
}

impl DynamicalSystem for RosslerSystem {
    fn dimension(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "Rössler"
    }

    fn derivative(&self, state: &[f64], output: &mut [f64]) -> Result<(), DynamicsError> {
        if state.len() != 3 {
            return Err(DynamicsError::DimensionMismatch {
                expected: 3,
                got: state.len(),
            });
        }
        if output.len() != 3 {
            return Err(DynamicsError::DimensionMismatch {
                expected: 3,
                got: output.len(),
            });
        }

        let (x1, x2, x3) = (state[0], state[1], state[2]);

        output[0] = -x2 - x3;
        output[1] = x1 + self.a * x2;
        output[2] = self.b + x3 * (x1 - self.c);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::integrator::{Rk4, DEFAULT_DT};

    #[test]
    fn rossler_derivative_at_origin() {
        let sys = RosslerSystem::default_chaotic();
        let state = [0.0, 0.0, 0.0];
        let mut output = [0.0; 3];
        sys.derivative(&state, &mut output).expect("origin");
        assert!((output[0]).abs() < 1e-15);
        assert!((output[1]).abs() < 1e-15);
        // ẋ₃ = b + 0*(0-c) = b = 0.2
        assert!((output[2] - 0.2).abs() < 1e-15);
    }

    #[test]
    fn rossler_bounded() {
        let sys = RosslerSystem::default_chaotic();
        let mut rk4 = Rk4::new(3);
        let trajectory = rk4
            .integrate(&sys, &[1.0, 1.0, 0.0], DEFAULT_DT, 10_000)
            .expect("rossler trajectory");

        let bound = 50.0;
        for state in &trajectory {
            for &val in state {
                assert!(val.abs() < bound, "Rössler state {val} exceeds {bound}");
            }
        }
    }

    #[test]
    fn rossler_invalid_params() {
        assert!(RosslerSystem::new(f64::NAN, 0.2, 5.7).is_err());
    }
}
