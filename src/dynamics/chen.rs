use super::error::DynamicsError;
use super::traits::DynamicalSystem;

/// Default Chen system parameters (from the paper).
/// ẋ₁ = a(x₂ - x₁)
/// ẋ₂ = (c - a)x₁ - x₁x₃ + cx₂
/// ẋ₃ = x₁x₂ - bx₃
pub const DEFAULT_CHEN_A: f64 = 35.0;
pub const DEFAULT_CHEN_B: f64 = 8.0 / 3.0;
pub const DEFAULT_CHEN_C: f64 = 28.0;

/// Chen chaotic attractor.
///
/// A 3D autonomous continuous-time dynamical system exhibiting chaotic
/// behavior for the default parameters (a=35, b=8/3, c=28).
#[derive(Debug, Clone)]
pub struct ChenSystem {
    /// Parameter a (controls coupling between x₁ and x₂)
    a: f64,
    /// Parameter b (damping on x₃)
    b: f64,
    /// Parameter c (drives the nonlinear dynamics)
    c: f64,
}

impl ChenSystem {
    /// Create a new Chen system with the given parameters.
    ///
    /// Returns an error if any parameter is non-finite.
    pub fn new(a: f64, b: f64, c: f64) -> Result<Self, DynamicsError> {
        if !a.is_finite() {
            return Err(DynamicsError::InvalidParameter {
                name: "a".to_string(),
                reason: format!("must be finite, got {a}"),
            });
        }
        if !b.is_finite() {
            return Err(DynamicsError::InvalidParameter {
                name: "b".to_string(),
                reason: format!("must be finite, got {b}"),
            });
        }
        if !c.is_finite() {
            return Err(DynamicsError::InvalidParameter {
                name: "c".to_string(),
                reason: format!("must be finite, got {c}"),
            });
        }
        Ok(Self { a, b, c })
    }

    /// Create a Chen system with the default paper parameters.
    pub fn default_paper() -> Self {
        // These constants are always valid, so this cannot fail.
        Self {
            a: DEFAULT_CHEN_A,
            b: DEFAULT_CHEN_B,
            c: DEFAULT_CHEN_C,
        }
    }

    /// Get parameter a.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get parameter b.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Get parameter c.
    pub fn c(&self) -> f64 {
        self.c
    }

    /// Helper to check dimension of a slice.
    fn check_dim(&self, slice: &[f64], label: &str) -> Result<(), DynamicsError> {
        if slice.len() != 3 {
            return Err(DynamicsError::DimensionMismatch {
                expected: 3,
                got: slice.len(),
            });
        }
        let _ = label;
        Ok(())
    }
}

impl DynamicalSystem for ChenSystem {
    fn dimension(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "Chen"
    }

    /// Compute ẋ = f(x) for the Chen system.
    ///
    /// ẋ₁ = a(x₂ - x₁)
    /// ẋ₂ = (c - a)x₁ - x₁x₃ + cx₂
    /// ẋ₃ = x₁x₂ - bx₃
    fn derivative(&self, state: &[f64], output: &mut [f64]) -> Result<(), DynamicsError> {
        self.check_dim(state, "state")?;
        self.check_dim(output, "output")?;

        let (x1, x2, x3) = (state[0], state[1], state[2]);
        let (a, b, c) = (self.a, self.b, self.c);

        output[0] = a * (x2 - x1);
        output[1] = (c - a) * x1 - x1 * x3 + c * x2;
        output[2] = x1 * x2 - b * x3;

        Ok(())
    }

    /// Compute the Jacobian Df(x) of the Chen system (row-major, 3x3).
    ///
    /// Df = [[-a,     a,    0   ],
    ///       [c-a-x₃, c,   -x₁ ],
    ///       [x₂,     x₁,  -b  ]]
    fn jacobian(&self, state: &[f64], jacobian: &mut [f64]) -> Result<(), DynamicsError> {
        self.check_dim(state, "state")?;
        if jacobian.len() != 9 {
            return Err(DynamicsError::DimensionMismatch {
                expected: 9,
                got: jacobian.len(),
            });
        }

        let (x1, x2, x3) = (state[0], state[1], state[2]);
        let (a, b, c) = (self.a, self.b, self.c);

        // Row 0: d(ẋ₁)/d(x₁, x₂, x₃)
        jacobian[0] = -a;
        jacobian[1] = a;
        jacobian[2] = 0.0;

        // Row 1: d(ẋ₂)/d(x₁, x₂, x₃)
        jacobian[3] = c - a - x3;
        jacobian[4] = c;
        jacobian[5] = -x1;

        // Row 2: d(ẋ₃)/d(x₁, x₂, x₃)
        jacobian[6] = x2;
        jacobian[7] = x1;
        jacobian[8] = -b;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params() {
        let chen = ChenSystem::default_paper();
        assert!((chen.a() - 35.0).abs() < 1e-15);
        assert!((chen.b() - 8.0 / 3.0).abs() < 1e-15);
        assert!((chen.c() - 28.0).abs() < 1e-15);
        assert_eq!(chen.dimension(), 3);
        assert_eq!(chen.name(), "Chen");
    }

    #[test]
    fn invalid_params() {
        assert!(ChenSystem::new(f64::NAN, 1.0, 1.0).is_err());
        assert!(ChenSystem::new(1.0, f64::INFINITY, 1.0).is_err());
        assert!(ChenSystem::new(1.0, 1.0, f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn derivative_at_origin() {
        let chen = ChenSystem::default_paper();
        let state = [0.0, 0.0, 0.0];
        let mut output = [0.0; 3];
        chen.derivative(&state, &mut output)
            .expect("derivative at origin");
        // At origin, all derivatives should be zero
        assert!((output[0]).abs() < 1e-15);
        assert!((output[1]).abs() < 1e-15);
        assert!((output[2]).abs() < 1e-15);
    }

    #[test]
    fn derivative_at_known_point() {
        let chen = ChenSystem::default_paper();
        let state = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        chen.derivative(&state, &mut output)
            .expect("derivative at known point");

        let a = 35.0;
        let b = 8.0 / 3.0;
        let c = 28.0;
        // ẋ₁ = a(x₂ - x₁) = 35*(2-1) = 35
        assert!((output[0] - a * (2.0 - 1.0)).abs() < 1e-12);
        // ẋ₂ = (c-a)*x₁ - x₁*x₃ + c*x₂ = (28-35)*1 - 1*3 + 28*2 = -7 - 3 + 56 = 46
        assert!((output[1] - ((c - a) * 1.0 - 1.0 * 3.0 + c * 2.0)).abs() < 1e-12);
        // ẋ₃ = x₁*x₂ - b*x₃ = 1*2 - (8/3)*3 = 2 - 8 = -6
        assert!((output[2] - (1.0 * 2.0 - b * 3.0)).abs() < 1e-12);
    }

    #[test]
    fn derivative_dimension_mismatch() {
        let chen = ChenSystem::default_paper();
        let state = [1.0, 2.0]; // wrong size
        let mut output = [0.0; 3];
        assert!(chen.derivative(&state, &mut output).is_err());

        let state = [1.0, 2.0, 3.0];
        let mut output = [0.0; 2]; // wrong size
        assert!(chen.derivative(&state, &mut output).is_err());
    }

    #[test]
    fn jacobian_at_known_point() {
        let chen = ChenSystem::default_paper();
        let state = [1.0, 2.0, 3.0];
        let mut jac = [0.0; 9];
        chen.jacobian(&state, &mut jac).expect("jacobian");

        let a = 35.0;
        let b = 8.0 / 3.0;
        let c = 28.0;

        assert!((jac[0] - (-a)).abs() < 1e-12);
        assert!((jac[1] - a).abs() < 1e-12);
        assert!((jac[2]).abs() < 1e-12);
        assert!((jac[3] - (c - a - 3.0)).abs() < 1e-12);
        assert!((jac[4] - c).abs() < 1e-12);
        assert!((jac[5] - (-1.0)).abs() < 1e-12);
        assert!((jac[6] - 2.0).abs() < 1e-12);
        assert!((jac[7] - 1.0).abs() < 1e-12);
        assert!((jac[8] - (-b)).abs() < 1e-12);
    }

    #[test]
    fn jacobian_dimension_mismatch() {
        let chen = ChenSystem::default_paper();
        let state = [1.0, 2.0, 3.0];
        let mut jac = [0.0; 4]; // wrong
        assert!(chen.jacobian(&state, &mut jac).is_err());
    }
}
