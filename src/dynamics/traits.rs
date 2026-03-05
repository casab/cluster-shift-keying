/// Core trait for autonomous continuous-time dynamical systems.
///
/// Implementations represent systems of the form ẋ = f(x),
/// e.g. the Chen attractor, Rössler system, etc.
pub trait DynamicalSystem: Send + Sync {
    /// Number of state variables (dimension of the phase space).
    fn dimension(&self) -> usize;

    /// Compute the time derivative f(x) at the given `state`.
    ///
    /// Writes the result into `output`, which must have length == `self.dimension()`.
    /// Returns an error if slice lengths are incorrect.
    fn derivative(&self, state: &[f64], output: &mut [f64]) -> Result<(), DynamicsError>;

    /// Human-readable name of the system (e.g. "Chen", "Rössler").
    fn name(&self) -> &str;

    /// Compute the Jacobian matrix Df(x) at the given `state`.
    ///
    /// Writes into `jacobian` which must have length == dimension * dimension (row-major).
    /// Default implementation returns `Err(DynamicsError::JacobianNotImplemented)`.
    fn jacobian(&self, _state: &[f64], _jacobian: &mut [f64]) -> Result<(), DynamicsError> {
        Err(DynamicsError::JacobianNotImplemented {
            system: self.name().to_string(),
        })
    }
}

use super::error::DynamicsError;
