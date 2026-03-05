pub mod ber;
pub mod error;
pub mod stats;
pub mod sync_energy;

pub use ber::{BerCurve, BerEvaluator, BerPoint, TrialResult};
pub use error::MetricsError;
pub use stats::{MonteCarloConfig, MonteCarloRunner, ProgressCallback, TrialRunner};
pub use sync_energy::{
    BinarySyncMatrix, MinIntraScoring, RatioScoring, ScoringFunction, SyncEnergyDetector,
    SyncEnergyMatrix,
};
