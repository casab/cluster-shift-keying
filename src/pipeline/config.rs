use serde::{Deserialize, Serialize};

use super::error::PipelineError;

/// Top-level simulation configuration, deserializable from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Dynamical system parameters.
    pub system: SystemConfig,
    /// Network topology configuration.
    pub topology: TopologyConfig,
    /// Coupling and symbol mapping configuration.
    pub coupling: CouplingConfig,
    /// Codec configuration (timing, framing).
    pub codec: CodecConfig,
    /// Channel model configuration.
    pub channel: ChannelConfig,
    /// Simulation run parameters.
    pub simulation: RunConfig,
}

impl SimulationConfig {
    /// Load configuration from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, PipelineError> {
        let config: Self = toml::from_str(toml_str).map_err(|e| PipelineError::ConfigError {
            reason: format!("TOML parse error: {e}"),
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Serialize configuration to a TOML string.
    pub fn to_toml(&self) -> Result<String, PipelineError> {
        toml::to_string_pretty(self).map_err(|e| PipelineError::ConfigError {
            reason: format!("TOML serialize error: {e}"),
        })
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), PipelineError> {
        self.system.validate()?;
        self.topology.validate()?;
        self.coupling.validate()?;
        self.codec.validate()?;
        self.channel.validate()?;
        self.simulation.validate()?;
        Ok(())
    }

    /// Create a default configuration matching the paper's octagon setup.
    pub fn default_paper() -> Self {
        Self {
            system: SystemConfig::default(),
            topology: TopologyConfig::default(),
            coupling: CouplingConfig::default(),
            codec: CodecConfig::default(),
            channel: ChannelConfig::default(),
            simulation: RunConfig::default(),
        }
    }
}

/// Dynamical system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System type: "chen" (only supported type currently).
    #[serde(default = "default_system_type")]
    pub system_type: String,
    /// Chen parameter a.
    #[serde(default = "default_chen_a")]
    pub a: f64,
    /// Chen parameter b.
    #[serde(default = "default_chen_b")]
    pub b: f64,
    /// Chen parameter c.
    #[serde(default = "default_chen_c")]
    pub c: f64,
}

fn default_system_type() -> String {
    "chen".to_string()
}
fn default_chen_a() -> f64 {
    35.0
}
fn default_chen_b() -> f64 {
    8.0 / 3.0
}
fn default_chen_c() -> f64 {
    28.0
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            system_type: default_system_type(),
            a: default_chen_a(),
            b: default_chen_b(),
            c: default_chen_c(),
        }
    }
}

impl SystemConfig {
    fn validate(&self) -> Result<(), PipelineError> {
        if self.system_type != "chen" {
            return Err(PipelineError::ConfigError {
                reason: format!("unsupported system type: {}", self.system_type),
            });
        }
        if !self.a.is_finite() || !self.b.is_finite() || !self.c.is_finite() {
            return Err(PipelineError::ConfigError {
                reason: "system parameters must be finite".to_string(),
            });
        }
        Ok(())
    }
}

/// Network topology configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Topology type: "octagon" or "ring".
    #[serde(default = "default_topology_type")]
    pub topology_type: String,
    /// Number of nodes (used for "ring" topology).
    #[serde(default = "default_node_count")]
    pub node_count: usize,
}

fn default_topology_type() -> String {
    "octagon".to_string()
}
fn default_node_count() -> usize {
    8
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: default_topology_type(),
            node_count: default_node_count(),
        }
    }
}

impl TopologyConfig {
    fn validate(&self) -> Result<(), PipelineError> {
        match self.topology_type.as_str() {
            "octagon" => {
                if self.node_count != 8 {
                    return Err(PipelineError::ConfigError {
                        reason: "octagon topology requires node_count = 8".to_string(),
                    });
                }
            }
            "ring" => {
                if self.node_count < 3 {
                    return Err(PipelineError::ConfigError {
                        reason: "ring topology requires at least 3 nodes".to_string(),
                    });
                }
            }
            other => {
                return Err(PipelineError::ConfigError {
                    reason: format!("unsupported topology: {other}"),
                });
            }
        }
        Ok(())
    }
}

/// Coupling and symbol mapping configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingConfig {
    /// Inner coupling matrix diagonal: Γ = diag(gamma).
    #[serde(default = "default_gamma")]
    pub gamma: Vec<f64>,
    /// Symbol definitions: each entry is (epsilon, pattern).
    pub symbols: Vec<SymbolConfig>,
    /// Channel link node indices.
    pub channel_links: Vec<usize>,
}

fn default_gamma() -> Vec<f64> {
    vec![0.0, 1.0, 0.0]
}

impl Default for CouplingConfig {
    fn default() -> Self {
        Self {
            gamma: default_gamma(),
            symbols: vec![
                SymbolConfig {
                    epsilon: 8.0,
                    pattern: vec![0, 1, 0, 1, 0, 1, 0, 1],
                },
                SymbolConfig {
                    epsilon: 12.0,
                    pattern: vec![0, 0, 1, 1, 0, 0, 1, 1],
                },
            ],
            channel_links: vec![0, 3],
        }
    }
}

impl CouplingConfig {
    fn validate(&self) -> Result<(), PipelineError> {
        if self.symbols.len() < 2 {
            return Err(PipelineError::ConfigError {
                reason: "at least 2 symbols required".to_string(),
            });
        }
        for (i, sym) in self.symbols.iter().enumerate() {
            if !sym.epsilon.is_finite() || sym.epsilon <= 0.0 {
                return Err(PipelineError::ConfigError {
                    reason: format!("symbol {i}: epsilon must be positive and finite"),
                });
            }
            if sym.pattern.is_empty() {
                return Err(PipelineError::ConfigError {
                    reason: format!("symbol {i}: pattern must not be empty"),
                });
            }
        }
        if self.channel_links.is_empty() {
            return Err(PipelineError::ConfigError {
                reason: "channel_links must not be empty".to_string(),
            });
        }
        Ok(())
    }
}

/// Configuration for a single symbol in the alphabet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolConfig {
    /// Coupling strength ε for this symbol.
    pub epsilon: f64,
    /// Cluster pattern assignment for this symbol.
    pub pattern: Vec<usize>,
}

/// Codec timing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Bit period T_b in time units.
    #[serde(default = "default_bit_period")]
    pub bit_period: f64,
    /// Guard interval between symbols (time units).
    #[serde(default)]
    pub guard_interval: f64,
    /// Integration time step dt.
    #[serde(default = "default_dt")]
    pub dt: f64,
    /// Initial state for oscillators.
    #[serde(default = "default_initial_state")]
    pub initial_state: Vec<f64>,
}

fn default_bit_period() -> f64 {
    10.0
}
fn default_dt() -> f64 {
    0.001
}
fn default_initial_state() -> Vec<f64> {
    vec![1.0, 1.0, 1.0]
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            bit_period: default_bit_period(),
            guard_interval: 0.0,
            dt: default_dt(),
            initial_state: default_initial_state(),
        }
    }
}

impl CodecConfig {
    fn validate(&self) -> Result<(), PipelineError> {
        if self.bit_period <= 0.0 || !self.bit_period.is_finite() {
            return Err(PipelineError::ConfigError {
                reason: format!(
                    "bit_period must be positive and finite: {}",
                    self.bit_period
                ),
            });
        }
        if self.guard_interval < 0.0 || !self.guard_interval.is_finite() {
            return Err(PipelineError::ConfigError {
                reason: format!(
                    "guard_interval must be non-negative and finite: {}",
                    self.guard_interval
                ),
            });
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(PipelineError::ConfigError {
                reason: format!("dt must be positive and finite: {}", self.dt),
            });
        }
        if self.dt > self.bit_period {
            return Err(PipelineError::ConfigError {
                reason: format!(
                    "dt ({}) must not exceed bit_period ({})",
                    self.dt, self.bit_period
                ),
            });
        }
        Ok(())
    }
}

/// Channel model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel type: "ideal" or "gaussian".
    #[serde(default = "default_channel_type")]
    pub channel_type: String,
    /// Noise standard deviation σ (for gaussian channel).
    #[serde(default)]
    pub sigma: f64,
    /// Noise mode: "additive" or "multiplicative".
    #[serde(default = "default_noise_mode")]
    pub noise_mode: String,
    /// RNG seed for channel noise.
    #[serde(default = "default_channel_seed")]
    pub seed: u64,
}

fn default_channel_type() -> String {
    "ideal".to_string()
}
fn default_noise_mode() -> String {
    "additive".to_string()
}
fn default_channel_seed() -> u64 {
    42
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            channel_type: default_channel_type(),
            sigma: 0.0,
            noise_mode: default_noise_mode(),
            seed: default_channel_seed(),
        }
    }
}

impl ChannelConfig {
    fn validate(&self) -> Result<(), PipelineError> {
        match self.channel_type.as_str() {
            "ideal" => {}
            "gaussian" => {
                if self.sigma < 0.0 || !self.sigma.is_finite() {
                    return Err(PipelineError::ConfigError {
                        reason: format!("sigma must be non-negative and finite: {}", self.sigma),
                    });
                }
                match self.noise_mode.as_str() {
                    "additive" | "multiplicative" => {}
                    other => {
                        return Err(PipelineError::ConfigError {
                            reason: format!("unsupported noise mode: {other}"),
                        });
                    }
                }
            }
            other => {
                return Err(PipelineError::ConfigError {
                    reason: format!("unsupported channel type: {other}"),
                });
            }
        }
        Ok(())
    }
}

/// Simulation run parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Number of symbols to transmit per trial.
    #[serde(default = "default_num_symbols")]
    pub num_symbols: usize,
    /// Base RNG seed for symbol generation.
    #[serde(default = "default_run_seed")]
    pub seed: u64,
}

fn default_num_symbols() -> usize {
    100
}
fn default_run_seed() -> u64 {
    12345
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            num_symbols: default_num_symbols(),
            seed: default_run_seed(),
        }
    }
}

impl RunConfig {
    fn validate(&self) -> Result<(), PipelineError> {
        if self.num_symbols == 0 {
            return Err(PipelineError::ConfigError {
                reason: "num_symbols must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let config = SimulationConfig::default_paper();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn roundtrip_toml() {
        let config = SimulationConfig::default_paper();
        let toml_str = config.to_toml().expect("serialize");
        let parsed = SimulationConfig::from_toml(&toml_str).expect("parse");
        assert_eq!(parsed.system.a, config.system.a);
        assert_eq!(parsed.topology.node_count, config.topology.node_count);
        assert_eq!(parsed.coupling.symbols.len(), config.coupling.symbols.len());
    }

    #[test]
    fn parse_minimal_toml() {
        let toml_str = r#"
[system]
[topology]
[coupling]
channel_links = [0, 3]
[[coupling.symbols]]
epsilon = 8.0
pattern = [0, 1, 0, 1, 0, 1, 0, 1]
[[coupling.symbols]]
epsilon = 12.0
pattern = [0, 0, 1, 1, 0, 0, 1, 1]
[codec]
[channel]
[simulation]
"#;
        let config = SimulationConfig::from_toml(toml_str).expect("parse minimal");
        assert_eq!(config.system.system_type, "chen");
        assert_eq!(config.topology.topology_type, "octagon");
    }

    #[test]
    fn invalid_system_type() {
        let mut config = SimulationConfig::default_paper();
        config.system.system_type = "lorenz".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_topology() {
        let mut config = SimulationConfig::default_paper();
        config.topology.topology_type = "star".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_ring_too_small() {
        let mut config = SimulationConfig::default_paper();
        config.topology.topology_type = "ring".to_string();
        config.topology.node_count = 2;
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_too_few_symbols() {
        let mut config = SimulationConfig::default_paper();
        config.coupling.symbols = vec![SymbolConfig {
            epsilon: 8.0,
            pattern: vec![0, 1, 0, 1, 0, 1, 0, 1],
        }];
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_negative_epsilon() {
        let mut config = SimulationConfig::default_paper();
        config.coupling.symbols[0].epsilon = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_bit_period() {
        let mut config = SimulationConfig::default_paper();
        config.codec.bit_period = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_dt_exceeds_bit_period() {
        let mut config = SimulationConfig::default_paper();
        config.codec.dt = 20.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_channel_type() {
        let mut config = SimulationConfig::default_paper();
        config.channel.channel_type = "fading".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_negative_sigma() {
        let mut config = SimulationConfig::default_paper();
        config.channel.channel_type = "gaussian".to_string();
        config.channel.sigma = -0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_zero_symbols() {
        let mut config = SimulationConfig::default_paper();
        config.simulation.num_symbols = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn gaussian_channel_config() {
        let mut config = SimulationConfig::default_paper();
        config.channel.channel_type = "gaussian".to_string();
        config.channel.sigma = 0.5;
        config.channel.noise_mode = "multiplicative".to_string();
        config.validate().expect("valid gaussian config");
    }
}
