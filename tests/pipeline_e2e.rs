use cluster_shift_keying::pipeline::config::SimulationConfig;
use cluster_shift_keying::pipeline::simulation::Simulation;

#[test]
fn pipeline_default_config_runs() {
    let mut config = SimulationConfig::default_paper();
    // Use short bit period and few symbols for fast test
    config.codec.bit_period = 2.0;
    config.simulation.num_symbols = 4;

    let sim = Simulation::new(config).expect("create simulation");
    let result = sim.run().expect("run simulation");

    assert_eq!(result.tx_symbols.len(), 4);
    assert_eq!(result.rx_symbols.len(), 4);
    assert!(result.ser >= 0.0 && result.ser <= 1.0);
}

#[test]
fn pipeline_from_toml() {
    let toml_str = r#"
[system]
system_type = "chen"
a = 35.0
b = 2.6666666666666665
c = 28.0

[topology]
topology_type = "octagon"
node_count = 8

[coupling]
channel_links = [0, 3]
gamma = [0.0, 1.0, 0.0]

[[coupling.symbols]]
epsilon = 8.0
pattern = [0, 1, 0, 1, 0, 1, 0, 1]

[[coupling.symbols]]
epsilon = 12.0
pattern = [0, 0, 1, 1, 0, 0, 1, 1]

[codec]
bit_period = 2.0
guard_interval = 0.0
dt = 0.001
initial_state = [1.0, 1.0, 1.0]

[channel]
channel_type = "ideal"

[simulation]
num_symbols = 4
seed = 42
"#;

    let config = SimulationConfig::from_toml(toml_str).expect("parse TOML");
    let sim = Simulation::new(config).expect("create simulation");
    let result = sim.run().expect("run simulation");

    assert_eq!(result.tx_symbols.len(), 4);
    assert_eq!(result.rx_symbols.len(), 4);
}

#[test]
fn pipeline_with_gaussian_channel() {
    let mut config = SimulationConfig::default_paper();
    config.codec.bit_period = 2.0;
    config.simulation.num_symbols = 4;
    config.channel.channel_type = "gaussian".to_string();
    config.channel.sigma = 0.01; // very low noise
    config.channel.seed = 123;

    let sim = Simulation::new(config).expect("create simulation");
    let result = sim.run().expect("run simulation");

    assert_eq!(result.tx_symbols.len(), 4);
    assert_eq!(result.rx_symbols.len(), 4);
}

#[test]
fn pipeline_deterministic_with_seed() {
    let mut config = SimulationConfig::default_paper();
    config.codec.bit_period = 1.0;
    config.simulation.num_symbols = 4;
    config.simulation.seed = 999;

    let sim1 = Simulation::new(config.clone()).expect("sim1");
    let result1 = sim1.run().expect("run1");

    let sim2 = Simulation::new(config).expect("sim2");
    let result2 = sim2.run().expect("run2");

    assert_eq!(result1.tx_symbols, result2.tx_symbols);
    assert_eq!(result1.rx_symbols, result2.rx_symbols);
    assert!((result1.ser - result2.ser).abs() < 1e-15);
}

#[test]
fn pipeline_config_roundtrip_toml() {
    let config = SimulationConfig::default_paper();
    let toml_str = config.to_toml().expect("serialize");
    let parsed = SimulationConfig::from_toml(&toml_str).expect("parse");

    assert!((parsed.system.a - 35.0).abs() < 1e-10);
    assert_eq!(parsed.topology.topology_type, "octagon");
    assert_eq!(parsed.coupling.symbols.len(), 2);
    assert_eq!(parsed.coupling.channel_links, vec![0, 3]);
}
