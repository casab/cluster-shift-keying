pub mod demodulator;
pub mod error;
pub mod framing;
pub mod modulator;
pub mod multi_bit;
pub mod symbol_map;
pub mod traits;

pub use demodulator::{Demodulator, DemodulatorConfig, DemodulatorWithSystem};
pub use error::CodecError;
pub use framing::FrameConfig;
pub use modulator::{Modulator, ModulatorConfig, ModulatorWithSystem};
pub use multi_bit::{build_mary_clsk, MaryClskConfig, MaryClskSystem};
pub use symbol_map::{Symbol, SymbolEntry, SymbolMap};
pub use traits::{CodecChain, Decoder, Encoder};
