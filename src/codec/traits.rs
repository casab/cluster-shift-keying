/// Trait for encoding symbols into the physical layer.
///
/// Implementations drive the transmitter network to produce
/// channel signals corresponding to the given symbol.
pub trait Encoder {
    /// The symbol type (e.g. `u8` for binary, `usize` for M-ary).
    type Symbol;
    /// Error type for encoding failures.
    type Error: std::error::Error;

    /// Encode a single symbol, advancing the internal network state.
    fn encode(&mut self, symbol: &Self::Symbol) -> Result<(), Self::Error>;
}

/// Trait for decoding symbols from received signals.
///
/// Implementations observe the receiver network state and
/// recover the transmitted symbol via energy detection.
pub trait Decoder {
    /// The symbol type being decoded.
    type Symbol;
    /// Error type for decoding failures.
    type Error: std::error::Error;

    /// Decode the next symbol from the current receiver state.
    fn decode(&mut self) -> Result<Self::Symbol, Self::Error>;
}

/// Composable codec chain: applies an outer codec around an inner one.
///
/// This enables layering (e.g. ECC encoding wrapping CLSK encoding)
/// without modifying either implementation.
pub struct CodecChain<Outer, Inner> {
    /// The outer codec (e.g. error-correcting code).
    pub outer: Outer,
    /// The inner codec (e.g. CLSK modulator/demodulator).
    pub inner: Inner,
}
