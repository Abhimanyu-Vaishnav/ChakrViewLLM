# Quantization and Weight Compression

## 1. Compression Schemes
- Static linear quantization: INT8 symmetric per-channel weights.
- Activation clipping range: Absolute magnitude bounded at [-6.0, 6.0].
- Calibration dataset footprint: Exactly 512 representative domain tokens.
- Quantized checkpoint file suffix: `.q8.pt`.
