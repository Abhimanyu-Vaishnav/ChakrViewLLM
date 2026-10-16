# TAGE Branch Predictor and Speculative Execution

## 1. Predictor Structures
- Tagged geometric history: Multiple predictor tables indexed by geometric progression of global branch histories.
- Branch Target Buffer (BTB): 4096-entry associative array caches indirect branch target addresses for immediate fetch.
- Return Stack Buffer (RSB): 32-entry circular stack predicts subroutine return pointers with 99.4 percent accuracy.
- Misprediction penalty: Pipeline flush incurs 16 clock cycles of instruction re-fetch from primary instruction caches.
- Speculative window depth: Reorder buffer tracks up to 256 instructions in flight awaiting in-order retirement.
