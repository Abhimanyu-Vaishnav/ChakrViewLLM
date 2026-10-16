# Advanced Programmable Interrupt Controller and Vector Dispatch

## 1. Interrupt Subsystems
- Vector allocation: Fixed vector ranges from 32 to 255 allocated across symmetric multicore processing units.
- Non-maskable interrupt (NMI) delivery: Dedicated physical pin triggering panic handlers during uncorrectable ECC faults.
- Inter-processor interrupts (IPI): Directed unicast signals coordinate cache invalidation shootdowns across CPU sockets.
- Interrupt mitigation window: Dynamic interrupt coalescing delays packet arrivals by 50 microseconds under high throughput.
- Direct Memory Access (DMA): Ring buffer circular descriptors receive raw packets without CPU intervention.
