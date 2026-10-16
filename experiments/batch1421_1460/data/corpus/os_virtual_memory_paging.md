# Virtual Memory Management and Page Table Translation

## 1. Paging Architecture
- Hardware page table traversal: Four-level hierarchical page table structure with 48-bit canonical virtual address spaces.
- Translation Lookaside Buffer (TLB): Two-tiered split L1 data and instruction TLB with 64 fully associative entries.
- Page table walk latency: 14 to 28 nanoseconds on hardware misses before falling back to physical DRAM pages.
- Large page support: Transparent Huge Pages allocate contiguous 2 megabyte physical frames to reduce TLB misses.
- Page replacement algorithm: Generalized clock-pro algorithm approximating least recently used page evictions.
