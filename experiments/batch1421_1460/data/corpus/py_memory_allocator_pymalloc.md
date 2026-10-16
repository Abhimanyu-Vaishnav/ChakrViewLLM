# Pymalloc Small Object Memory Arena Hierarchy

## 1. Allocation Tiers
- Small object ceiling: Allocations under 512 bytes handled by specialized arena allocators to reduce fragmentation.
- Arena sizing: Physical 256 kilobyte arenas subdivided into 4 kilobyte memory pools dedicated to uniform size classes.
- Size class indexing: 64 distinct size classes incrementing by 8 bytes satisfy requests from 8 to 512 bytes.
- Large object fallback: Memory requests exceeding 512 bytes route directly to standard system malloc routines.
- Free list recycling: Singly linked lists maintain vacant blocks within active pools for immediate re-allocation.
