# Shared Memory and Inter-Process Communication

## 1. Process Coordination
- Posix shared memory segments: Named memory regions mapped into distinct process address spaces via shm_open calls.
- Raw byte buffers: SharedMemory abstractions wrap memory-mapped files supporting cross-process NumPy array buffers.
- Synchronization primitives: Inter-process Semaphores and Locks backed by operating system futex implementations.
- Zero-copy deserialization: Reading shared memory buffers avoids pickling overhead during high-volume data exchange.
- Process pool management: Forkserver and spawn start methods prevent resource leaks and thread deadlocks during initialization.
