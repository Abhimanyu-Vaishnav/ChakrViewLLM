# Asyncio Event Loop and Coroutine Scheduling

## 1. Loop Architecture
- I/O multiplexing engine: Epoll and Kqueue selector backends poll ready file descriptors with zero CPU idle consumption.
- Task wrapper abstractions: Coroutine frames suspended via yield expressions wrapped inside Task task futures.
- Callback execution order: Ready callbacks execute sequentially before timers and pending network socket signals.
- Custom loop exception handlers: Unhandled coroutine exceptions routed through configurable global logging callbacks.
- ThreadPoolExecutor offloading: Blocking CPU-bound routines dispatched onto background thread pools via run_in_executor.
