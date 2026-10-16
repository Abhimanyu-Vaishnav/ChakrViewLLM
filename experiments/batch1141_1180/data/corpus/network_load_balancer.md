# Traffic Routing and Reverse Proxy

## 1. Routing Algorithms
- Primary load-balancing algorithm: Weighted round-robin based on CPU capacity.
- Health probe endpoint: `/healthz` HTTP GET probe.
- Health probe failure threshold: 3 consecutive failed probes.
