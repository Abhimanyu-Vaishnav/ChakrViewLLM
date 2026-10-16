# Node Maintenance and Drain Protocol

## 1. Drain Sequencing
- Step 1: Mark target compute node unschedulable using `chakr node cordon <node_id>`.
- Step 2: Evict active worker processes to neighboring peer nodes.
- Step 3: Wait for graceful termination timeout of 60 seconds before host shutdown.
