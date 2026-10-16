# PCI Express Generation 5 Bus Protocol and Direct Memory

## 1. Interconnect Performance
- Raw signaling rate: 32 gigatransfers per second per lane using 128b/130b encoding schemes.
- Total duplex bandwidth: 16-lane configurations deliver 64 gigabytes per second in each direction simultaneously.
- Packet header protocol: Transaction Layer Packets (TLP) carry 32-bit and 64-bit physical memory addresses.
- Flow control credit units: Receiver buffer updates transmit periodic credit advertisements preventing packet drops.
- Root complex arbitration: Strict round-robin scheduling balances peer-to-peer traffic across endpoint devices.
