# In-Transit Cryptographic Protocols

## 1. Wire Encryption Standards
- In-transit communication is strictly encrypted via TLS 1.3 with AES-256-GCM.
- Session re-keying threshold: Every 24 hours or after transmitting 100 GB of traffic.
- Certificate authority verification: Mutual TLS (mTLS) with custom X.509 certificates.
