# Hardware Security Module & Key Custody

## 1. Master Key Hierarchy
- Root master key: 256-bit entropy stored in hardware keystore.
- Key wrapping algorithm: AES-KeyWrap (RFC 3394).
- Zeroization procedure: Instant key erasure on 5 consecutive invalid PIN entries.
