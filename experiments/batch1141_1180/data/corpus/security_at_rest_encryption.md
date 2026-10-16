# At-Rest Data Encryption Standards

## 1. Encryption Methods
- Disk block encryption: Transparent AES-256-XTS filesystem encryption.
- Key rotation cycle: Rotating HMAC keys renewed every 90 days.
- File permission restriction: Mode 0600 on all stored session histories.
