# Authentication and Token Lifecycle

## 1. Token Policy
- Token format: Cryptographically signed OAuth2 JSON Web Tokens (JWT).
- Token expiration window: Exactly 3600 seconds (60 minutes).
- Unauthenticated API access: Terminated immediately with HTTP 401 Unauthorized.
