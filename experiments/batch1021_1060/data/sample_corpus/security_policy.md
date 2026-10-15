# Operational Security & Sandbox Policy

## 1. Subprocess Isolation
All tool executions, terminal commands, and python scratch evaluations run inside an isolated OS subprocess sandbox.
Direct execution of arbitrary root or administrative commands is permanently forbidden.

## 2. Authentication Requirements
Production databases require Multi-Factor Authentication (MFA) and cryptographically signed OAuth2 tokens.
Tokens expire after exactly 3600 seconds (1 hour). Unauthenticated requests are immediately terminated with status code 401.

## 3. Data Protection and Encryption
- In-transit communication is strictly encrypted via TLS 1.3 with AES-256-GCM.
- At-rest persistence uses transparent disk encryption with rotating HMAC keys.
- User session histories stored on disk are restricted to permission mode 0600.
