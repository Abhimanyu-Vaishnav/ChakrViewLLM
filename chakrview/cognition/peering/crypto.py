"""
Cryptographic Peer Identity and Ed25519 Signing for Cross-Zone Federation (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. CRYPTOGRAPHIC_IDENTITY != AUTHORITY:
   A valid cryptographic identity proves key possession, NOT capability authority.
2. AUTHENTICATION != AUTHORIZATION & AUTHENTICATION != TRUST:
   Proving identity does not confer trust or authorization. All requests must
   still pass TrustModel, FederationPolicy, and CapabilityGate.
3. ZERO PRIVATE KEY LEAKAGE:
   Private keys must never be exposed via __repr__, __str__, dictionary exports,
   audit logs, telemetry traces, or wire messages.
4. NO CUSTOM CIPHERS / NO FAKE CRYPTOGRAPHY:
   Uses standard Ed25519 (RFC 8032) via the established cryptography library.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


class KeyLifecycleState(str, Enum):
    """Lifecycle state of an asymmetric cryptographic key."""
    ACTIVE = "ACTIVE"
    ROTATING = "ROTATING"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class CryptoError(Exception):
    """Base exception for cryptographic operations."""
    pass


class SignatureVerificationError(CryptoError):
    """Raised when an Ed25519 digital signature fails mathematical verification."""
    pass


class KeyStateError(CryptoError):
    """Raised when an operation is attempted with a revoked or expired key."""
    pass


@dataclass(frozen=True)
class Ed25519PublicKeyWrapper:
    """
    Immutable wrapper around an Ed25519 public key.
    Safe for logging, serialization, and telemetry.
    """
    public_bytes: bytes
    public_hex: str
    fingerprint: str  # SHA-256 hex digest of public_bytes

    @classmethod
    def from_raw_bytes(cls, raw: bytes) -> "Ed25519PublicKeyWrapper":
        if len(raw) != 32:
            raise ValueError(f"Ed25519 public key must be exactly 32 bytes, got {len(raw)}.")
        pub_hex = raw.hex()
        fingerprint = hashlib.sha256(raw).hexdigest()
        return cls(public_bytes=raw, public_hex=pub_hex, fingerprint=fingerprint)

    @classmethod
    def from_hex(cls, hex_str: str) -> "Ed25519PublicKeyWrapper":
        raw = bytes.fromhex(hex_str.strip())
        return cls.from_raw_bytes(raw)

    def verify(self, signature: bytes, message: bytes) -> bool:
        """
        Verify a 64-byte Ed25519 signature over message bytes.
        Returns True if valid, False if signature is mathematically invalid.
        """
        if len(signature) != 64:
            return False
        try:
            ed_pub = ed25519.Ed25519PublicKey.from_public_bytes(self.public_bytes)
            ed_pub.verify(signature, message)
            return True
        except (InvalidSignature, Exception):
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "public_hex": self.public_hex,
            "fingerprint": self.fingerprint,
            "key_type": "Ed25519",
        }


class Ed25519PrivateKeyWrapper:
    """
    Secure wrapper around an Ed25519 private key.
    Strictly safeguards private material against accidental exposure.
    """

    def __init__(self, private_key: ed25519.Ed25519PrivateKey) -> None:
        self._private_key = private_key
        # Extract public key for fast access
        pub_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self._public_wrapper = Ed25519PublicKeyWrapper.from_raw_bytes(pub_bytes)

    @classmethod
    def generate(cls) -> "Ed25519PrivateKeyWrapper":
        """Generate a cryptographically secure fresh Ed25519 keypair."""
        priv = ed25519.Ed25519PrivateKey.generate()
        return cls(priv)

    @classmethod
    def from_seed(cls, seed: bytes) -> "Ed25519PrivateKeyWrapper":
        """Instantiate private key from 32-byte deterministic seed."""
        if len(seed) != 32:
            raise ValueError(f"Ed25519 seed must be exactly 32 bytes, got {len(seed)}.")
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        return cls(priv)

    def public_key(self) -> Ed25519PublicKeyWrapper:
        """Return the corresponding public key wrapper."""
        return self._public_wrapper

    def sign(self, message: bytes) -> bytes:
        """
        Compute detached 64-byte Ed25519 signature over arbitrary message bytes.
        """
        return self._private_key.sign(message)

    def sign_hex(self, message: bytes) -> str:
        """Compute detached signature and return as hexadecimal string."""
        return self.sign(message).hex()

    def __repr__(self) -> str:
        # Zero private key leakage
        return f"<Ed25519PrivateKeyWrapper fingerprint={self._public_wrapper.fingerprint[:12]}... [PRIVATE KEY REDACTED]>"

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> Dict[str, Any]:
        """Strictly prohibit private key dictionary export."""
        raise PermissionError(
            "Architectural violation: Private keys must never be serialized or exported to dict."
        )


@dataclass
class KeyRevocationRecord:
    """Formal audit record of cryptographic key revocation."""
    key_fingerprint: str
    revoked_epoch: int
    reason: str
    revoked_by: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_fingerprint": self.key_fingerprint,
            "revoked_epoch": self.revoked_epoch,
            "reason": self.reason,
            "revoked_by": self.revoked_by,
            "timestamp": self.timestamp,
        }


@dataclass
class CryptographicPeerIdentity:
    """
    Strongly typed cryptographic peer identity binding public key material,
    zone domain, and deterministic peer identifier.
    """
    peer_id: str
    zone_id: str
    organization_id: str
    public_key: Ed25519PublicKeyWrapper
    key_state: KeyLifecycleState = KeyLifecycleState.ACTIVE
    created_epoch: int = 1
    expires_epoch: int = 100
    protocol_version: str = "30.0"
    architecture_version: str = "0.1"
    revocation_record: Optional[KeyRevocationRecord] = None
    rotation_history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        zone_id: str,
        organization_id: str,
        public_key: Ed25519PublicKeyWrapper,
        peer_id: Optional[str] = None,
        created_epoch: int = 1,
        ttl_epochs: int = 100,
        protocol_version: str = "30.0",
        architecture_version: str = "0.1",
    ) -> "CryptographicPeerIdentity":
        """
        Factory deriving a deterministic peer identifier from public key fingerprint.
        """
        # If peer_id is not specified, derive deterministically from public key fingerprint
        derived_peer_id = peer_id or f"peer_{public_key.fingerprint[:16]}"
        return cls(
            peer_id=derived_peer_id,
            zone_id=zone_id,
            organization_id=organization_id,
            public_key=public_key,
            key_state=KeyLifecycleState.ACTIVE,
            created_epoch=created_epoch,
            expires_epoch=created_epoch + ttl_epochs,
            protocol_version=protocol_version,
            architecture_version=architecture_version,
        )

    def is_valid(self, current_epoch: int) -> Tuple[bool, str]:
        """Verify whether cryptographic identity is currently active and unexpired."""
        if self.key_state == KeyLifecycleState.REVOKED:
            reason = self.revocation_record.reason if self.revocation_record else "Key revoked"
            return False, f"Cryptographic identity is REVOKED: {reason}"
        if self.key_state == KeyLifecycleState.EXPIRED or current_epoch > self.expires_epoch:
            return False, f"Cryptographic identity is EXPIRED at epoch {self.expires_epoch} (current {current_epoch})"
        return True, "Key is active and valid"

    def revoke(self, reason: str, revoked_epoch: int, revoked_by: str = "local_authority") -> KeyRevocationRecord:
        """Revoke key state immediately."""
        rec = KeyRevocationRecord(
            key_fingerprint=self.public_key.fingerprint,
            revoked_epoch=revoked_epoch,
            reason=reason,
            revoked_by=revoked_by,
        )
        self.key_state = KeyLifecycleState.REVOKED
        self.revocation_record = rec
        return rec

    def rotate_key(
        self,
        new_public_key: Ed25519PublicKeyWrapper,
        rotation_epoch: int,
        signature_from_old_key: str,
    ) -> None:
        """
        Record key rotation metadata.
        """
        # Verify signature from old key authorizing rotation to new key
        proof_payload = f"ROTATE_KEY:{self.public_key.fingerprint}:{new_public_key.fingerprint}:{rotation_epoch}".encode("utf-8")
        if not self.public_key.verify(bytes.fromhex(signature_from_old_key), proof_payload):
            raise SignatureVerificationError("Key rotation proof signature is invalid.")

        self.rotation_history.append({
            "previous_key_fingerprint": self.public_key.fingerprint,
            "new_key_fingerprint": new_public_key.fingerprint,
            "rotation_epoch": rotation_epoch,
            "proof_signature": signature_from_old_key,
        })
        self.public_key = new_public_key
        self.key_state = KeyLifecycleState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "organization_id": self.organization_id,
            "public_key": self.public_key.to_dict(),
            "key_state": self.key_state.value,
            "created_epoch": self.created_epoch,
            "expires_epoch": self.expires_epoch,
            "protocol_version": self.protocol_version,
            "architecture_version": self.architecture_version,
            "has_revocation": self.revocation_record is not None,
            "rotation_count": len(self.rotation_history),
        }
