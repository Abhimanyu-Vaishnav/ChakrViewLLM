"""
Transport and Wire Protocol Exceptions (Step 30).
"""


class TransportError(Exception):
    """Base exception for all physical and logical transport failures."""
    pass


class TransportTimeoutError(TransportError):
    """Raised when transport connection or I/O operation exceeds configured timeout."""
    pass


class TransportUnavailableError(TransportError):
    """Raised when a physical transport driver or endpoint is unavailable."""
    pass


class TransportProtocolError(TransportError):
    """Raised when incoming wire frames violate protocol envelope specifications."""
    pass


class FrameError(TransportProtocolError):
    """Raised when length-prefixed framing is corrupt or truncated."""
    pass


class OversizedPayloadError(TransportProtocolError):
    """Raised when a message envelope or payload exceeds maximum allowed byte limit."""
    pass


class ReplayAttackError(TransportProtocolError):
    """Raised when a replayed message or nonce is detected on the wire."""
    pass


class WireSecurityError(TransportError):
    """Raised when cryptographic verification or session authentication fails on wire."""
    pass
