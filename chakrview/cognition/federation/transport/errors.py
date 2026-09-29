"""
Exception Hierarchy for Production Federation Message Transport (Step 36).

All transport, framing, serialization, channel, and dispatch errors
derive from FederationTransportError to enable consistent fail-closed behavior.
"""


class FederationTransportError(Exception):
    """Base exception for all federation transport operations."""
    pass


# ============================================================================
# 1. Framing Errors
# ============================================================================

class FramingError(FederationTransportError):
    """Base error for length-prefixed wire framing failures."""
    pass


class OversizedFrameError(FramingError):
    """Raised when a frame header or payload exceeds the maximum allowed ceiling."""
    pass


class MalformedFrameError(FramingError):
    """Raised when frame headers or length prefixes are corrupt or invalid."""
    pass


class TruncatedFrameError(FramingError):
    """Raised when a frame is unexpectedly truncated or stream closes mid-frame."""
    pass


# ============================================================================
# 2. Codec & Serialization Errors
# ============================================================================

class CodecError(FederationTransportError):
    """Base error for message serialization and deserialization."""
    pass


class ProhibitedPayloadError(CodecError):
    """Raised when serialization of forbidden data (keys, secrets, weights, code) is attempted."""
    pass


class UnknownMessageTypeError(CodecError):
    """Raised when an un-recognized or unpermitted message type is encountered."""
    pass


class EnvelopeIntegrityError(CodecError):
    """Raised when envelope payload digest or cryptographic signature fails validation."""
    pass


# ============================================================================
# 3. Channel & Connection Lifecycle Errors
# ============================================================================

class ChannelError(FederationTransportError):
    """Base error for federation channel lifecycle operations."""
    pass


class ChannelStateError(ChannelError):
    """Raised when an invalid channel state transition or operation in current state is attempted."""
    pass


class ChannelAuthenticationError(ChannelError):
    """Raised when transport authentication (mTLS or certificate binding) fails."""
    pass


class ChannelClosedError(ChannelError):
    """Raised when attempting transmission over a closed or disconnecting channel."""
    pass


class ChannelTimeoutError(ChannelError):
    """Raised when a channel send, receive, or handshake exceeds allotted timeout."""
    pass


class ChannelQuarantinedError(ChannelError):
    """Raised when message operations are blocked due to channel / node quarantine."""
    pass


class ChannelRevokedError(ChannelError):
    """Raised when operations are blocked due to permanent terminal revocation."""
    pass


# ============================================================================
# 4. Dispatcher Errors
# ============================================================================

class DispatcherError(FederationTransportError):
    """Base error for message dispatching and handling."""
    pass


class HandlerNotFoundError(DispatcherError):
    """Raised when no registered handler exists for an incoming message type."""
    pass


class HandlerExecutionError(DispatcherError):
    """Raised when a message handler execution raises an unhandled error."""
    pass


class HandlerTimeoutError(DispatcherError):
    """Raised when message handling exceeds allotted processing deadline."""
    pass


class DispatcherQueueFullError(DispatcherError):
    """Raised when the dispatcher message queue exceeds bounded capacity."""
    pass


class UnauthorizedMessageError(DispatcherError):
    """Raised when an incoming message fails sovereign local authorization checks."""
    pass


# ============================================================================
# 5. Replay & Sequence Errors
# ============================================================================

class ReplayError(FederationTransportError):
    """Base error for replay attack and monotonic sequence violations."""
    pass


class SequenceRegressionError(ReplayError):
    """Raised when a message sequence number is less than or equal to local replay floor."""
    pass


class DuplicateMessageError(ReplayError):
    """Raised when a previously seen message ID is re-submitted."""
    pass


# ============================================================================
# 6. Reconnect Errors
# ============================================================================

class ReconnectError(FederationTransportError):
    """Base error for reconnection failures."""
    pass


class MaxReconnectAttemptsExceededError(ReconnectError):
    """Raised when exponential backoff retries exceed the maximum configured attempts."""
    pass
