"""
Production Federation Message Transport & Secure Inter-Node Communication (Step 36).

Public API exposing:
- FederationChannel: Secure, authenticated, replay-protected inter-node channel.
- FederationMessageFramer: Length-prefixed binary wire framing with ceiling bounds.
- FederationMessageCodec: Deterministic canonical envelope serializer/deserializer.
- FederationMessageDispatcher: Sovereign message dispatcher enforcing CapabilityGate and trust scopes.
- FederationTransportClient: Outbound channel client with bounded exponential backoff reconnection.
- FederationTransportServer: Inbound channel server under strict default-deny admission.
- Strongly typed message envelopes, channel states, and exception hierarchy.
"""

from chakrview.cognition.federation.transport.errors import (
    FederationTransportError,
    FramingError,
    OversizedFrameError,
    MalformedFrameError,
    TruncatedFrameError,
    CodecError,
    ProhibitedPayloadError,
    UnknownMessageTypeError,
    EnvelopeIntegrityError,
    ChannelError,
    ChannelStateError,
    ChannelAuthenticationError,
    ChannelClosedError,
    ChannelTimeoutError,
    ChannelQuarantinedError,
    ChannelRevokedError,
    DispatcherError,
    HandlerNotFoundError,
    HandlerExecutionError,
    HandlerTimeoutError,
    DispatcherQueueFullError,
    UnauthorizedMessageError,
    ReplayError,
    SequenceRegressionError,
    DuplicateMessageError,
    ReconnectError,
    MaxReconnectAttemptsExceededError,
)

from chakrview.cognition.federation.transport.models import (
    FEDERATION_PROTOCOL_VERSION,
    DEFAULT_MAX_FRAME_SIZE,
    DEFAULT_MAX_PAYLOAD_SIZE,
    DEFAULT_CHANNEL_TIMEOUT_SECONDS,
    DEFAULT_MAX_DISPATCHER_QUEUE_SIZE,
    ChannelState,
    VALID_CHANNEL_TRANSITIONS,
    FederationMessageType,
    ChannelMetrics,
    ReconnectPolicy,
    FederationMessageEnvelope,
)

from chakrview.cognition.federation.transport.framing import (
    FederationMessageFramer,
)

from chakrview.cognition.federation.transport.codec import (
    FederationMessageCodec,
)

from chakrview.cognition.federation.transport.channel import (
    FederationChannel,
)

from chakrview.cognition.federation.transport.dispatcher import (
    FederationMessageDispatcher,
    HandlerRegistration,
)

from chakrview.cognition.federation.transport.client import (
    FederationTransportClient,
)

from chakrview.cognition.federation.transport.server import (
    FederationTransportServer,
)

__all__ = [
    # Protocol constants
    "FEDERATION_PROTOCOL_VERSION",
    "DEFAULT_MAX_FRAME_SIZE",
    "DEFAULT_MAX_PAYLOAD_SIZE",
    "DEFAULT_CHANNEL_TIMEOUT_SECONDS",
    "DEFAULT_MAX_DISPATCHER_QUEUE_SIZE",
    # Enums & Models
    "ChannelState",
    "VALID_CHANNEL_TRANSITIONS",
    "FederationMessageType",
    "ChannelMetrics",
    "ReconnectPolicy",
    "FederationMessageEnvelope",
    # Core Components
    "FederationMessageFramer",
    "FederationMessageCodec",
    "FederationChannel",
    "FederationMessageDispatcher",
    "HandlerRegistration",
    "FederationTransportClient",
    "FederationTransportServer",
    # Errors
    "FederationTransportError",
    "FramingError",
    "OversizedFrameError",
    "MalformedFrameError",
    "TruncatedFrameError",
    "CodecError",
    "ProhibitedPayloadError",
    "UnknownMessageTypeError",
    "EnvelopeIntegrityError",
    "ChannelError",
    "ChannelStateError",
    "ChannelAuthenticationError",
    "ChannelClosedError",
    "ChannelTimeoutError",
    "ChannelQuarantinedError",
    "ChannelRevokedError",
    "DispatcherError",
    "HandlerNotFoundError",
    "HandlerExecutionError",
    "HandlerTimeoutError",
    "DispatcherQueueFullError",
    "UnauthorizedMessageError",
    "ReplayError",
    "SequenceRegressionError",
    "DuplicateMessageError",
    "ReconnectError",
    "MaxReconnectAttemptsExceededError",
]
