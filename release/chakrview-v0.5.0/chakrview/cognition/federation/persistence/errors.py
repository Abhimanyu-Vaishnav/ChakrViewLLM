"""
Strongly typed error taxonomy for Durable Security State & Failure Recovery (Step 34).

Enforces fail-closed semantics across all persistence, journal verification,
and crash recovery operations.
"""


class PersistenceError(Exception):
    """Base exception for all federation persistence failures."""
    pass


class DurableSchemaError(PersistenceError):
    """Raised when persisted security state fails schema validation or version checks."""
    pass


class JournalError(PersistenceError):
    """Base exception for journal operations."""
    pass


class JournalCorruptionError(JournalError):
    """Raised when a journal entry digest mismatches or previous-hash chain is broken."""
    pass


class JournalSequenceError(JournalError):
    """Raised when a non-monotonic or regressive sequence number is encountered."""
    pass


class JournalTruncationError(JournalError):
    """Raised when an incomplete or truncated journal record is detected."""
    pass


class JournalReplayError(JournalError):
    """Raised when replaying a journal entry fails or violates security invariants."""
    pass


class SnapshotError(PersistenceError):
    """Base exception for snapshot operations."""
    pass


class SnapshotCorruptionError(SnapshotError):
    """Raised when snapshot payload fails SHA-256 integrity verification."""
    pass


class SnapshotVersionError(SnapshotError):
    """Raised when snapshot version is incompatible or regressive."""
    pass


class RecoveryError(PersistenceError):
    """Base exception for failure recovery operations."""
    pass


class RecoveryFailedClosedError(RecoveryError):
    """
    CRITICAL: Raised when recovery detects corruption, divergence, or tampering.
    Unsafe security state must NEVER become active.
    """
    pass


class RuntimeLifecycleError(Exception):
    """Raised when federation runtime transitions violate valid states."""
    pass


class EngineHealthError(Exception):
    """Raised when engine health transitions or checks fail."""
    pass


class RejoinProtocolError(Exception):
    """Raised when engine rejoin fails authentication or consistency verification."""
    pass
