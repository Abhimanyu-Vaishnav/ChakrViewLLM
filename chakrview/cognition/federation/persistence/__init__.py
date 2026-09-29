"""
Persistence & Write-Ahead Journaling Subsystem for ChakrView Federation (Step 34).
"""

from chakrview.cognition.federation.persistence.errors import (
    PersistenceError,
    DurableSchemaError,
    JournalError,
    JournalCorruptionError,
    JournalSequenceError,
    JournalTruncationError,
    JournalReplayError,
    SnapshotError,
    SnapshotCorruptionError,
    SnapshotVersionError,
    RecoveryError,
    RecoveryFailedClosedError,
    RuntimeLifecycleError,
    EngineHealthError,
    RejoinProtocolError,
)
from chakrview.cognition.federation.persistence.models import (
    DURABLE_SCHEMA_VERSION,
    JOURNAL_GENESIS_DIGEST,
    GENESIS_JOURNAL_DIGEST,
    MAX_JOURNAL_PAYLOAD_BYTES,
    MAX_SNAPSHOT_BYTES,
    JournalEntryType,
    JournalEntry,
    DurableSecuritySnapshot,
    RecoveryManifest,
)
from chakrview.cognition.federation.persistence.base import SecurityStateStore
from chakrview.cognition.federation.persistence.memory import InMemorySecurityStateStore
from chakrview.cognition.federation.persistence.sqlite import SqliteSecurityStateStore
from chakrview.cognition.federation.persistence.journal import SecurityStateJournal

__all__ = [
    "PersistenceError",
    "DurableSchemaError",
    "JournalError",
    "JournalCorruptionError",
    "JournalSequenceError",
    "JournalTruncationError",
    "JournalReplayError",
    "SnapshotError",
    "SnapshotCorruptionError",
    "SnapshotVersionError",
    "RecoveryError",
    "RecoveryFailedClosedError",
    "RuntimeLifecycleError",
    "EngineHealthError",
    "RejoinProtocolError",
    "DURABLE_SCHEMA_VERSION",
    "JOURNAL_GENESIS_DIGEST",
    "GENESIS_JOURNAL_DIGEST",
    "MAX_JOURNAL_PAYLOAD_BYTES",
    "MAX_SNAPSHOT_BYTES",
    "JournalEntryType",
    "JournalEntry",
    "DurableSecuritySnapshot",
    "RecoveryManifest",
    "SecurityStateStore",
    "InMemorySecurityStateStore",
    "SqliteSecurityStateStore",
    "SecurityStateJournal",
]
