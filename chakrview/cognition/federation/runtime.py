"""
Multi-Node Federation Runtime, Engine Health Tracking & Rejoin Protocol (Step 34).

Coordinates the execution, durability, health monitoring, and recovery of federation engines
without becoming an authority-transfer layer.
- RUNTIME != AUTHORITY
- COORDINATOR != AUTHORITY
- REMOTE_ENGINE != LOCAL_AUTHORITY
- ENGINE_REJOIN != TRUST_GRANT
- UNREACHABLE != REVOKED
"""

from enum import Enum
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.peering.models import AuditEventType
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    HandshakeStatus,
    RevocationTargetType,
    MAX_FEDERATION_ENGINES,
)
from chakrview.cognition.federation.errors import (
    CoordinationCapacityError,
    EngineIdentityError,
)
from chakrview.cognition.federation.persistence.base import SecurityStateStore
from chakrview.cognition.federation.persistence.errors import (
    RuntimeLifecycleError,
    EngineHealthError,
    RejoinProtocolError,
)
from chakrview.cognition.federation.persistence.journal import SecurityStateJournal
from chakrview.cognition.federation.persistence.models import (
    JournalEntryType,
    DurableSecuritySnapshot,
)
from chakrview.cognition.federation.recovery import FederationRecoveryManager

logger = logging.getLogger(__name__)


class EngineRuntimeStatus(str, Enum):
    """Lifecycle states of the local federation runtime node."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    RECOVERING = "RECOVERING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class EngineHealthStatus(str, Enum):
    """Health classification for local and remote federation nodes."""
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNREACHABLE = "UNREACHABLE"
    RECOVERING = "RECOVERING"
    QUARANTINED = "QUARANTINED"
    TERMINATED = "TERMINATED"


class FederationRuntime:
    """
    Bounded operational runtime managing multi-node federation engines, durable security state,
    crash recovery, health tracking, and secure rejoin protocols.
    """

    def __init__(
        self,
        engine: Any,
        store: Optional[SecurityStateStore] = None,
        auto_recover: bool = True,
        membership_manager: Optional[Any] = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.recovery_manager = FederationRecoveryManager(store=self.store) if self.store else None
        self.journal = SecurityStateJournal()
        self.engine.runtime = self

        # Step 35 Node Membership Manager
        if membership_manager:
            self.membership_manager = membership_manager
        elif hasattr(self.engine, "_membership_manager") and self.engine._membership_manager:
            self.membership_manager = self.engine._membership_manager
            self.membership_manager.runtime = self
        else:
            from chakrview.cognition.federation.discovery.membership import FederationMembershipManager
            self.membership_manager = FederationMembershipManager(engine=self.engine, runtime=self)

        if hasattr(self.engine, "membership_manager"):
            self.engine.membership_manager = self.membership_manager

        self._lock = threading.Lock()
        self._status = EngineRuntimeStatus.INITIALIZING
        self._engine_health: Dict[str, EngineHealthStatus] = {}
        self._health_reasons: Dict[str, str] = {}
        self._registered_engines: Dict[str, FederationEngineIdentity] = {}
        self._snapshot_count = 0

        # Mark self as healthy initially
        self_id = self.engine.engine_identity.engine_id
        self._engine_health[self_id] = EngineHealthStatus.HEALTHY

        if auto_recover and self.recovery_manager:
            self.start()
        else:
            self._status = EngineRuntimeStatus.RUNNING

    @property
    def status(self) -> EngineRuntimeStatus:
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._status == EngineRuntimeStatus.RUNNING

    @property
    def dispatcher(self) -> Any:
        return getattr(self.engine, "dispatcher", None)

    @property
    def transport_client(self) -> Any:
        return getattr(self.engine, "transport_client", None)

    @property
    def transport_server(self) -> Any:
        return getattr(self.engine, "transport_server", None)

    @property
    def resource_manager(self) -> Any:
        return getattr(self.engine, "resource_manager", None)

    @property
    def grant_manager(self) -> Any:
        return getattr(self.engine, "grant_manager", None)

    @property
    def task_scheduler(self) -> Any:
        return getattr(self.engine, "task_scheduler", None)

    @property
    def task_executor(self) -> Any:
        return getattr(self.engine, "task_executor", None)

    @property
    def task_coordinator(self) -> Any:
        return getattr(self.engine, "task_coordinator", None)

    @property
    def consensus_engine(self) -> Any:
        return getattr(self.engine, "consensus_engine", None)

    # ========================================================================
    # 1. Runtime Lifecycle (Phase 5)
    # ========================================================================

    def start(self) -> None:
        """Start the federation runtime and recover durable state if configured."""
        with self._lock:
            if self._status == EngineRuntimeStatus.RUNNING:
                return

            self._status = EngineRuntimeStatus.RECOVERING

        if self.recovery_manager:
            manifest = self.recovery_manager.recover(self.engine)
            if not manifest.is_successful():
                with self._lock:
                    self._status = EngineRuntimeStatus.FAILED
                raise RuntimeLifecycleError(
                    f"Runtime failed to start: recovery error: {manifest.failure_reason}"
                )
            self.last_recovery_manifest = manifest
            if self.store:
                entries = self.store.read_journal_entries()
                if entries:
                    try:
                        self.journal.load_entries(entries)
                    except Exception:
                        pass

        with self._lock:
            self._status = EngineRuntimeStatus.RUNNING

    def stop(self) -> None:
        """Cleanly flush journal, write snapshot, and transition to STOPPED."""
        with self._lock:
            if self._status in (EngineRuntimeStatus.STOPPING, EngineRuntimeStatus.STOPPED):
                return
            self._status = EngineRuntimeStatus.STOPPING

        # Write final snapshot if store is present
        if self.store:
            self.take_snapshot()

        with self._lock:
            self._status = EngineRuntimeStatus.STOPPED

    def restart(self) -> None:
        """Cleanly stop runtime and restart with crash recovery."""
        self.stop()
        self.start()

    def take_snapshot(self) -> Optional[DurableSecuritySnapshot]:
        """Create and store an atomic security snapshot."""
        if not self.store:
            return None

        with self._lock:
            self._snapshot_count += 1
            snap_ver = self._snapshot_count
            offset = self.journal.last_sequence

        snap = FederationRecoveryManager.create_snapshot_from_engine(
            engine=self.engine,
            store=self.store,
            snapshot_version=snap_ver,
            journal_offset=offset,
        )
        return snap

    def flush_journal_to_store(self) -> None:
        """Persist unwritten in-memory journal entries to durable store."""
        if not self.store:
            return
        last_seq = self.store.get_last_journal_sequence()
        for entry in self.journal.entries:
            if entry.sequence_num > last_seq:
                self.store.append_journal_entry(entry)

    # ========================================================================
    # 2. Engine Management & Registration
    # ========================================================================

    def register_remote_engine(self, remote_identity: FederationEngineIdentity) -> None:
        """Register a remote coordination engine under capacity bounds."""
        with self._lock:
            if (
                remote_identity.engine_id not in self._registered_engines
                and len(self._registered_engines) >= MAX_FEDERATION_ENGINES
            ):
                raise CoordinationCapacityError(
                    f"Cannot register engine '{remote_identity.engine_id}': capacity ceiling reached."
                )
            self._registered_engines[remote_identity.engine_id] = remote_identity
            if remote_identity.engine_id not in self._engine_health:
                self._engine_health[remote_identity.engine_id] = EngineHealthStatus.UNKNOWN

        # Mirror in coordinator
        if hasattr(self.engine, "coordinator"):
            self.engine.coordinator.register_remote_engine(remote_identity)

    def get_registered_engine(self, engine_id: str) -> Optional[FederationEngineIdentity]:
        with self._lock:
            return self._registered_engines.get(engine_id)

    # ========================================================================
    # 3. Failure Detection & Health Tracking (Phase 6)
    # ========================================================================

    def get_engine_health(self, engine_id: str) -> EngineHealthStatus:
        with self._lock:
            return self._engine_health.get(engine_id, EngineHealthStatus.UNKNOWN)

    def set_engine_health(
        self,
        engine_id: str,
        status: EngineHealthStatus,
        reason: Optional[str] = None,
    ) -> None:
        """
        Update the health status of an engine.
        CRITICAL: UNREACHABLE != REVOKED. Network outages do not revoke peer trust.
        """
        with self._lock:
            prev = self._engine_health.get(engine_id, EngineHealthStatus.UNKNOWN)
            self._engine_health[engine_id] = status
            if reason:
                self._health_reasons[engine_id] = reason

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.ENGINE_HEALTH_CHANGED,
                epoch=self.engine.current_epoch,
                details={
                    "engine_id": engine_id,
                    "previous_status": prev.value,
                    "new_status": status.value,
                    "reason": reason or "Status updated",
                },
            )

        if status == EngineHealthStatus.QUARANTINED and audit:
            audit.log(
                event_type=AuditEventType.ENGINE_QUARANTINED,
                epoch=self.engine.current_epoch,
                details={"engine_id": engine_id, "reason": reason or "Quarantined"},
            )

    def record_heartbeat(
        self,
        engine_id: str,
        is_healthy: bool = True,
        reason: Optional[str] = None,
    ) -> None:
        """Record heartbeat observation for an engine, updating health status."""
        status = EngineHealthStatus.HEALTHY if is_healthy else EngineHealthStatus.UNREACHABLE
        self.set_engine_health(
            engine_id,
            status,
            reason=reason or ("Heartbeat received" if is_healthy else "Heartbeat missed"),
        )

    # ========================================================================
    # 4. Rejoin Protocol (Phase 7)
    # ========================================================================

    def execute_rejoin(
        self,
        remote_engine: Any,
    ) -> Dict[str, Any]:
        """
        Execute secure rejoin flow after engine restart or partition healing.
        Enforces:
        - ENGINE_REJOIN != TRUST_GRANT
        - ENGINE_REJOIN != CAPABILITY_ESCALATION
        - LOCAL_AUTHORITY > REMOTE_AUTHORITY
        """
        audit = getattr(self.engine, "audit_logger", None)
        remote_id = remote_engine.engine_identity.engine_id

        if audit:
            audit.log(
                event_type=AuditEventType.ENGINE_REJOIN_STARTED,
                epoch=self.engine.current_epoch,
                details={"remote_engine_id": remote_id},
            )

        try:
            # Step 1: Perform coordination handshake
            response = self.engine.initiate_coordination_handshake(remote_engine)
            if not response.is_accepted():
                self.set_engine_health(
                    remote_id,
                    EngineHealthStatus.DEGRADED,
                    reason=f"Handshake rejected: {response.details.get('error')}",
                )
                if audit:
                    audit.log(
                        event_type=AuditEventType.ENGINE_REJOIN_FAILED,
                        epoch=self.engine.current_epoch,
                        details={"remote_engine_id": remote_id, "reason": "Handshake rejected"},
                    )
                raise RejoinProtocolError(f"Rejoin handshake failed: {response.details.get('error')}")

            # Step 2: Synchronize advisory replay state if shared sessions exist
            for sid in list(self.engine.sessions.keys()):
                if sid in remote_engine.sessions:
                    self.engine.synchronize_replay_with_engine(remote_engine, sid)

            # Step 3: Synchronize trust claims under ceiling verification
            self.engine.synchronize_trust_with_engine(remote_engine)

            # Step 4: Re-validate local revocations (Local Revocation Always Wins)
            if hasattr(self.engine, "registry"):
                for reg in self.engine.registry.list_peers():
                    if reg.discovery_status.value == "REVOKED":
                        # Ensure remote also observes local revocation
                        self.engine.propagate_revocation_to_engines(
                            target_type=RevocationTargetType.PEER,
                            target_id=reg.identity.peer_id,
                            reason="Rejoin reconciliation",
                            remote_engines=[remote_engine],
                        )

            # Step 5: Mark engine healthy
            self.set_engine_health(
                remote_id,
                EngineHealthStatus.HEALTHY,
                reason="Rejoin protocol completed successfully",
            )

            if audit:
                audit.log(
                    event_type=AuditEventType.ENGINE_REJOIN_COMPLETED,
                    epoch=self.engine.current_epoch,
                    details={"remote_engine_id": remote_id},
                )

            return {
                "status": "REJOIN_SUCCESSFUL",
                "remote_engine_id": remote_id,
                "health": EngineHealthStatus.HEALTHY.value,
            }

        except Exception as e:
            self.set_engine_health(
                remote_id,
                EngineHealthStatus.DEGRADED,
                reason=f"Rejoin error: {e}",
            )
            if audit:
                audit.log(
                    event_type=AuditEventType.ENGINE_REJOIN_FAILED,
                    epoch=self.engine.current_epoch,
                    details={"remote_engine_id": remote_id, "error": str(e)},
                )
            raise RejoinProtocolError(f"Engine rejoin failed: {e}") from e
