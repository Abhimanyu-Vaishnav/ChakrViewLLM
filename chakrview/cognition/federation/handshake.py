"""
Federation State Handshake Protocol (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. FEDERATION_HANDSHAKE != TRUST_GRANT:
   Completing a handshake establishes coordination compatibility; it grants
   ZERO trust, capability permissions, or cognitive task delegation rights.
2. FEDERATION_HANDSHAKE != AUTHORIZATION:
   A successful handshake allows security state exchange (replays, revocations);
   it never authorizes execution.
3. FAIL-CLOSED INCOMPATIBILITY:
   Protocol mismatches, forged engine identities, or same-version digest conflicts
   unconditionally reject the handshake.
"""

from typing import Dict, List, Optional, Tuple, Any
import secrets

from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    FederationHandshakeRequest,
    FederationHandshakeResponse,
    HandshakeStatus,
    FEDERATION_PROTOCOL_VERSION,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.state import FederationStateManager
from chakrview.cognition.federation.errors import (
    HandshakeError,
    ProtocolMismatchError,
    EngineIdentityError,
    StateDigestConflictError,
)
from chakrview.cognition.peering.session import SecurePeerSession
from chakrview.cognition.peering.registry import PeerRegistry
from chakrview.cognition.peering.models import RevocationRecord, AuditEventType


class FederationHandshakeManager:
    """
    Manages bounded coordination handshakes between distributed federation engines.
    """

    def __init__(self, local_identity: FederationEngineIdentity, state_manager: FederationStateManager) -> None:
        self.local_identity = local_identity
        self.state_manager = state_manager

    def create_handshake_request(
        self,
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocations: Optional[List[RevocationRecord]] = None,
    ) -> FederationHandshakeRequest:
        """
        Construct an outbound coordination handshake request.
        """
        composite_state = self.state_manager.compute_state_digest(
            session=session,
            registry=registry,
            revocation_records=revocations,
        )

        return FederationHandshakeRequest(
            sender_identity=self.local_identity,
            protocol_version=self.local_identity.protocol_version,
            epoch=self.state_manager.current_epoch,
            state_version=self.state_manager.current_version.version,
            composite_digest=composite_state.compute_composite_digest(),
            replay_digest=composite_state.replay_digest,
            trust_digest=composite_state.trust_digest,
            revocation_digest=composite_state.revocation_digest,
            nonce=secrets.token_hex(16),
        )

    def process_handshake_request(
        self,
        request: FederationHandshakeRequest,
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocations: Optional[List[RevocationRecord]] = None,
    ) -> FederationHandshakeResponse:
        """
        Validate and respond to an inbound federation handshake request.
        """
        # 1. Validate remote engine identity structure & fingerprint
        is_valid_id, reason = FederationEngineIdentityProvider.validate_identity(request.sender_identity)
        if not is_valid_id:
            return FederationHandshakeResponse(
                responder_identity=self.local_identity,
                status=HandshakeStatus.REJECTED,
                protocol_version=self.local_identity.protocol_version,
                epoch=self.state_manager.current_epoch,
                state_version=self.state_manager.current_version.version,
                sync_required=False,
                details={"error": f"Invalid sender identity: {reason}"},
                nonce=request.nonce,
            )

        # 2. Protocol version compatibility check
        if request.protocol_version != self.local_identity.protocol_version:
            return FederationHandshakeResponse(
                responder_identity=self.local_identity,
                status=HandshakeStatus.INCOMPATIBLE_PROTOCOL,
                protocol_version=self.local_identity.protocol_version,
                epoch=self.state_manager.current_epoch,
                state_version=self.state_manager.current_version.version,
                sync_required=False,
                details={
                    "error": (
                        f"Protocol mismatch: expected '{self.local_identity.protocol_version}', "
                        f"got '{request.protocol_version}'"
                    )
                },
                nonce=request.nonce,
            )

        # 3. Compute local composite digest for comparison
        local_composite = self.state_manager.compute_state_digest(
            session=session,
            registry=registry,
            revocation_records=revocations,
        )
        local_digest_str = local_composite.compute_composite_digest()

        # 4. Version & digest comparison
        local_ver = self.state_manager.current_version.version
        remote_ver = request.state_version

        # Case C: Same engine identity with divergent digest (split-brain/impersonation conflict)
        if request.sender_identity.engine_id == self.local_identity.engine_id and request.composite_digest != local_digest_str:
            return FederationHandshakeResponse(
                responder_identity=self.local_identity,
                status=HandshakeStatus.DIGEST_CONFLICT,
                protocol_version=self.local_identity.protocol_version,
                epoch=self.state_manager.current_epoch,
                state_version=local_ver,
                sync_required=True,
                details={
                    "error": "Split-brain or identity collision detected: same engine ID with divergent digest.",
                    "local_digest": local_digest_str,
                    "remote_digest": request.composite_digest,
                },
                nonce=request.nonce,
            )

        # Determine if synchronization is required
        sync_required = (remote_ver != local_ver) or (request.composite_digest != local_digest_str)

        return FederationHandshakeResponse(
            responder_identity=self.local_identity,
            status=HandshakeStatus.ACCEPTED,
            protocol_version=self.local_identity.protocol_version,
            epoch=self.state_manager.current_epoch,
            state_version=local_ver,
            sync_required=sync_required,
            details={
                "remote_version": remote_ver,
                "local_version": local_ver,
                "digests_match": request.composite_digest == local_digest_str,
            },
            nonce=request.nonce,
        )
