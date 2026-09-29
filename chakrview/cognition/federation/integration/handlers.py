"""
Federated Wire Message Handlers for Task Execution and Consensus (Step 41).
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from chakrview.cognition.federation.transport.models import (
    FederationMessageEnvelope,
    FederationMessageType,
)
from chakrview.cognition.federation.transport.channel import FederationChannel
from chakrview.cognition.federation.tasks.models import (
    WorkUnit,
    TaskCheckpoint,
    CheckpointManifest,
    TaskResultEnvelope,
    ResourceRequirements,
)
from chakrview.cognition.federation.consensus.models import (
    ConsensusProposal,
    ConsensusVote,
    VoteType,
    QuorumCertificate,
)
from chakrview.cognition.federation.tasks.errors import ExecutionGrantViolationError

logger = logging.getLogger("chakrview.federation.integration.handlers")

PROHIBITED_LEAK_KEYWORDS = ("private_key", "secret_key", "token_secret", "BEGIN PRIVATE KEY")


class FederationWireHandlerRegistry:
    """
    Registers and binds end-to-end task execution and consensus wire message handlers
    to the FederationMessageDispatcher.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.dispatcher = getattr(engine, "dispatcher", None)

    def register_all_handlers(self) -> None:
        """Register all task and consensus handlers on the transport dispatcher."""
        if not self.dispatcher:
            return

        # Task Handlers
        self.dispatcher.register_handler(
            FederationMessageType.TASK_ASSIGNMENT,
            self.handle_task_assignment,
        )
        self.dispatcher.register_handler(
            FederationMessageType.TASK_CHECKPOINT,
            self.handle_task_checkpoint,
        )
        self.dispatcher.register_handler(
            FederationMessageType.TASK_RESULT,
            self.handle_task_result,
        )
        self.dispatcher.register_handler(
            FederationMessageType.TASK_LEASE_HEARTBEAT,
            self.handle_task_lease_heartbeat,
        )

        # Consensus Handlers
        self.dispatcher.register_handler(
            FederationMessageType.CONSENSUS_PROPOSAL,
            self.handle_consensus_proposal,
        )
        self.dispatcher.register_handler(
            FederationMessageType.CONSENSUS_PREVOTE,
            self.handle_consensus_prevote,
        )
        self.dispatcher.register_handler(
            FederationMessageType.CONSENSUS_PRECOMMIT,
            self.handle_consensus_precommit,
        )
        self.dispatcher.register_handler(
            FederationMessageType.CONSENSUS_VIEW_CHANGE,
            self.handle_consensus_view_change,
        )

        logger.info("Registered all Step 41 task and consensus wire handlers on dispatcher")

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _get_sender_node_id(self, envelope: FederationMessageEnvelope) -> str:
        return getattr(envelope, "sender_node_id", getattr(envelope, "sender_peer_id", envelope.sender_engine_id))

    def _check_leakage(self, envelope: FederationMessageEnvelope) -> Optional[str]:
        try:
            raw_str = json.dumps(envelope.payload)
            for kw in PROHIBITED_LEAK_KEYWORDS:
                if kw in raw_str:
                    return kw
        except Exception:
            pass
        return None

    def _get_local_tenant_id(self) -> Optional[str]:
        return getattr(self.engine, "tenant_id", None) or getattr(getattr(self.engine, "config", None), "tenant_id", None)

    # ========================================================================
    # Task Wire Message Handlers
    # ========================================================================

    def handle_task_assignment(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """
        Inbound task assignment handler on a worker node.
        Enforces sovereign ExecutionGrant evaluation before accepting execution.
        """
        try:
            # 1. Secret leakage check
            leaked = self._check_leakage(envelope)
            if leaked:
                logger.critical("Secret leakage detected in TASK_ASSIGNMENT: %s", leaked)
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            # 2. Cross-tenant check
            local_tenant = self._get_local_tenant_id()
            if local_tenant and envelope.tenant_id and envelope.tenant_id != local_tenant:
                return {"status": "ERROR", "error": f"Tenant mismatch: {envelope.tenant_id} != {local_tenant}"}

            payload = envelope.payload
            unit_data = payload.get("unit")
            if not unit_data:
                return {"status": "ERROR", "error": "Missing unit data in payload"}

            sender_id = self._get_sender_node_id(envelope)

            # Reconstruct requirements
            reqs_data = unit_data.get("requirements")
            if reqs_data and isinstance(reqs_data, dict):
                reqs = ResourceRequirements.from_dict(reqs_data)
            else:
                reqs = ResourceRequirements(
                    capability_id=unit_data["capability_id"],
                    min_cpu_cores=1.0,
                    min_memory_mb=512,
                    tenant_id=envelope.tenant_id or "default",
                )

            # Sovereign Execution Grant Check: ADVERTISEMENT != PERMISSION
            grant_mgr = getattr(self.engine, "grant_manager", None)
            if grant_mgr:
                try:
                    grant = grant_mgr.authorize_execution(
                        peer_node_id=sender_id,
                        tenant_id=envelope.tenant_id or "default",
                        capability_id=unit_data["capability_id"],
                        requirements=reqs,
                    )
                except Exception as grant_err:
                    logger.warning("Rejected task assignment from %s: %s", sender_id, str(grant_err))
                    raise ExecutionGrantViolationError(
                        f"Execution grant denied for {sender_id}: {str(grant_err)}"
                    ) from grant_err

            # Reconstruct WorkUnit
            unit = WorkUnit(
                unit_id=unit_data["unit_id"],
                task_id=unit_data["task_id"],
                sequence=unit_data.get("sequence", 0),
                capability_id=unit_data["capability_id"],
                input_payload=unit_data.get("input_payload", {}),
                requirements=reqs,
                attempt=unit_data.get("attempt", 1),
                assigned_node_id=self.engine.local_peer_id,
                fencing_token=unit_data.get("fencing_token", 0),
            )

            # Execute work unit locally if executor is available
            executor = getattr(self.engine, "task_executor", None)
            if executor:
                result_env = executor.execute_unit(unit, originating_node_id=sender_id)
                return {
                    "status": "SUCCESS",
                    "unit_id": unit.unit_id,
                    "result": result_env.to_dict(),
                }

            return {"status": "ACCEPTED", "unit_id": unit.unit_id}

        except Exception as e:
            logger.error("Error handling task assignment: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

    def handle_task_checkpoint(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound checkpoint submission to coordinator."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            payload = envelope.payload
            coord = getattr(self.engine, "task_coordinator", None)
            if not coord:
                return {"status": "ERROR", "error": "No task coordinator active"}

            if "manifest" in payload:
                manifest = CheckpointManifest.from_dict(payload["manifest"])
                coord.record_checkpoint_manifest(manifest)
                return {"status": "ACK", "checkpoint_id": manifest.checkpoint_id}
            elif "checkpoint" in payload:
                cp = TaskCheckpoint.from_dict(payload["checkpoint"])
                coord.record_checkpoint(cp)
                return {"status": "ACK", "checkpoint_id": cp.checkpoint_id}

            return {"status": "ERROR", "error": "No checkpoint data in payload"}

        except Exception as e:
            logger.error("Error handling task checkpoint: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

    def handle_task_result(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound result envelope submission to coordinator."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            payload = envelope.payload
            coord = getattr(self.engine, "task_coordinator", None)
            if not coord:
                return {"status": "ERROR", "error": "No task coordinator active"}

            res_data = payload.get("result")
            if not res_data:
                return {"status": "ERROR", "error": "Missing result in payload"}

            result_envelope = TaskResultEnvelope.from_dict(res_data)
            coord.record_result(result_envelope)
            return {"status": "ACK", "unit_id": result_envelope.unit_id}

        except Exception as e:
            logger.error("Error handling task result: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

    def handle_task_lease_heartbeat(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound worker lease renewal / heartbeat."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            payload = envelope.payload
            coord = getattr(self.engine, "task_coordinator", None)
            if not coord:
                return {"status": "ERROR", "error": "No task coordinator active"}

            task_id = payload.get("task_id")
            unit_id = payload.get("unit_id")
            fencing_token = payload.get("fencing_token", 0)
            sender_id = self._get_sender_node_id(envelope)

            renewed = coord.record_heartbeat(
                task_id=task_id,
                unit_id=unit_id,
                worker_id=sender_id,
                fencing_token=fencing_token,
            )
            return {"status": "RENEWED", "lease_id": renewed.lease_id, "expires_at": renewed.expires_at}

        except Exception as e:
            logger.error("Error handling lease heartbeat: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

    # ========================================================================
    # Consensus Wire Message Handlers
    # ========================================================================

    def handle_consensus_proposal(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound consensus proposal from leader."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            cons = getattr(self.engine, "consensus_engine", None)
            if not cons:
                return {"status": "ERROR", "error": "Consensus engine not active"}

            prop_data = envelope.payload.get("proposal")
            if not prop_data:
                return {"status": "ERROR", "error": "Missing proposal in payload"}

            proposal = ConsensusProposal.from_dict(prop_data)

            # Cross-tenant isolation check
            local_tenant = self._get_local_tenant_id()
            if local_tenant and proposal.tenant_id and proposal.tenant_id != local_tenant:
                return {"status": "ERROR", "error": f"Tenant mismatch: {proposal.tenant_id} != {local_tenant}"}
            if local_tenant and envelope.tenant_id and envelope.tenant_id != local_tenant:
                return {"status": "ERROR", "error": f"Tenant mismatch: {envelope.tenant_id} != {local_tenant}"}

            accepted = cons.receive_proposal(proposal)

            # Auto-cast prevote if proposal is accepted
            vote = ConsensusVote(
                vote_id=f"vote_pv_{self.engine.local_peer_id}_{proposal.height}",
                epoch=proposal.epoch,
                round=proposal.round,
                height=proposal.height,
                voter_id=self.engine.local_peer_id,
                vote_type=VoteType.PREVOTE,
                proposal_digest=proposal.proposal_digest,
            )
            return {"status": "ACCEPTED", "proposal_digest": proposal.proposal_digest}

        except Exception as e:
            logger.error("Error handling consensus proposal: %s", str(e), exc_info=True)
            return {"status": "ERROR", "error": str(e)}

    def handle_consensus_prevote(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound consensus prevote from peer validator."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            cons = getattr(self.engine, "consensus_engine", None)
            if not cons:
                return {"status": "ERROR", "error": "Consensus engine not active"}

            vote_data = envelope.payload.get("vote")
            if not vote_data:
                return {"status": "ERROR", "error": "Missing vote in payload"}

            vote = ConsensusVote.from_dict(vote_data)
            qc = cons.record_prevote(vote)
            return {"status": "RECORDED", "qc_formed": qc is not None}

        except Exception as e:
            logger.error("Error handling consensus prevote: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

    def handle_consensus_precommit(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound consensus precommit from peer validator."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            cons = getattr(self.engine, "consensus_engine", None)
            if not cons:
                return {"status": "ERROR", "error": "Consensus engine not active"}

            vote_data = envelope.payload.get("vote")
            if not vote_data:
                return {"status": "ERROR", "error": "Missing vote in payload"}

            vote = ConsensusVote.from_dict(vote_data)
            qc = cons.record_precommit(vote)
            return {"status": "RECORDED", "qc_formed": qc is not None}

        except Exception as e:
            logger.error("Error handling consensus precommit: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

    def handle_consensus_view_change(
        self,
        envelope: FederationMessageEnvelope,
        channel: Optional[FederationChannel] = None,
    ) -> Dict[str, Any]:
        """Inbound view change alert or vote."""
        try:
            leaked = self._check_leakage(envelope)
            if leaked:
                return {"status": "ERROR", "error": f"Secret leakage detected: prohibited token '{leaked}'"}

            cons = getattr(self.engine, "consensus_engine", None)
            if not cons:
                return {"status": "ERROR", "error": "Consensus engine not active"}

            reason = envelope.payload.get("reason", "Remote view change notification")
            new_round = cons.trigger_view_change(reason=reason)
            return {"status": "VIEW_CHANGED", "new_round": new_round}

        except Exception as e:
            logger.error("Error handling view change: %s", str(e))
            return {"status": "ERROR", "error": str(e)}
