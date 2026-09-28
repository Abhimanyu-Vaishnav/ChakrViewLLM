"""
Federated Agent Registry (Step 26).

Enforces:
1. Strict identity validation & duplicate collision prevention.
2. Hard capacity bounds on registered agents (default max <= 8).
3. Multi-tenant and session scoping: agents registered under Tenant A
   are completely invisible and inaccessible to Tenant B.
4. Agent health and operational status tracking.
"""

from typing import Dict, List, Optional, Any, Set

from chakrview.cognition.federated.models import (
    AgentContract,
    AgentIdentity,
    AgentRole,
    AgentStatus,
)


class DuplicateAgentIdentityError(ValueError):
    """Raised when an agent with an identical ID is registered."""
    pass


class RegistryCapacityExceededError(RuntimeError):
    """Raised when the registered agent count exceeds the hard ceiling."""
    pass


class AgentNotFoundError(KeyError):
    """Raised when querying an unregistered or inaccessible agent."""
    pass


class AgentRegistry:
    """
    Bounded, tenant-scoped registry managing active cognitive agent contracts.
    """

    def __init__(self, max_agents: int = 8) -> None:
        self.max_agents = min(max_agents, 8)  # Enforce hard ceiling of 8
        # _agents[tenant_id][agent_id] = AgentContract
        self._agents: Dict[str, Dict[str, AgentContract]] = {}

    def register(self, contract: AgentContract) -> None:
        """
        Register a new cognitive agent contract.
        Validates contract, checks tenant capacity, and prevents duplicate collisions.
        """
        contract.validate()
        tenant_id = contract.identity.tenant_id
        agent_id = contract.identity.agent_id

        if tenant_id not in self._agents:
            self._agents[tenant_id] = {}

        if agent_id in self._agents[tenant_id]:
            raise DuplicateAgentIdentityError(
                f"Agent with ID '{agent_id}' is already registered under tenant '{tenant_id}'."
            )

        if len(self._agents[tenant_id]) >= self.max_agents:
            raise RegistryCapacityExceededError(
                f"Registry capacity exceeded: maximum {self.max_agents} agents allowed for tenant '{tenant_id}'."
            )

        self._agents[tenant_id][agent_id] = contract

    def unregister(self, agent_id: str, tenant_id: str) -> bool:
        """Unregister an agent under a given tenant."""
        if tenant_id in self._agents and agent_id in self._agents[tenant_id]:
            del self._agents[tenant_id][agent_id]
            return True
        return False

    def get_agent(self, agent_id: str, tenant_id: str) -> Optional[AgentContract]:
        """Retrieve contract for an agent scoped strictly by tenant."""
        if tenant_id not in self._agents:
            return None
        return self._agents[tenant_id].get(agent_id)

    def find_by_role(
        self,
        role: AgentRole,
        tenant_id: str,
        session_id: Optional[str] = None,
        only_ready: bool = True,
    ) -> List[AgentContract]:
        """Discover agents matching role, tenant, and optional status."""
        if tenant_id not in self._agents:
            return []
        results = []
        for contract in self._agents[tenant_id].values():
            if contract.identity.role == role:
                if session_id and contract.identity.session_id != session_id:
                    continue
                if only_ready and contract.identity.status != AgentStatus.READY:
                    continue
                results.append(contract)
        return results

    def list_agents(self, tenant_id: str, session_id: Optional[str] = None) -> List[AgentContract]:
        """List all registered agents for a specific tenant and session."""
        if tenant_id not in self._agents:
            return []
        if session_id:
            return [c for c in self._agents[tenant_id].values() if c.identity.session_id == session_id]
        return list(self._agents[tenant_id].values())

    def update_status(self, agent_id: str, tenant_id: str, status: AgentStatus) -> None:
        """Update operational status of an agent."""
        agent = self.get_agent(agent_id, tenant_id)
        if not agent:
            raise AgentNotFoundError(f"Agent '{agent_id}' not found under tenant '{tenant_id}'.")
        agent.identity.status = status

    def clear(self, tenant_id: Optional[str] = None) -> None:
        """Clear registry records."""
        if tenant_id:
            if tenant_id in self._agents:
                self._agents[tenant_id].clear()
        else:
            self._agents.clear()
