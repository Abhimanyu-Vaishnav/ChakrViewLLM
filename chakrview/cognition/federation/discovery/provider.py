"""
Controlled Discovery Providers for Federation Nodes (Step 35).

Implements bounded, non-broadcast discovery mechanisms:
1. Static Configuration
2. Local Configuration File (JSON)
3. In-Process Advertisement
CRITICAL: Discovery derives CANDIDATES only; it creates zero trust and zero execution authority.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from chakrview.cognition.federation.discovery.models import (
    FederationNodeEndpoint,
    FederationNodeCandidate,
    NodeDiscoverySource,
    NodeProtocol,
    NodeAddress,
)
from chakrview.cognition.federation.discovery.errors import (
    EndpointValidationError,
    CandidateRegistrationError,
)
from chakrview.cognition.transport.security.models import TLSMode

logger = logging.getLogger(__name__)


class DiscoveryProvider(ABC):
    """Abstract interface for bounded node discovery mechanisms."""

    @abstractmethod
    def discover_candidates(self, current_epoch: int = 1) -> List[FederationNodeCandidate]:
        """Discover and return unauthenticated candidate nodes."""
        pass


class StaticConfigDiscoveryProvider(DiscoveryProvider):
    """
    Discovers nodes from a pre-configured list of explicit endpoint descriptors.
    """

    def __init__(self, endpoints: List[Any], zone_id: Optional[str] = None) -> None:
        self.endpoints: List[FederationNodeEndpoint] = []
        for ep in endpoints:
            if isinstance(ep, FederationNodeEndpoint):
                self.endpoints.append(ep)
            elif isinstance(ep, dict):
                proto = ep.get("protocol", "MTLS")
                if isinstance(proto, str):
                    proto = NodeProtocol(proto.upper())
                ep_obj = FederationNodeEndpoint.create(
                    host=ep["host"],
                    port=int(ep["port"]),
                    protocol=proto,
                    zone_id=ep.get("zone_id", zone_id or "zone-default"),
                )
                self.endpoints.append(ep_obj)
        self.zone_id = zone_id

    def discover(self, current_epoch: int = 1) -> List[FederationNodeCandidate]:
        """Convenience alias for discover_candidates."""
        return self.discover_candidates(current_epoch=current_epoch)

    def discover_candidates(self, current_epoch: int = 1) -> List[FederationNodeCandidate]:
        candidates: List[FederationNodeCandidate] = []
        for ep in self.endpoints:
            cand = FederationNodeCandidate.create(
                endpoint=ep,
                discovery_source=NodeDiscoverySource.STATIC_CONFIG,
                discovered_epoch=current_epoch,
            )
            candidates.append(cand)
        return candidates


class FileConfigDiscoveryProvider(DiscoveryProvider):
    """
    Discovers nodes from a local configuration file on disk.
    File must be valid JSON containing a list of permitted endpoint records.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)

    def discover_candidates(self, current_epoch: int = 1) -> List[FederationNodeCandidate]:
        if not self.config_path.exists():
            logger.warning(f"Discovery config file not found: {self.config_path}")
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise EndpointValidationError("Discovery config file must contain a JSON list of endpoint objects.")

            candidates: List[FederationNodeCandidate] = []
            for item in data:
                ep = FederationNodeEndpoint.create(
                    host=item["host"],
                    port=int(item["port"]),
                    protocol=NodeProtocol(item.get("protocol", NodeProtocol.MTLS.value)),
                    zone_id=item.get("zone_id", "zone-remote"),
                    tls_mode=TLSMode(item.get("tls_mode", TLSMode.MTLS.value)),
                    expected_cert_fingerprint=item.get("expected_cert_fingerprint"),
                    expected_san=item.get("expected_san"),
                    expected_engine_id=item.get("expected_engine_id"),
                    metadata=item.get("metadata", {}),
                )
                cand = FederationNodeCandidate.create(
                    endpoint=ep,
                    discovery_source=NodeDiscoverySource.CONFIG_FILE,
                    discovered_epoch=current_epoch,
                )
                candidates.append(cand)
            return candidates

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise EndpointValidationError(f"Malformed discovery configuration in {self.config_path}: {e}") from e


class InProcessAdvertisementDiscoveryProvider(DiscoveryProvider):
    """
    Discovers candidate endpoints advertised by locally running engines or dev fixtures.
    """

    def __init__(self) -> None:
        self._advertised_endpoints: List[FederationNodeEndpoint] = []

    def advertise_engine(
        self,
        engine: Any,
        host: str = "127.0.0.1",
        port: int = 9000,
        protocol: NodeProtocol = NodeProtocol.MTLS,
        expected_cert_fingerprint: Optional[str] = None,
    ) -> FederationNodeEndpoint:
        """Register an in-process engine's transport endpoint."""
        zone_id = getattr(engine, "local_zone_id", "zone-local")
        engine_id = getattr(getattr(engine, "engine_identity", None), "engine_id", None)
        ep = FederationNodeEndpoint.create(
            host=host,
            port=port,
            protocol=protocol,
            zone_id=zone_id,
            tls_mode=TLSMode.MTLS if protocol == NodeProtocol.MTLS else (TLSMode.TLS if protocol == NodeProtocol.TLS else TLSMode.PLAINTEXT_TEST_ONLY),
            expected_cert_fingerprint=expected_cert_fingerprint,
            expected_engine_id=engine_id,
            metadata={"in_process": True},
        )
        self._advertised_endpoints.append(ep)
        return ep

    def advertise(self, endpoint: FederationNodeEndpoint) -> FederationNodeEndpoint:
        """Directly register an advertised endpoint."""
        self._advertised_endpoints.append(endpoint)
        return endpoint

    def discover_candidates(self, current_epoch: int = 1) -> List[FederationNodeCandidate]:
        candidates: List[FederationNodeCandidate] = []
        for ep in self._advertised_endpoints:
            cand = FederationNodeCandidate.create(
                endpoint=ep,
                discovery_source=NodeDiscoverySource.IN_PROCESS_ADVERTISEMENT,
                discovered_epoch=current_epoch,
            )
            candidates.append(cand)
        return candidates


class CompositeDiscoveryService:
    """
    Coordinates multiple controlled discovery providers, deduplicates candidates,
    and enforces maximum candidate capacity ceilings.
    """

    def __init__(
        self,
        providers: Optional[List[DiscoveryProvider]] = None,
        max_candidates: int = 32,
    ) -> None:
        self.providers: List[DiscoveryProvider] = list(providers or [])
        self.max_candidates = max_candidates
        self._seen_candidate_ids: Set[str] = set()

    def add_provider(self, provider: DiscoveryProvider) -> None:
        self.providers.append(provider)

    def discover(self, current_epoch: int = 1) -> List[FederationNodeCandidate]:
        """
        Poll all configured providers, deduplicate by endpoint_id, and return candidate list.
        """
        discovered: List[FederationNodeCandidate] = []
        seen_endpoints: Set[str] = set()

        for provider in self.providers:
            cands = provider.discover_candidates(current_epoch=current_epoch)
            for c in cands:
                if c.endpoint.endpoint_id in seen_endpoints:
                    continue
                if len(discovered) >= self.max_candidates:
                    logger.warning(f"Candidate capacity ceiling ({self.max_candidates}) reached during discovery.")
                    break

                seen_endpoints.add(c.endpoint.endpoint_id)
                self._seen_candidate_ids.add(c.candidate_id)
                discovered.append(c)

        return discovered
