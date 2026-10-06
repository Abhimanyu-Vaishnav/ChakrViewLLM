"""
ChakrView Step 137: Sovereign Domain Module Architecture.

Defines the formal governed contract and lifecycle for domain modules:
- Universal Neural Core remains untouched.
- Domain modules encapsulate domain vocabulary, knowledge sources, curriculum identity,
  evaluation suite, capability profile, resource requirements, and lifecycle state.
- Lifecycle: DISCOVER -> REGISTER -> VALIDATE -> ACTIVATE -> EVALUATE -> UPDATE -> DEPRECATE
"""

from __future__ import annotations

import enum
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DomainModuleStatus(str, enum.Enum):
    DISCOVERED = "DISCOVERED"
    REGISTERED = "REGISTERED"
    VALIDATED = "VALIDATED"
    ACTIVATED = "ACTIVATED"
    DEPRECATED = "DEPRECATED"


@dataclass
class DomainCapabilityProfile:
    primary_tasks: List[str] = field(default_factory=list)
    required_context_tokens: int = 256
    minimum_accuracy_threshold: float = 0.70
    max_acceptable_perplexity: float = 50.0
    allows_heuristics: bool = True


@dataclass
class DomainResourceRequirements:
    cpu_cores: int = 1
    memory_mb: int = 512
    max_latency_ms: float = 1000.0


@dataclass
class DomainModuleContract:
    domain_id: str
    version: str
    description: str
    domain_vocabulary: List[str]
    knowledge_sources: List[str]
    curriculum_identity: str
    evaluation_suite: str
    capability_profile: DomainCapabilityProfile
    resource_requirements: DomainResourceRequirements
    status: DomainModuleStatus = DomainModuleStatus.DISCOVERED
    validation_status: bool = False
    registered_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainModuleContract:
        data_copy = dict(data)
        if "status" in data_copy and isinstance(data_copy["status"], str):
            data_copy["status"] = DomainModuleStatus(data_copy["status"])
        if "capability_profile" in data_copy and isinstance(data_copy["capability_profile"], dict):
            data_copy["capability_profile"] = DomainCapabilityProfile(**data_copy["capability_profile"])
        if "resource_requirements" in data_copy and isinstance(data_copy["resource_requirements"], dict):
            data_copy["resource_requirements"] = DomainResourceRequirements(**data_copy["resource_requirements"])
        return cls(**data_copy)


class GovernedDomainRegistry:
    """
    Durable SQLite registry governing domain modules and enforcing lifecycle transitions.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domain_modules (
                    domain_id TEXT PRIMARY KEY,
                    version TEXT,
                    description TEXT,
                    status TEXT,
                    validation_status INTEGER,
                    contract_json TEXT,
                    registered_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()

    def _save_domain(self, contract: DomainModuleContract) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO domain_modules
                (domain_id, version, description, status, validation_status, contract_json, registered_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                contract.domain_id,
                contract.version,
                contract.description,
                contract.status.value,
                1 if contract.validation_status else 0,
                json.dumps(contract.to_dict()),
                contract.registered_at,
                contract.updated_at,
            ))
            conn.commit()
        return True

    def register_domain(self, contract: DomainModuleContract) -> bool:
        """Registers a domain in REGISTERED state."""
        contract.status = DomainModuleStatus.REGISTERED
        contract.updated_at = time.time()
        return self._save_domain(contract)

    def validate_domain(self, domain_id: str, is_valid: bool) -> bool:
        """Transitions registered domain to VALIDATED if is_valid is True."""
        contract = self.get_domain(domain_id)
        if not contract:
            return False
        if contract.status not in (DomainModuleStatus.REGISTERED, DomainModuleStatus.VALIDATED):
            return False
        contract.validation_status = is_valid
        contract.status = DomainModuleStatus.VALIDATED if is_valid else DomainModuleStatus.REGISTERED
        contract.updated_at = time.time()
        return self._save_domain(contract)

    def activate_domain(self, domain_id: str) -> bool:
        """Activates a validated domain module."""
        contract = self.get_domain(domain_id)
        if not contract or not contract.validation_status:
            return False
        contract.status = DomainModuleStatus.ACTIVATED
        contract.updated_at = time.time()
        return self._save_domain(contract)

    def deprecate_domain(self, domain_id: str) -> bool:
        """Deprecates a domain module."""
        contract = self.get_domain(domain_id)
        if not contract:
            return False
        contract.status = DomainModuleStatus.DEPRECATED
        contract.updated_at = time.time()
        return self._save_domain(contract)

    def get_domain(self, domain_id: str) -> Optional[DomainModuleContract]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT contract_json FROM domain_modules WHERE domain_id = ?", (domain_id,))
            row = cursor.fetchone()
            if row:
                return DomainModuleContract.from_dict(json.loads(row[0]))
        return None

    def list_domains(self, status: Optional[DomainModuleStatus] = None) -> List[DomainModuleContract]:
        domains: List[DomainModuleContract] = []
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute("SELECT contract_json FROM domain_modules WHERE status = ?", (status.value,))
            else:
                cursor = conn.execute("SELECT contract_json FROM domain_modules")
            for row in cursor.fetchall():
                domains.append(DomainModuleContract.from_dict(json.loads(row[0])))
        return domains
