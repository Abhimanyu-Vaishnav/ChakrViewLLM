"""
schema_controller.py
ChakrView Experiment 12 — Schema-Constrained Decoding Controller

Implements a deterministic finite-state controller that:
1. Parses requested fields from the prompt instruction ("entity", "status", "code").
2. Enforces valid JSON object syntax: starts with '{', ends with '}'.
3. Enforces canonical key ordering without duplicates: ["entity", "status", "code"].
4. Emits only requested fields.
5. Injects pointer-selected or oracle-selected values cleanly enclosed in quotes.
6. Handles missing values ("none") per dataset policy.
7. Guarantees 100% JSON parse validity, 100% schema compliance, and 0% key repetition.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Optional, Tuple

class SchemaController:
    """
    Finite-State Controller for Structured JSON Schema Enforcement.
    """
    CANONICAL_ORDER = ["entity", "status", "code"]

    def __init__(self, schema_keys: Optional[List[str]] = None):
        self.schema_keys = schema_keys or self.CANONICAL_ORDER

    def parse_requested_fields(self, instruction: str) -> List[str]:
        """
        Deterministically extracts the requested field names from the task instruction.
        Preserves canonical ordering: ["entity", "status", "code"].
        """
        inst_lower = instruction.lower()
        requested = []
        for key in self.CANONICAL_ORDER:
            if key in inst_lower:
                requested.append(key)
        
        # Fallback if "all" or general phrasing
        if not requested:
            requested = list(self.CANONICAL_ORDER)
        return requested

    def format_constrained_json(
        self,
        requested_fields: List[str],
        field_values: Dict[str, str]
    ) -> Tuple[str, Dict[str, str], bool]:
        """
        Synthesizes the structured JSON string strictly according to the controller state machine.
        
        Guarantees:
        - Exact JSON format: {"key1": "val1", "key2": "val2"}
        - No duplicate keys
        - No unrequested keys
        - No omitted requested keys
        - Clean quote escaping
        """
        ordered_dict = {}
        for key in self.CANONICAL_ORDER:
            if key in requested_fields:
                val = field_values.get(key, "none")
                # Sanitize value string (remove unescaped quotes or trailing junk)
                val_clean = str(val).strip().strip('"').strip("'")
                if not val_clean:
                    val_clean = "none"
                ordered_dict[key] = val_clean

        json_str = json.dumps(ordered_dict, separators=(", ", ": "))
        return json_str, ordered_dict, True
