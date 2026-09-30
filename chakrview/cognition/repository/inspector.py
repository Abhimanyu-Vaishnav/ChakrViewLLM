"""
ChakrView Step 59: AST-Based Repository Inspector.

Extracts structured information from Python files:
- Imported modules and specific imported symbols
- Defined functions and classes
- Docstrings and signatures
- Test functions and referenced targets
Deterministic, pure CPU-first AST parsing with fail-closed safety.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class ModuleInspection:
    """Structured inspection summary for a single Python module."""
    rel_path: str
    module_name: str
    imports: List[str] = field(default_factory=list)  # e.g., ["models", "tax_service"]
    imported_symbols: Dict[str, str] = field(default_factory=dict)  # symbol -> module
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    is_test: bool = False
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryInspector:
    """
    Lightweight, CPU-first AST scanner for analyzing repository files.
    """

    @staticmethod
    def inspect_source(rel_path: str, content: str) -> ModuleInspection:
        """Parse source code string and extract modules, functions, classes, and calls."""
        mod_name = Path(rel_path).stem
        is_test = rel_path.startswith("test_") or "/test_" in rel_path or "\\test_" in rel_path

        try:
            tree = ast.parse(content, filename=rel_path)
        except Exception as e:
            return ModuleInspection(
                rel_path=rel_path,
                module_name=mod_name,
                is_test=is_test,
                parse_error=str(e),
            )

        imports: List[str] = []
        imported_symbols: Dict[str, str] = {}
        functions: List[str] = []
        classes: List[str] = []
        calls: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imports.append(mod)
                for alias in node.names:
                    imported_symbols[alias.name] = mod
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        return ModuleInspection(
            rel_path=rel_path,
            module_name=mod_name,
            imports=sorted(list(set(imports))),
            imported_symbols=imported_symbols,
            functions=sorted(list(set(functions))),
            classes=sorted(list(set(classes))),
            calls=sorted(list(set(calls))),
            is_test=is_test,
            parse_error=None,
        )

    @classmethod
    def inspect_manifest(cls, manifest_files: List[Any]) -> Dict[str, ModuleInspection]:
        """Inspect all files declared in a ProjectManifest."""
        inspections: Dict[str, ModuleInspection] = {}
        for f in manifest_files:
            rel_path = getattr(f, "path", "")
            content = getattr(f, "content", "")
            if rel_path.endswith(".py"):
                inspections[rel_path] = cls.inspect_source(rel_path, content)
        return inspections
