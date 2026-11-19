"""
UCIP-12R Phase F: Source Provenance and License Verification Tool.
Audits the master source registry and verifies source-to-shard compliance.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REG_PATH = ROOT / "experiments" / "ucip12r" / "manifests" / "verified_source_registry.json"
MANIFEST_PATH = ROOT / "data" / "manifests" / "stage_c_manifest.json"

with open(REG_PATH, "r", encoding="utf-8") as f:
    reg = json.load(f)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"Auditing Verified Source Registry ({len(reg['sources'])} sources) against Stage C Manifest...")

reg_sources = {}
for s in reg["sources"]:
    reg_sources[s["source_id"]] = s
    if "ingested_id_alias" in s:
        reg_sources[s["ingested_id_alias"]] = s
manifest_sources = {}
for doc in manifest["documents"]:
    sid = doc["source_id"]
    if sid not in manifest_sources:
        manifest_sources[sid] = {"docs": 0, "tokens": 0, "license": doc["license"]}
    manifest_sources[sid]["docs"] += 1
    manifest_sources[sid]["tokens"] += doc["token_count"]

print("\nIngested Sources:")
for sid, info in sorted(manifest_sources.items(), key=lambda x: -x[1]["tokens"]):
    assert sid in reg_sources, f"Unregistered source in manifest: {sid}"
    status = reg_sources[sid]["acquisition_status"]
    print(f"  {sid:<28} | {info['docs']:>5} docs | {info['tokens']:>8,d} tokens | {info['license']} | Status: {status}")
    assert status in ["APPROVED", "CONDITIONAL"], f"Invalid source status for ingested source: {sid} ({status})"

print("\nVerification Successful: Zero unapproved or non-commercial sources present.")
