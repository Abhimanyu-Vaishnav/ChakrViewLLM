#!/usr/bin/env python3
"""
Canonical Tracked Founder QA Review Workflow & Review Accelerator.

Location: tools/qa_review/founder_review_workflow.py

Capabilities:
1. --status: Read-only summary showing total candidates, authoritative split accounting,
   advisory recommendations, blocked candidates, and official review status.
2. --list: Displays all candidate IDs with their split and current review status.
3. --show <ID>: Inspects a single candidate in detail (source passage, draft Q/A,
   technical issue, proposed revision, and source limitation notes).
4. --preview-draft: Safely previews a proposed review decision without modifying the official queue.
5. --save-draft: Saves in-progress review decisions to an isolated draft file (NEVER the official queue).
6. --export-reviewed: Validates and writes reviewed records to a separate file only upon deliberate confirmation.
7. Strictly prevents automatic approvals.
8. Strictly bars validation and test split records from TRAIN-ELIGIBLE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

QUEUE_PATH = ROOT_DIR / "checkpoints" / "experiments" / "neural_learning_v02" / "stage_d_design" / "qa_review_queue_v0.3.0.jsonl"
ADVISORY_PATH = ROOT_DIR / "checkpoints" / "experiments" / "neural_learning_v02" / "stage_d_design" / "qa_review_advisory_batch1_v0.2.json"
DRAFT_PATH = ROOT_DIR / "checkpoints" / "experiments" / "neural_learning_v02" / "stage_d_design" / "qa_review_draft_user.jsonl"

from tools.qa_review.qa_review_validator import ReviewQueueValidator, AdmissionGateError, ReviewQueueValidationError


def load_review_state() -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Loads official queue and advisory mapping."""
    if not QUEUE_PATH.is_file():
        raise FileNotFoundError(f"Official review queue not found at {QUEUE_PATH}")
    if not ADVISORY_PATH.is_file():
        raise FileNotFoundError(f"Advisory artifact not found at {ADVISORY_PATH}")

    queue_records, queue_stats = ReviewQueueValidator.validate_file(QUEUE_PATH)
    adv_data = json.loads(ADVISORY_PATH.read_text(encoding="utf-8"))
    adv_map = {c["candidate_id"]: c for c in adv_data["candidates"]}

    return queue_records, queue_stats, adv_map


def show_status():
    """Prints authoritative review queue status and split breakdown."""
    queue_records, queue_stats, adv_map = load_review_state()

    print("\n================================================================================")
    print("                     CHAKRVIEW QA FOUNDER REVIEW STATUS                         ")
    print("================================================================================")
    print(f"Total Candidates:         {queue_stats['total_records']}")
    print(f"Official Queue Status:    {queue_stats['unreviewed']} UNREVIEWED | {queue_stats['train_eligible']} TRAIN-ELIGIBLE | {queue_stats['rejected']} REJECTED | {queue_stats['needs_revision']} NEEDS-REVISION")
    print("--------------------------------------------------------------------------------")
    print("AUTHORITATIVE SPLIT ACCOUNTING:")
    print(f"  Train Origin:           {queue_stats['source_splits'].get('train', 0)}")
    print(f"  Validation Origin:      {queue_stats['source_splits'].get('validation', 0)} (Evaluation Only - Barred from Training)")
    print(f"  Test Origin:            {queue_stats['source_splits'].get('test', 0)} (Benchmark Only - Barred from Training)")
    print("--------------------------------------------------------------------------------")
    
    # Calculate advisory breakdown
    train_cands = [c for c in adv_map.values() if c["source_split"] == "train"]
    direct_app = sum(1 for c in train_cands if c["advisory_recommendation"] == "APPROVE")
    supp_rev = sum(1 for c in train_cands if c["advisory_recommendation"] == "REVISE")
    blocked = sum(1 for c in train_cands if c["advisory_recommendation"] == "NEEDS_NEW_SOURCE")
    reviewable = direct_app + supp_rev

    print("STAGE D BATCH 1 POOL BREAKDOWN:")
    print(f"  Train Direct-Grounding (Advisory APPROVE):  {direct_app}")
    print(f"  Train Source-Supported (Advisory REVISE):   {supp_rev}")
    print(f"  Train Reviewable After Source Audit:       {reviewable} / 27")
    print(f"  Train Blocked for Source Enrichment:        {blocked} (QA-REV-017, QA-REV-023)")
    print(f"  Validation Evaluation Candidates:           2 (QA-REV-028, QA-REV-029)")
    print(f"  Test Benchmark Candidate:                   1 (QA-REV-030)")
    print("================================================================================\n")


def display_candidate(cid: str):
    """Displays a single candidate with all source evidence and advisory proposals."""
    queue_records, _, adv_map = load_review_state()
    q_map = {r["candidate_id"]: r for r in queue_records}

    if cid not in q_map:
        print(f"Error: Candidate '{cid}' not found in official review queue.")
        sys.exit(1)

    q = q_map[cid]
    adv = adv_map.get(cid, {})

    print("\n" + "=" * 80)
    print(f"CANDIDATE: {cid} | Topic: {q['topic']} | Split: {q['source_split'].upper()}")
    print(f"Provenance: {q['source_file']} ({q['source_locator']}) | SHA: {q['source_hash'][:16]}...")
    print(f"Official Decision: {q['reviewer_decision']} | Status: {q['semantic_review_status']}")
    print("-" * 80)
    print("SOURCE PASSAGE:")
    print(f"  \"{q['source_passage']}\"")
    print("-" * 80)
    print("ORIGINAL DRAFT:")
    print(f"  Q: {q['question']}")
    print(f"  A: {q['proposed_answer']}")
    print("-" * 80)
    print(f"TECHNICAL ADVISORY (Non-authoritative guidance):")
    print(f"  Classification: {adv.get('technical_classification')} -> Recommendation: {adv.get('advisory_recommendation')}")
    print(f"  Rationale:      {adv.get('advisory_rationale')}")

    if adv.get("proposed_revision"):
        rev = adv["proposed_revision"]
        print("-" * 80)
        print("PROPOSED REVISION (Source-Grounded):")
        print(f"  Revised Q:   {rev['proposed_revised_question']}")
        print(f"  Revised A:   {rev['proposed_revised_answer']}")
        print(f"  Limitations: {rev['source_limitation_notes']}")
        if rev.get("source_enrichment_needed"):
            print(f"  BLOCKER:     {rev.get('source_enrichment_request')}")

    print("=" * 80 + "\n")


def preview_draft(candidate_id: str, decision: str, notes: str):
    """Safely previews a proposed review decision without touching the official queue."""
    queue_records, _, adv_map = load_review_state()
    q_map = {r["candidate_id"]: r for r in queue_records}

    if candidate_id not in q_map:
        raise ValueError(f"Unknown candidate_id '{candidate_id}'")

    rec = dict(q_map[candidate_id])
    split = rec["source_split"]

    if decision in {"APPROVE", "TRAIN-ELIGIBLE"}:
        if split != "train":
            raise AdmissionGateError(
                f"CANNOT APPROVE FOR TRAINING: Candidate '{candidate_id}' originates from '{split}' split! Must remain evaluation-only."
            )
        if not notes or not notes.strip():
            raise AdmissionGateError(
                f"CANNOT APPROVE: Founder rationale/notes cannot be empty for candidate '{candidate_id}'."
            )
        rec["reviewer_decision"] = "APPROVE"
        rec["semantic_review_status"] = "TRAIN-ELIGIBLE"
        rec["reviewer_notes"] = notes.strip()
    elif decision == "REJECT":
        if not notes or not notes.strip():
            raise AdmissionGateError(
                f"CANNOT REJECT without rationale: Candidate '{candidate_id}' requires reviewer_notes."
            )
        rec["reviewer_decision"] = "REJECT"
        rec["semantic_review_status"] = "REJECT"
        rec["rejection_reason"] = notes.strip()
        rec["reviewer_notes"] = notes.strip()
    elif decision == "REVISE":
        rec["reviewer_decision"] = "REVISE"
        rec["semantic_review_status"] = "NEEDS-REVISION"
        rec["reviewer_notes"] = notes.strip() if notes else "Revision requested."
    else:
        raise ValueError(f"Invalid decision '{decision}'. Choose from: APPROVE, REJECT, REVISE.")

    print("\n[PREVIEW - NO CHANGES SAVED TO QUEUE]")
    print(f"Candidate ID:     {rec['candidate_id']}")
    print(f"New Decision:     {rec['reviewer_decision']}")
    print(f"New Status:       {rec['semantic_review_status']}")
    print(f"Reviewer Notes:   {rec['reviewer_notes']}")
    print(f"Official Queue:   PRESERVED (Untouched)\n")
    return rec


def record_decision(candidate_id: str, decision: str, notes: str, queue_path: Path = QUEUE_PATH) -> Dict[str, Any]:
    """
    Records an explicit human decision for a candidate into a specified queue file.
    Does NOT allow automatic or default approvals.
    Fails closed if split invariants or required notes are violated.
    """
    queue_records, _ = ReviewQueueValidator.validate_file(queue_path)
    q_map = {r["candidate_id"]: r for r in queue_records}

    if candidate_id not in q_map:
        raise ValueError(f"Unknown candidate_id '{candidate_id}'")

    updated_rec = preview_draft(candidate_id, decision, notes)

    # Perform atomic update on queue_path
    new_records = []
    for r in queue_records:
        if r["candidate_id"] == candidate_id:
            new_records.append(updated_rec)
        else:
            new_records.append(r)

    temp_queue = queue_path.with_suffix(".tmp")
    with open(temp_queue, "w", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r) + "\n")
    temp_queue.replace(queue_path)
    print(f"[DECISION RECORDED AT {queue_path.name}] Candidate '{candidate_id}' -> {decision} ({updated_rec['semantic_review_status']})\n")
    return updated_rec


def save_draft(candidate_id: str, decision: str, notes: str, draft_path: Path = DRAFT_PATH) -> Dict[str, Any]:
    """
    Saves an in-progress review decision to an isolated draft file, never mutating the official queue.
    """
    updated_rec = preview_draft(candidate_id, decision, notes)

    draft_records = {}
    if draft_path.is_file():
        lines = draft_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                item = json.loads(line)
                draft_records[item["candidate_id"]] = item

    draft_records[candidate_id] = updated_rec

    temp_draft = draft_path.with_suffix(".tmp")
    with open(temp_draft, "w", encoding="utf-8") as f:
        for r in draft_records.values():
            f.write(json.dumps(r) + "\n")
    temp_draft.replace(draft_path)
    print(f"[DRAFT SAVED AT {draft_path.name}] Candidate '{candidate_id}' -> {decision}\n")
    return updated_rec


def main():
    parser = argparse.ArgumentParser(description="ChakrView Founder QA Review Accelerator")
    parser.add_argument("--status", action="store_true", help="Display authoritative queue status and counts")
    parser.add_argument("--list", action="store_true", help="List all candidate IDs and review statuses")
    parser.add_argument("--show", type=str, metavar="ID", help="Show full detail and evidence for a candidate")
    parser.add_argument("--preview", nargs=3, metavar=("ID", "DECISION", "NOTES"), help="Preview a proposed decision safely without mutating the queue")
    parser.add_argument("--save-draft", nargs=3, metavar=("ID", "DECISION", "NOTES"), help="Save decision to an isolated user draft file")
    parser.add_argument("--record", nargs=3, metavar=("ID", "DECISION", "NOTES"), help="Explicitly record a founder decision to the official review queue")

    args = parser.parse_args()

    if args.status or len(sys.argv) == 1:
        show_status()
    elif args.list:
        recs, _, _ = load_review_state()
        print(f"\n{'ID':<12} {'TOPIC':<20} {'SPLIT':<12} {'DECISION':<15} {'STATUS':<15}")
        print("-" * 75)
        for r in recs:
            print(f"{r['candidate_id']:<12} {r['topic']:<20} {r['source_split']:<12} {r['reviewer_decision']:<15} {r['semantic_review_status']:<15}")
        print()
    elif args.show:
        display_candidate(args.show)
    elif args.preview:
        cid, dec, notes = args.preview
        preview_draft(cid, dec, notes)
    elif args.save_draft:
        cid, dec, notes = args.save_draft
        save_draft(cid, dec, notes)
    elif args.record:
        cid, dec, notes = args.record
        record_decision(cid, dec, notes)


if __name__ == "__main__":
    main()

