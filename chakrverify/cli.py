"""
ChakrVerify Local Alpha: Command Line Interface.
Location: chakrverify/cli.py

Usage:
  python -m chakrverify.cli --docs path/to/doc.txt --claim "Field Programmable Gate Array is configurable"
  python -m chakrverify.cli --docs doc1.txt doc2.txt --batch claims.json --output report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chakrverify.core import (
    ClaimVerificationEngine,
    VerificationVerdict,
    VerificationResult,
)


def main():
    parser = argparse.ArgumentParser(
        prog="chakrverify",
        description="ChakrVerify Local Alpha: Indigenous, Local Document Claim Verification Engine.",
    )
    parser.add_argument("--docs", nargs="+", required=True, help="Path(s) to reference document(s) (TXT, Markdown)")
    parser.add_argument("--claim", type=str, help="Single claim string to verify")
    parser.add_argument("--batch", type=str, help="Path to JSON file containing list of claim strings")
    parser.add_argument("--output", type=str, help="Path to export results as JSON")

    args = parser.parse_args()

    # Ingest documents
    engine = ClaimVerificationEngine()
    for doc_path in args.docs:
        p = Path(doc_path)
        if not p.exists():
            print(f"Error: document '{doc_path}' not found.", file=sys.stderr)
            sys.exit(1)
        engine.add_document(p)

    claims: list[str] = []
    if args.claim:
        claims.append(args.claim)
    elif args.batch:
        batch_p = Path(args.batch)
        if not batch_p.exists():
            print(f"Error: batch file '{args.batch}' not found.", file=sys.stderr)
            sys.exit(1)
        data = json.loads(batch_p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            claims.extend([c if isinstance(c, str) else c.get("claim", "") for c in data])
    else:
        print("Error: must provide either --claim or --batch.", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = []
    print(f"\n=======================================================")
    print(f"  CHAKRVERIFY LOCAL ALPHA — CLAIM VERIFICATION REPORT  ")
    print(f"=======================================================\n")
    print(f"Indexed {len(args.docs)} document(s) across {len(engine.chunks)} chunk(s).\n")

    for c in claims:
        res = engine.verify_claim(c)
        results.append(res.to_dict())

        # Colored formatting
        v_tag = f"[{res.verdict.value}]"
        print(f"Claim: \"{res.claim}\"")
        print(f"Verdict:    {v_tag} (Confidence: {res.confidence*100:.1f}%)")
        print(f"Reasoning:  {res.explanation}")
        if res.citations:
            print("Citations:")
            for cit in res.citations:
                print(f"  - Document: {cit.doc_id} (chars {cit.start_char}-{cit.end_char})")
                print(f"    Excerpt:  \"{cit.exact_text[:120]}...\"")
        print(f"Latency:    {res.latency_ms:.2f} ms\n-------------------------------------------------------")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nExported {len(results)} verification results to: {out_p}")


if __name__ == "__main__":
    main()
