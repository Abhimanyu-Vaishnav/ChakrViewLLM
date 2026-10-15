"""
CLI Interface for ChakrView Local Knowledge Assistant MVP.
Location: chakrview/runtime/assistant_cli.py

Usage:
  python -m chakrview.runtime.assistant_cli ingest <dir_or_file>
  python -m chakrview.runtime.assistant_cli ask "<question>"
  python -m chakrview.runtime.assistant_cli list-docs
  python -m chakrview.runtime.assistant_cli interactive
"""
import argparse
import os
import sys
from pathlib import Path

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant


def main():
    parser = argparse.ArgumentParser(
        description="ChakrView Local Knowledge Assistant MVP: Grounded Answers, Traceable Citations & Session Memory."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document file or directory of documents")
    ingest_parser.add_argument("path", type=str, help="Path to text or markdown document or directory")

    # Ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a question against indexed documents")
    ask_parser.add_argument("query", type=str, help="Question string")
    ask_parser.add_argument("--doc-dir", type=str, default="experiments/batch1021_1060/data/sample_corpus", help="Corpus directory to auto-load")

    # List docs subcommand
    list_parser = subparsers.add_parser("list-docs", help="List currently indexed documents")
    list_parser.add_argument("--doc-dir", type=str, default="experiments/batch1021_1060/data/sample_corpus", help="Corpus directory to inspect")

    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Start an interactive multi-turn session")
    interactive_parser.add_argument("--doc-dir", type=str, default="experiments/batch1021_1060/data/sample_corpus", help="Corpus directory to auto-load")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    assistant = LocalKnowledgeAssistant()

    if args.command == "ingest":
        p = Path(args.path)
        if p.is_dir():
            count = assistant.ingest_directory(p)
            print(f"Successfully ingested directory '{p}': indexed {count} chunks.")
        elif p.is_file():
            count = assistant.ingest_document(p)
            print(f"Successfully ingested document '{p}': indexed {count} chunks.")
        else:
            print(f"Error: Path '{p}' does not exist.")
            sys.exit(1)

    elif args.command == "list-docs":
        p = Path(args.doc_dir)
        if p.exists():
            assistant.ingest_directory(p)
        docs = assistant.list_indexed_documents()
        print(f"\nIndexed Documents ({len(docs)} total):")
        print("-" * 65)
        for d in docs:
            print(f" - {d['doc_id']} ({d['chunk_count']} chunks, {d['file_size_bytes']} bytes, SHA-256: {d['sha256'][:10]}...)")
        print("-" * 65)

    elif args.command == "ask":
        p = Path(args.doc_dir)
        if p.exists():
            assistant.ingest_directory(p)
        resp = assistant.ask(args.query)
        print(f"\nDecision: {resp.decision} (confidence: {resp.confidence}) | Latency: {resp.latency_ms} ms")
        print(f"Answer: {resp.answer}")
        if resp.citations:
            print(f"\nCitations ({len(resp.citations)}):")
            for i, c in enumerate(resp.citations, 1):
                print(f"  [{i}] {c['doc_id']} [char {c['start_char']}-{c['end_char']}]")

    elif args.command == "interactive":
        p = Path(args.doc_dir)
        if p.exists():
            count = assistant.ingest_directory(p)
            print(f"Initialized assistant with {count} chunks from '{p}'.")
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_q = input("\nUser > ").strip()
                if not user_q or user_q.lower() in ["exit", "quit"]:
                    break
                resp = assistant.ask(user_q)
                print(f"\nAssistant [{resp.decision}]: {resp.answer}")
                if resp.citations:
                    print(f"  Citations: {[c['doc_id'] for c in resp.citations]}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")


if __name__ == "__main__":
    main()
