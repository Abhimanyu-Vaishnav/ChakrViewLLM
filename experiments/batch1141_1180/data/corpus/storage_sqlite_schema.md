# Metadata Database Schema

## 1. Schema Tables
- Table `documents`: Tracks doc_id, file_path, file_size_bytes, mtime, sha256.
- Table `chunks`: Tracks chunk_id, doc_id, chunk_index, start_char, end_char, chunk_hash.
- WAL mode: Write-Ahead Logging enabled with synchronous NORMAL.
