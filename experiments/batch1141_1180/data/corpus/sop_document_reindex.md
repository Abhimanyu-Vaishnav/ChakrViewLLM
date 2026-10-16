# Document Ingestion and Re-Indexing Procedure

## 1. Re-indexing Steps
- Step 1: Scan target directory for files matching `.txt` and `.md` extensions.
- Step 2: Compare file modification time and SHA-256 hash against metadata table.
- Step 3: Delete orphaned chunks for modified or removed files.
- Step 4: Re-chunk modified files at 800-character boundaries with 100-character overlap.
