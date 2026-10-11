"""
ChakrView Corpus Pipeline: Document Loader Module.

Loads multi-domain research corpora across categories:
- Hindi
- Sanskrit
- English
- Hinglish
- Mixed Indian-language text
- Programming Code
- Mathematics / technical formulas
- Numbers and structured items

Provides memory-conscious streaming loaders to avoid loading large files
into memory all at once.
"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass(frozen=True)
class CorpusDocument:
    """Represents a validated, tracked document in the ChakrView corpus pipeline."""
    text: str
    category: str
    subcategory: str
    source_file: str
    line_number: int
    byte_count: int
    char_count: int
    doc_type: str  # 'raw', 'cleaned', 'benchmark', 'validation', 'adversarial'
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_file_documents(
    file_path: Union[str, Path],
    category: str,
    subcategory: str = "general",
    doc_type: str = "raw",
) -> List[CorpusDocument]:
    """
    Load a text file line-by-line into a list of CorpusDocument records.
    Preserves exact string content.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    docs: List[CorpusDocument] = []
    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        line_bytes = line.encode("utf-8")
        doc_hash = hashlib.sha256(line_bytes).hexdigest()
        docs.append(
            CorpusDocument(
                text=line,
                category=category,
                subcategory=subcategory,
                source_file=path.name,
                line_number=line_idx,
                byte_count=len(line_bytes),
                char_count=len(line),
                doc_type=doc_type,
                sha256=doc_hash,
            )
        )
    return docs


def load_corpus_tree(
    root_dir: Union[str, Path],
    doc_type: str = "raw",
) -> Dict[str, List[CorpusDocument]]:
    """
    Recursively load all text files from a category directory structure.
    Expected structure: root_dir / <category> / <filename>.txt
    """
    p = Path(root_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {p}")

    corpus: Dict[str, List[CorpusDocument]] = {}

    for entry in sorted(p.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            category = entry.name
            cat_docs: List[CorpusDocument] = []
            for txt_file in sorted(entry.glob("*.txt")):
                subcat = txt_file.stem
                docs = load_file_documents(
                    file_path=txt_file,
                    category=category,
                    subcategory=subcat,
                    doc_type=doc_type,
                )
                cat_docs.extend(docs)
            if cat_docs:
                corpus[category] = cat_docs

    return corpus


def stream_corpus_documents(
    root_dir: Union[str, Path],
    chunk_size: int = 500,
    doc_type: str = "raw",
) -> Iterator[List[CorpusDocument]]:
    """
    Memory-conscious document streaming generator.
    Yields batches of CorpusDocument records without reading the entire dataset into RAM.
    """
    p = Path(root_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {p}")

    batch: List[CorpusDocument] = []
    for cat_dir in sorted(p.iterdir()):
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            category = cat_dir.name
            for txt_file in sorted(cat_dir.glob("*.txt")):
                subcat = txt_file.stem
                with txt_file.open("r", encoding="utf-8", errors="replace") as f:
                    for line_idx, line in enumerate(f, start=1):
                        cleaned = line.rstrip("\r\n")
                        if not cleaned.strip():
                            continue
                        line_bytes = cleaned.encode("utf-8")
                        doc_hash = hashlib.sha256(line_bytes).hexdigest()
                        doc = CorpusDocument(
                            text=cleaned,
                            category=category,
                            subcategory=subcat,
                            source_file=txt_file.name,
                            line_number=line_idx,
                            byte_count=len(line_bytes),
                            char_count=len(cleaned),
                            doc_type=doc_type,
                            sha256=doc_hash,
                        )
                        batch.append(doc)
                        if len(batch) >= chunk_size:
                            yield batch
                            batch = []
    if batch:
        yield batch
