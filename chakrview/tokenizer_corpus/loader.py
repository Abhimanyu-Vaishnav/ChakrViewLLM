"""
ChakrView Tokenizer Corpus Pipeline: Loader Module.

Loads text corpus files from data/tokenizer_corpus/ into memory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


@dataclass(frozen=True)
class CorpusItem:
    """Represents a single document/line item in the tokenizer corpus."""
    text: str
    category: str
    source_file: str
    line_number: int


def load_category(file_path: Union[str, Path], category_name: str) -> List[CorpusItem]:
    """
    Load a single category file into a list of CorpusItems.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    items: List[CorpusItem] = []
    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if line.strip():  # Skip empty or whitespace-only lines
            items.append(
                CorpusItem(
                    text=line,
                    category=category_name,
                    source_file=path.name,
                    line_number=line_idx,
                )
            )
    return items


def load_corpus(corpus_dir: Union[str, Path]) -> Dict[str, List[CorpusItem]]:
    """
    Load all .txt files from the designated corpus directory.
    Returns:
        Mapping from category name (e.g. 'hindi', 'english') to list of CorpusItems.
    """
    path = Path(corpus_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {path}")

    corpus: Dict[str, List[CorpusItem]] = {}
    for txt_file in sorted(path.glob("*.txt")):
        category = txt_file.stem
        items = load_category(txt_file, category)
        corpus[category] = items

    return corpus
