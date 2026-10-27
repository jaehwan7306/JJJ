from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

LABELS = ("Entailment", "Contradiction", "NotMentioned")
EVALUATION_LABELS = LABELS


def load_for_config(config):
    dataset = load_dataset(config.dataset, config.max_documents)
    if config.expected_dataset_sha256 is not None and dataset.sha256 != config.expected_dataset_sha256:
        raise ValueError("Dataset SHA256 differs from the frozen protocol; refusing altered data")
    return dataset


@dataclass(frozen=True)
class Example:
    document_id: str
    hypothesis_id: str
    hypothesis: str
    gold: str
    evidence: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.document_id}/{self.hypothesis_id}"


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class Dataset:
    documents: tuple[Document, ...]
    sha256: str
    total_pairs: int
    not_mentioned_pairs: int

    @property
    def examples(self) -> tuple[Example, ...]:
        return tuple(e for d in self.documents for e in d.examples)


def load_dataset(path: str | Path, max_documents: int | None = None) -> Dataset:
    """Keep all three gold labels; evidence is absent for NotMentioned, never invented."""
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    raw = Path(path).read_bytes()
    content = json.loads(raw)
    documents = []
    total = not_mentioned = 0
    seen = set()
    for entry in content["documents"][:max_documents]:
        doc_id = str(entry["id"])
        if doc_id in seen:
            raise ValueError(f"Duplicate document id: {doc_id}")
        seen.add(doc_id)
        text = entry["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty/non-text document: {doc_id}")
        sets = entry["annotation_sets"]
        if len(sets) != 1:
            raise ValueError("Multiple annotation sets require an explicit aggregation policy")
        examples = []
        for hid, annotation in sets[0]["annotations"].items():
            label = annotation["choice"]
            if label not in LABELS:
                raise ValueError(f"Unknown label: {label}")
            total += 1
            if label == "NotMentioned":
                not_mentioned += 1
            evidence = []
            for index in annotation["spans"]:
                if not isinstance(index, int) or not 0 <= index < len(entry["spans"]):
                    raise ValueError(f"Invalid evidence index in {doc_id}/{hid}")
                start, end = entry["spans"][index]
                if not 0 <= start < end <= len(text):
                    raise ValueError(f"Invalid evidence offsets in {doc_id}/{hid}")
                evidence.append(text[start:end])
            examples.append(Example(doc_id, hid, content["labels"][hid]["hypothesis"],
                                    label, tuple(evidence)))
        documents.append(Document(doc_id, text, tuple(examples)))
    return Dataset(tuple(documents), hashlib.sha256(raw).hexdigest(), total, not_mentioned)
