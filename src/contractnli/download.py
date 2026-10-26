"""Pinned upstream dataset, HTTPS and SHA256 checked before extraction."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
URL = f"https://raw.githubusercontent.com/stanfordnlp/contract-nli/{COMMIT}/resources/contract-nli.zip"
SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
FILES = ("train.json", "dev.json", "test.json", "LICENSE", "TERMS", "README.md")


def download(destination: str | Path, archive_path: str | Path | None = None) -> Path:
    if archive_path:
        raw = Path(archive_path).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Dataset archive checksum mismatch; nothing extracted")
    destination = Path(destination)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        # Fixed member list avoids extracting raw PDFs or arbitrary archive paths.
        contents = {name: archive.read(f"contract-nli/{name}") for name in FILES}
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite modified dataset file: {path}")
    for name, content in contents.items():
        (destination / name).write_bytes(content)
    return destination
