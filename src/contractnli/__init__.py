"""ContractNLI compression experiments. Heavy inference dependencies are optional."""

from pathlib import Path

# Jupyter can keep old class definitions after editable source files change.
# Refuse to fingerprint those definitions as if the newly saved files ran.
SOURCE_AT_IMPORT = {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")}
