"""Freeze metadata only; no model weights or inference are needed."""
from dataclasses import replace
from pathlib import Path
import re

from .config import load_config
from .storage import write_json


def freeze_config(config, destination):
    from transformers import AutoConfig

    destination = Path(destination)
    if destination.exists():
        previous = load_config(destination)
        a, b = config.to_dict(), previous.to_dict()
        a.pop('require_pinned_revisions')
        b.pop('require_pinned_revisions')
        for role in ('summarizer', 'judge', 'compressor'):
            a[role].pop('revision')
            b[role].pop('revision')
            requested = getattr(config, role).revision
            if re.fullmatch(r'[0-9a-f]{40}', requested) and requested != getattr(previous, role).revision:
                raise ValueError('Explicit model revision differs from frozen config')
        if a != b or not previous.require_pinned_revisions:
            raise ValueError('Frozen config differs from template. Use a new file and output directory.')
        return previous
    roles = {}
    for role in ('summarizer', 'judge', 'compressor'):
        model = getattr(config, role)
        if re.fullmatch(r'[0-9a-f]{40}', model.revision):
            commit = model.revision
        else:
            try:
                metadata = AutoConfig.from_pretrained(model.name, revision=model.revision,
                                                      local_files_only=True, trust_remote_code=False)
            except OSError:
                metadata = AutoConfig.from_pretrained(model.name, revision=model.revision, trust_remote_code=False)
            commit = getattr(metadata, '_commit_hash', None)
        if not commit or not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError(f'Cannot resolve immutable model revision: {role}')
        roles[role] = replace(model, revision=commit)
    frozen = replace(config, **roles, require_pinned_revisions=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, frozen.to_dict())
    return frozen
