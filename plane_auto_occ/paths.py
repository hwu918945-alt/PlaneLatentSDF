"""Resolve moved sampling artifacts by locked digest, without rewriting weights."""
import hashlib
from pathlib import Path


def resolve_sampling_pool(checkpoint, reference):
    checkpoint = Path(checkpoint).resolve()
    name = Path(reference['path']).name
    candidates = []
    for parent in list(checkpoint.parents)[:6]:
        candidates.extend((parent/'sampling_pools'/name, parent/'pools'/name,
                           parent/'outputs/fit/sampling_pools'/name))
    candidates.append(Path(reference['path']))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == reference['sha256']:
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot locate digest-matching sampling pool {name} near {checkpoint}')
