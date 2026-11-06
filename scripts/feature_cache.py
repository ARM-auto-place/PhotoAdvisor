"""Integrity-checked offline tensors; never cache test inference for head selection."""

import hashlib
import json
from pathlib import Path
import secrets

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_cache(path, metadata, tensors):
    path = Path(path)
    if path.exists() or path.with_suffix(".sha256").exists():
        raise ValueError("new feature cache path required")
    if any("test" in key.lower() for key in tensors):
        raise ValueError("test tensors cannot enter development feature cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(6))
    torch.save(dict(metadata=metadata, tensors=tensors), temporary)
    checksum = digest(temporary)
    temporary.replace(path)
    path.with_suffix(".sha256").write_text(checksum, encoding="ascii")
    path.with_suffix(".json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
    )


def read_cache(path, metadata, schema, targets=None):
    path = Path(path)
    if digest(path) != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("feature cache checksum mismatch")
    stored = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    if stored.get("metadata") != metadata:
        raise ValueError("feature cache provenance mismatch")
    tensors = stored.get("tensors", {})
    if set(tensors) != set(schema) or any("test" in key.lower() for key in tensors):
        raise ValueError("feature cache split/key mismatch")
    for key, (shape, dtype) in schema.items():
        value = tensors[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype
        ):
            raise ValueError("feature cache tensor shape/dtype mismatch")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("feature cache nonfinite features")
    for key, expected in (targets or {}).items():
        if not torch.equal(tensors[key], expected):
            raise ValueError("feature cache target order mismatch")
    return tensors
