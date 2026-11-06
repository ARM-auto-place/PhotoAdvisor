"""Shared source provenance rules; no collectors, network or ML dependencies."""

import hashlib
import re


def author_group(record):
    author = record.get('author_url') or record['author']
    if any(t in author.lower() for t in ('hightechdad', 'michael sheehan')):
        author = 'hightechdad'
    return 'source-group-' + hashlib.sha256(author.strip().lower().encode()).hexdigest()[:16]


def allowed_license(name):
    return bool(re.fullmatch(r'CC BY(?:-SA)? [1-4]\.0|CC0|Public domain', name))
