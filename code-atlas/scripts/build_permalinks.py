#!/usr/bin/env python3
"""Utilities for building GitHub permalinks pinned to commit SHAs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True)
class RepoRemote:
    origin: str
    github_base: str


def normalize_github_origin(origin: str) -> RepoRemote:
    """Normalize git origin URL to https://github.com/org/repo."""
    origin = origin.strip()

    if origin.startswith("git@github.com:"):
        slug = origin.split("git@github.com:", 1)[1]
    elif origin.startswith("ssh://git@github.com/"):
        slug = origin.split("ssh://git@github.com/", 1)[1]
    elif origin.startswith("https://github.com/"):
        slug = origin.split("https://github.com/", 1)[1]
    elif origin.startswith("http://github.com/"):
        slug = origin.split("http://github.com/", 1)[1]
    else:
        raise ValueError(f"Unsupported/non-GitHub origin: {origin}")

    if slug.endswith(".git"):
        slug = slug[: -len(".git")]

    slug = re.sub(r"^/+", "", slug)
    slug = re.sub(r"/+$", "", slug)

    if "/" not in slug:
        raise ValueError(f"Invalid GitHub slug from origin: {origin}")

    return RepoRemote(origin=origin, github_base=f"https://github.com/{slug}")


def build_permalink(github_base: str, ref: str, path: str, start: int, end: int) -> str:
    path = path.lstrip("/")
    if start <= 0 or end <= 0:
        raise ValueError("Line numbers must be 1-based positive integers")
    if end < start:
        start, end = end, start

    # Keep path separators, but encode special chars in each segment.
    path_escaped = "/".join(quote(seg, safe="._-") for seg in path.split("/"))
    # Keep slash in ref so branch names like "feature/foo" remain readable/valid.
    ref_escaped = quote(ref, safe="/._-")
    anchor = f"#L{start}" if start == end else f"#L{start}-L{end}"
    return f"{github_base}/blob/{ref_escaped}/{path_escaped}{anchor}"


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build a GitHub permalink")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    remote = normalize_github_origin(args.origin)
    url = build_permalink(remote.github_base, args.ref, args.path, args.start, args.end)
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
