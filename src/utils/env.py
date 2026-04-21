"""Small helpers for loading repo-local .env files without extra dependencies."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def load_repo_env(project_root: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load a simple .env file from the repo root into os.environ."""
    project_root = Path(project_root).resolve()
    env_path = project_root / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()

        key, sep, raw_value = line.partition("=")
        if not sep:
            continue

        key = key.strip()
        if not key:
            continue

        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
        value = " ".join(tokens).strip()

        if not override and key in os.environ and str(os.environ[key]).strip() != "":
            continue

        os.environ[key] = value
        loaded[key] = value

    return loaded
