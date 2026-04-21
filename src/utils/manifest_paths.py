"""Portable path helpers for annotation manifests and packages."""

from __future__ import annotations

import os
from pathlib import Path


REPO_DATA_DIRNAME = "KneeXrayData"


def to_repo_local_manifest_path(
    image_path: str | Path,
    project_root: str | Path,
    data_root: str | Path | None = None,
    repo_data_dirname: str = REPO_DATA_DIRNAME,
) -> str:
    """Serialize an image path into a repo-local manifest path when possible."""
    image_path = Path(image_path).resolve()
    project_root = Path(project_root).resolve()

    try:
        return image_path.relative_to(project_root).as_posix()
    except ValueError:
        pass

    if data_root is not None:
        data_root = Path(data_root).resolve()
        try:
            relative_to_data = image_path.relative_to(data_root)
            return (Path(repo_data_dirname) / relative_to_data).as_posix()
        except ValueError:
            pass

    return os.path.relpath(str(image_path), str(project_root))


def resolve_manifest_path(
    manifest_path: str | Path,
    project_root: str | Path,
    data_root: str | Path | None = None,
    repo_data_dirname: str = REPO_DATA_DIRNAME,
) -> Path:
    """Resolve a manifest path to a local file path on the current device."""
    raw_path = Path(str(manifest_path))
    if raw_path.is_absolute():
        return raw_path.resolve()

    project_root = Path(project_root).resolve()
    repo_candidate = (project_root / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    if data_root is not None:
        data_root = Path(data_root).resolve()
        parts = raw_path.parts
        if parts and parts[0] == repo_data_dirname:
            data_candidate = (data_root / Path(*parts[1:])).resolve()
        else:
            data_candidate = (data_root / raw_path).resolve()
        if data_candidate.exists():
            return data_candidate
        return data_candidate

    return repo_candidate
