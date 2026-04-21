"""Remove duplicate copied files whose names end with a numeric suffix."""

from __future__ import annotations

import argparse
import filecmp
import os
import re
from pathlib import Path


DUPLICATE_SUFFIX_RE = re.compile(r"^(?P<stem>.+?) (?P<copy>\d+)(?P<suffix>\.[^.]+)$")
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "KneeXrayData",
    "checkpoints",
    "features",
    "outputs",
    "results",
}


def canonical_path_for_duplicate(path: Path) -> Path | None:
    """Map `foo 2.py` to `foo.py`."""
    match = DUPLICATE_SUFFIX_RE.match(path.name)
    if match is None:
        return None
    return path.with_name(f"{match.group('stem')}{match.group('suffix')}")


def scan_duplicate_suffix_files(project_root: str | Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, str]]]:
    """Return removable duplicates and conflicts under a project root."""
    project_root = Path(project_root).resolve()
    removable: list[tuple[Path, Path]] = []
    conflicts: list[tuple[Path, str]] = []

    for path in _iter_candidate_files(project_root):
        canonical = canonical_path_for_duplicate(path)
        if canonical is None:
            continue
        if not canonical.exists():
            conflicts.append((path, f"missing canonical file: {canonical.name}"))
            continue
        if not filecmp.cmp(path, canonical, shallow=False):
            conflicts.append((path, f"content differs from canonical file: {canonical.name}"))
            continue
        removable.append((path, canonical))

    return removable, conflicts


def _iter_candidate_files(project_root: Path):
    for root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        root_path = Path(root)
        for filename in sorted(filenames):
            yield root_path / filename


def cleanup_duplicate_suffix_files(project_root: str | Path, remove: bool = False) -> dict[str, int]:
    """Delete safe duplicates when requested and return a summary."""
    removable, conflicts = scan_duplicate_suffix_files(project_root)

    for path, canonical in removable:
        print(f"[duplicate] {path} -> {canonical}")
        if remove:
            path.unlink()

    for path, reason in conflicts:
        print(f"[conflict]  {path} ({reason})")

    return {
        "removed": len(removable) if remove else 0,
        "removable": len(removable),
        "conflicts": len(conflicts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root to scan.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Delete safe numbered duplicates instead of only listing them.",
    )
    args = parser.parse_args()

    summary = cleanup_duplicate_suffix_files(args.project_root, remove=args.remove)
    action = "Removed" if args.remove else "Found"
    print(
        f"{action} {summary['removed'] if args.remove else summary['removable']} safe duplicates; "
        f"{summary['conflicts']} conflicts remain."
    )
    if summary["conflicts"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
