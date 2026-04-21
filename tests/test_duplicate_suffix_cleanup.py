from pathlib import Path

from scripts.clean_duplicate_suffix_files import (
    canonical_path_for_duplicate,
    cleanup_duplicate_suffix_files,
    scan_duplicate_suffix_files,
)


def test_canonical_path_for_duplicate():
    duplicate = Path("/tmp/train_sclerosis 2.py")
    assert canonical_path_for_duplicate(duplicate) == Path("/tmp/train_sclerosis.py")


def test_cleanup_duplicate_suffix_files_removes_identical_copies(tmp_path):
    canonical = tmp_path / "scripts" / "train_sclerosis.py"
    duplicate = tmp_path / "scripts" / "train_sclerosis 2.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("print('same')\n", encoding="utf-8")
    duplicate.write_text("print('same')\n", encoding="utf-8")

    removable, conflicts = scan_duplicate_suffix_files(tmp_path)
    assert removable == [(duplicate, canonical)]
    assert conflicts == []

    summary = cleanup_duplicate_suffix_files(tmp_path, remove=True)
    assert summary["removed"] == 1
    assert not duplicate.exists()


def test_cleanup_duplicate_suffix_files_reports_conflicts(tmp_path):
    canonical = tmp_path / "scripts" / "bootstrap.py"
    duplicate = tmp_path / "scripts" / "bootstrap 2.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("print('base')\n", encoding="utf-8")
    duplicate.write_text("print('different')\n", encoding="utf-8")

    removable, conflicts = scan_duplicate_suffix_files(tmp_path)
    assert removable == []
    assert conflicts == [(duplicate, "content differs from canonical file: bootstrap.py")]
