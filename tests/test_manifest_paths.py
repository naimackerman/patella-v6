from pathlib import Path

from src.utils.manifest_paths import resolve_manifest_path, to_repo_local_manifest_path


def test_manifest_paths_round_trip_inside_repo(tmp_path):
    project_root = tmp_path / "patella-v6"
    image_path = project_root / "KneeXrayData" / "ClsKLData" / "kneeKL224" / "train" / "0" / "9001695L.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")

    manifest_path = to_repo_local_manifest_path(
        image_path,
        project_root=project_root,
        data_root=project_root / "KneeXrayData",
    )

    assert manifest_path == "KneeXrayData/ClsKLData/kneeKL224/train/0/9001695L.png"
    assert resolve_manifest_path(
        manifest_path,
        project_root=project_root,
        data_root=project_root / "KneeXrayData",
    ) == image_path.resolve()


def test_manifest_paths_map_external_data_root_back_to_repo_layout(tmp_path):
    project_root = tmp_path / "patella-v6"
    external_data_root = tmp_path / "shared-data"
    image_path = external_data_root / "ClsKLData" / "kneeKL224" / "val" / "2" / "9804376R.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")

    manifest_path = to_repo_local_manifest_path(
        image_path,
        project_root=project_root,
        data_root=external_data_root,
    )

    assert manifest_path == "KneeXrayData/ClsKLData/kneeKL224/val/2/9804376R.png"
    assert resolve_manifest_path(
        manifest_path,
        project_root=project_root,
        data_root=external_data_root,
    ) == image_path.resolve()
