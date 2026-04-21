"""Refresh manual-review manifests without re-running the full bootstrap pipeline."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts.bootstrap_pseudo_labels import save_annotation_manifests, scan_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    data_root = Path(cfg.data.root)
    annotation_dir = Path(cfg.annotation_dir)
    project_root = Path(cfg.project_root)

    splits = scan_dataset(data_root, project_root=project_root)
    save_annotation_manifests(splits, annotation_dir)


if __name__ == "__main__":
    main()
