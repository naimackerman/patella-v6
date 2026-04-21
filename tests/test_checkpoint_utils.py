from pathlib import Path

import torch

from src.utils.checkpoints import checkpoint_score, find_best_lightning_checkpoint


def test_find_best_lightning_checkpoint_reads_stored_scores(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    low_path = ckpt_dir / "epoch-low.ckpt"
    high_path = ckpt_dir / "epoch-high.ckpt"

    torch.save(
        {
            "callbacks": {
                "ModelCheckpoint": {
                    "monitor": "val_qwk",
                    "current_score": torch.tensor(0.51),
                    "best_model_score": torch.tensor(0.51),
                }
            }
        },
        low_path,
    )
    torch.save(
        {
            "callbacks": {
                "ModelCheckpoint": {
                    "monitor": "val_qwk",
                    "current_score": torch.tensor(0.73),
                    "best_model_score": torch.tensor(0.73),
                }
            }
        },
        high_path,
    )

    assert checkpoint_score(low_path, monitor="val_qwk") == 0.51
    assert find_best_lightning_checkpoint(ckpt_dir, monitor="val_qwk") == high_path
