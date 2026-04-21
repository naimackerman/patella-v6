import unittest

import torch
from omegaconf import OmegaConf

from src.models.sclerosis_classifier import SclerosisClassifier


def _base_cfg(input_mode: str):
    return OmegaConf.create({
        "input_mode": input_mode,
        "cnn_backbone": "efficientnet_b0",
        "pretrained": False,
        "in_channels": 1,
        "num_classes": 3,
        "texture_feature_dim": 65,
        "side_embedding_dim": 4,
        "fusion_hidden_dim": 16,
        "use_side_specific_heads": False,
        "side_head_hidden_dim": 8,
        "dropout_cnn": 0.0,
        "dropout_fusion": 0.0,
    })


class TestSclerosisClassifierModes(unittest.TestCase):
    def test_texture_only_forward_does_not_require_cnn_branch(self):
        model = SclerosisClassifier(_base_cfg("texture_only"))
        images = torch.zeros(2, 1, 96, 96)
        textures = torch.zeros(2, 65)
        side_ids = torch.tensor([0, 1])

        logits = model(images, textures, side_ids)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertIsNone(model.cnn)

    def test_hybrid_forward_keeps_expected_output_shape(self):
        model = SclerosisClassifier(_base_cfg("hybrid"))
        images = torch.zeros(2, 1, 96, 96)
        textures = torch.zeros(2, 65)
        side_ids = torch.tensor([0, 1])

        logits = model(images, textures, side_ids)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_binary_num_classes_changes_output_shape(self):
        cfg = _base_cfg("hybrid")
        cfg.num_classes = 2
        model = SclerosisClassifier(cfg)
        images = torch.zeros(2, 1, 96, 96)
        textures = torch.zeros(2, 65)
        side_ids = torch.tensor([0, 1])

        logits = model(images, textures, side_ids)

        self.assertEqual(tuple(logits.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
