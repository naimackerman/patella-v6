import tempfile
import unittest
from pathlib import Path

from src.utils.pipeline_readiness import assess_pipeline_readiness


CORE_FRAMEWORK_FILES = [
    "src/models/roi_detector.py",
    "src/models/jsn_segmenter.py",
    "src/models/osteophyte_grader.py",
    "src/models/sclerosis_classifier.py",
    "src/features/feature_aggregator.py",
    "src/xai/overlay_renderer.py",
    "src/xai/report_generator.py",
    "app/inference.py",
    "app/gradio_app.py",
    "app/api.py",
    "scripts/export_pdf_report.py",
]


class PipelineReadinessTests(unittest.TestCase):
    def _touch(self, root: Path, relative_path: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _prepare_framework_scaffold(self, root: Path) -> None:
        for relative_path in CORE_FRAMEWORK_FILES:
            self._touch(root, relative_path)

    def test_json_xgboost_artifact_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._prepare_framework_scaffold(root)
            self._touch(root, "checkpoints/kl_xgboost.json")

            report = assess_pipeline_readiness(root)
            kl_stage = next(stage for stage in report["study_pipeline"] if stage["name"] == "kl_models")
            xgb_artifact = next(
                artifact for artifact in kl_stage["artifacts"] if artifact["label"] == "Transparent XGBoost model"
            )

            self.assertTrue(xgb_artifact["present"])
            self.assertEqual(xgb_artifact["path"], "checkpoints/kl_xgboost.json")

    def test_roi_stage_accepts_reproduce_roi_outputs_without_detector_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._prepare_framework_scaffold(root)
            self._touch(root, "features/rois_osteophyte_clahe_full/train/example_medial_femur.png")
            self._touch(root, "features/rois_osteophyte_clahe_full/val/example_medial_femur.png")
            self._touch(root, "features/rois_osteophyte_clahe_full/test/example_medial_femur.png")

            report = assess_pipeline_readiness(root)
            roi_stage = next(stage for stage in report["study_pipeline"] if stage["name"] == "roi_pipeline")

            self.assertEqual(roi_stage["status"], "ready")

    def test_missing_roi_stage_recommends_reproduce_roi_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._prepare_framework_scaffold(root)

            report = assess_pipeline_readiness(root)

            self.assertIn(
                (
                    "PYTHONPATH=. python3 scripts/train_jsn_segmenter.py training.label_mode=manual +model=unetpp"
                ),
                report["recommended_commands"],
            )
            self.assertIn(
                (
                    "PYTHONPATH=. python3 scripts/extract_jsn_features.py training.label_mode=manual +model=unetpp"
                ),
                report["recommended_commands"],
            )
            self.assertIn(
                (
                    "PYTHONPATH=. python3 scripts/extract_roi_with_fullimage_clahe.py "
                    "--input-dir KneeXrayData/ClsKLData/kneeKL224 "
                    "--output-dir features/rois_osteophyte_clahe_full "
                    "--scan-all-images "
                    "--reviewed-jsn-mask-dir annotations/jsn_masks "
                    "--predicted-jsn-mask-dir features/jsn/masks "
                    "--clahe-clip 3.0 "
                    "--clahe-tile 8"
                ),
                report["recommended_commands"],
            )
            self.assertTrue(
                any("features/rois_osteophyte_clahe_full/" in blocker for blocker in report["blockers"])
            )

    def test_missing_osteophyte_stage_recommends_main_manual_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._prepare_framework_scaffold(root)

            report = assess_pipeline_readiness(root)

            self.assertIn(
                (
                    "PYTHONPATH=. python3 scripts/train_osteophyte_grader.py "
                    "training.label_mode=manual "
                    "preprocessing.augmentation.horizontal_flip_p=0.0 "
                    "preprocessing.clahe=null "
                    "preprocessing.histogram_clip=null "
                    "training.scheduler=reduce_on_plateau "
                    "training.scheduler_params.mode=max "
                    "training.scheduler_params.factor=0.5 "
                    "training.scheduler_params.patience=10 "
                    "training.layer_wise_lr.enabled=true "
                    "training.layer_wise_lr.backbone_ratio=0.1 "
                    "training.osteophyte_class_balance.enabled=true "
                    "training.osteophyte_sampling.strategy=mean_class_balance "
                    "training.osteophyte_refinement.sites='[lateral_femur,medial_tibia]' "
                    "training.osteophyte_warm_start_checkpoint=checkpoints/clahe_fullimage_ordinal/hybrid-epoch=078-val_kappa_mean=0.8264.ckpt "
                    "training.osteophyte_force_retrain_multitask=true "
                    "osteophyte_roi_dir=features/rois_osteophyte_clahe_full"
                ),
                report["recommended_commands"],
            )
            self.assertIn(
                (
                    "PYTHONPATH=. python3 scripts/extract_osteophyte_features.py "
                    "training.label_mode=manual "
                    "preprocessing.clahe=null "
                    "preprocessing.histogram_clip=null "
                    "osteophyte_roi_dir=features/rois_osteophyte_clahe_full"
                ),
                report["recommended_commands"],
            )
            self.assertIn(
                (
                    "PYTHONPATH=. python3 scripts/evaluate_osteophyte_grader.py "
                    "training.label_mode=manual "
                    "preprocessing.clahe=null "
                    "preprocessing.histogram_clip=null "
                    "osteophyte_roi_dir=features/rois_osteophyte_clahe_full "
                    "result_dir=results/osteophyte_main_manual"
                ),
                report["recommended_commands"],
            )


if __name__ == "__main__":
    unittest.main()
