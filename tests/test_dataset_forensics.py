"""
Unit test suite for Dataset Forensic Analysis & Visual QA Engine (Phase C).
BORDER SENTINEL — SIH AI Surveillance Grid.

Verifies criteria C1 through C12:
- C1: Dataset Integrity Audit (clean vs corrupted/malformed labels, coordinate ranges)
- C2: Class Distribution Analysis & Percentages
- C3: Bounding Box Size Distributions & Percentiles
- C4: Tiny / Distant Object Detection (<32x32 px)
- C5: Crowded Frame Density & Percentiles
- C6: Edge-of-Frame Bounding Box Margin Detection
- C7: Per-Sequence Aggregation
- C8: Split Comparison & Distribution Shift
- C9: Rejected Annotations Accounting from Manifest
- C10: Visual QA Montage & Bounding Box Overlay Generation
- C11: Factual Bias & Risk Assessment
- C12: Actionable YOLO Training Recommendations
- Deterministic Sampling with Random Seeds
"""

import json
from pathlib import Path
import shutil
import sys
import unittest

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.dataset_forensics import DatasetForensics, ID_TO_CLASS


class TestDatasetForensics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = ROOT_DIR / "training_lab" / "datasets" / "_tmp_test_forensics"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.report_dir = ROOT_DIR / "training_lab" / "reports" / "_tmp_test_forensics_report"
        cls.report_dir.mkdir(parents=True, exist_ok=True)

        # Build synthetic dataset
        cls._create_synthetic_dataset()

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)
        if cls.report_dir.exists():
            shutil.rmtree(cls.report_dir, ignore_errors=True)

    @classmethod
    def _create_synthetic_dataset(cls):
        """Creates a controlled synthetic YOLO dataset with known properties."""
        img_w, img_h = 960, 540

        for split in ["train", "val", "test"]:
            (cls.test_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (cls.test_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Train sequence 1: MVI_20011 (3 frames)
        for f in range(1, 4):
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            img_name = f"dataset_v001_train_MVI_20011_f{f:05d}.jpg"
            lbl_name = f"dataset_v001_train_MVI_20011_f{f:05d}.txt"
            cv2.imwrite(str(cls.test_dir / "images" / "train" / img_name), img)

            # Frame 1: 1 normal car (ID 1), 1 tiny bus (ID 3, <32x32: 20x20 px -> 20/960=0.0208, 20/540=0.037)
            # Frame 2: 1 edge car touching left border (x=0.005, w=0.010 -> left margin = 0.0)
            # Frame 3: Crowded frame with 10 cars
            lbl_path = cls.test_dir / "labels" / "train" / lbl_name
            with open(lbl_path, "w", encoding="utf-8") as lf:
                if f == 1:
                    lf.write("1 0.500000 0.500000 0.100000 0.100000\n")  # 96x54 px (5184 px^2, medium)
                    lf.write("3 0.200000 0.200000 0.020833 0.037037\n")  # 20x20 px (400 px^2, tiny)
                elif f == 2:
                    lf.write("1 0.020000 0.500000 0.040000 0.100000\n")  # x1 = 0.0, touches left edge, 38.4x54 = 2073 px^2 (>1024)
                elif f == 3:
                    for k in range(10):
                        lf.write(f"1 {0.1 + k*0.08:.6f} 0.600000 0.050000 0.050000\n")

        # Train sequence 2: MVI_20012 (1 frame)
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        img_name = "dataset_v001_train_MVI_20012_f00001.jpg"
        lbl_name = "dataset_v001_train_MVI_20012_f00001.txt"
        cv2.imwrite(str(cls.test_dir / "images" / "train" / img_name), img)
        with open(cls.test_dir / "labels" / "train" / lbl_name, "w", encoding="utf-8") as lf:
            lf.write("1 0.400000 0.400000 0.200000 0.200000\n")  # 192x108 px (20736 px^2, large)

        # Val sequence: MVI_40131 (2 frames)
        for f in range(1, 3):
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            img_name = f"dataset_v001_val_MVI_40131_f{f:05d}.jpg"
            lbl_name = f"dataset_v001_val_MVI_40131_f{f:05d}.txt"
            cv2.imwrite(str(cls.test_dir / "images" / "val" / img_name), img)
            with open(cls.test_dir / "labels" / "val" / lbl_name, "w", encoding="utf-8") as lf:
                lf.write("1 0.500000 0.500000 0.080000 0.080000\n")
                if f == 2:
                    lf.write("3 0.800000 0.800000 0.150000 0.150000\n")

        # Test sequence: MVI_63521 (1 frame)
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        img_name = "dataset_v001_test_MVI_63521_f00001.jpg"
        lbl_name = "dataset_v001_test_MVI_63521_f00001.txt"
        cv2.imwrite(str(cls.test_dir / "images" / "test" / img_name), img)
        with open(cls.test_dir / "labels" / "test" / lbl_name, "w", encoding="utf-8") as lf:
            lf.write("1 0.300000 0.300000 0.050000 0.050000\n")

        # Manifest with source accounting and rejected annotations
        manifest = {
            "dataset_version": "dataset_v001",
            "total_frames": 7,
            "total_annotations": 18,
            "source_accounting": {
                "total_xml_frames": 10,
                "available_physical_frames": 7,
                "unannotated_physical_frames_skipped": 0,
                "missing_physical_frames_skipped": 3,
                "generated_annotated_frames": 7,
                "total_bounding_boxes_in_xml": 20,
                "rejected_annotations": 2,
                "rejected_annotations_breakdown": {
                    "by_class": {"car": 1, "bus": 1},
                    "by_sequence": {"MVI_20011": 2},
                    "reason": "bounding box dimensions < 10x10 px or degenerate after frame boundary clipping",
                },
            },
            "split_summary": {
                "train": {"sequences": 2, "frames": 4, "annotations": 14},
                "val": {"sequences": 1, "frames": 2, "annotations": 3},
                "test": {"sequences": 1, "frames": 1, "annotations": 1},
            },
        }
        with open(cls.test_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def setUp(self):
        self.forensics = DatasetForensics(
            dataset_path=self.test_dir,
            report_dir=self.report_dir,
            random_seed=42,
            edge_margin_px=5.0,
            tiny_area_px_threshold=1024.0,
        )

    def test_c1_integrity_clean_dataset(self):
        """Verify C1 passes on a clean dataset with 100% paired files and valid coordinates."""
        audit = self.forensics.audit_integrity()
        self.assertEqual(audit["status"], "PASSED")
        self.assertEqual(audit["total_images"], 7)
        self.assertEqual(audit["total_labels"], 7)
        self.assertEqual(audit["paired_images_labels"], 7)
        self.assertEqual(audit["missing_images"], 0)
        self.assertEqual(audit["missing_labels"], 0)
        self.assertEqual(audit["malformed_rows"], 0)
        self.assertEqual(audit["invalid_coordinate_ranges"], 0)
        self.assertEqual(audit["corrupted_images"], 0)
        self.assertEqual(audit["nan_or_inf_values"], 0)

    def test_c1_integrity_detects_malformed_and_unpaired(self):
        """Verify C1 correctly flags unparseable rows, invalid coordinates, and unpaired files."""
        bad_dir = ROOT_DIR / "training_lab" / "datasets" / "_tmp_bad_forensics"
        bad_dir.mkdir(parents=True, exist_ok=True)
        try:
            (bad_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
            (bad_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

            # 1. Unpaired image (no label)
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(bad_dir / "images" / "train" / "unpaired.jpg"), dummy_img)

            # 2. Corrupted coordinate row (> 1.0)
            cv2.imwrite(str(bad_dir / "images" / "train" / "bad_coord.jpg"), dummy_img)
            with open(bad_dir / "labels" / "train" / "bad_coord.txt", "w") as f:
                f.write("1 1.500000 0.500000 0.200000 0.200000\n")

            # 3. Malformed row (not 5 tokens)
            cv2.imwrite(str(bad_dir / "images" / "train" / "malformed.jpg"), dummy_img)
            with open(bad_dir / "labels" / "train" / "malformed.txt", "w") as f:
                f.write("1 0.5 0.5\n")

            # 4. NaN value
            cv2.imwrite(str(bad_dir / "images" / "train" / "nan_val.jpg"), dummy_img)
            with open(bad_dir / "labels" / "train" / "nan_val.txt", "w") as f:
                f.write("1 nan 0.5 0.2 0.2\n")

            forensics_bad = DatasetForensics(dataset_path=bad_dir)
            audit = forensics_bad.audit_integrity()

            self.assertEqual(audit["status"], "FAILED")
            self.assertGreater(audit["missing_labels"], 0)
            self.assertGreater(audit["invalid_coordinate_ranges"], 0)
            self.assertGreater(audit["malformed_rows"], 0)
            self.assertGreater(audit["nan_or_inf_values"], 0)
        finally:
            shutil.rmtree(bad_dir, ignore_errors=True)

    def test_c2_class_distribution(self):
        """Verify C2 counts instances and computes class balance accurately."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        dist = self.forensics.analyze_class_distribution(ann_by_split, meta_by_split)

        self.assertEqual(dist["total_bounding_boxes"], 18)
        # Train has: frame1 (1 car, 1 bus), frame2 (1 car), frame3 (10 cars), frame4 (1 car) -> 13 cars, 1 bus
        # Val has: frame1 (1 car), frame2 (1 car, 1 bus) -> 2 cars, 1 bus
        # Test has: frame1 (1 car) -> 1 car
        # Total cars = 13 + 2 + 1 = 16. Total buses = 1 + 1 = 2.
        self.assertEqual(dist["classes"]["car"]["total_count"], 16)
        self.assertEqual(dist["classes"]["bus"]["total_count"], 2)
        self.assertAlmostEqual(dist["classes"]["car"]["percentage_of_total"], (16 / 18) * 100, places=2)
        self.assertAlmostEqual(dist["classes"]["bus"]["percentage_of_total"], (2 / 18) * 100, places=2)

        # Surveillance classes with 0 instances must be represented with 0 count
        self.assertEqual(dist["classes"]["person"]["total_count"], 0)
        self.assertIn("person", dist["zero_instance_classes"])
        self.assertEqual(len(dist["active_classes"]), 2)

    def test_c3_bbox_statistics_and_percentiles(self):
        """Verify C3 accurately computes area, width, height, aspect ratio percentiles."""
        ann_by_split, _ = self.forensics._collect_annotations_and_meta()
        stats = self.forensics.analyze_bbox_sizes(ann_by_split)

        self.assertIn("percentiles_pixel", stats)
        self.assertIn("percentiles_normalized", stats)
        self.assertIn("size_categories", stats)

        p_px = stats["percentiles_pixel"]
        for metric in ["area", "width", "height", "aspect_ratio"]:
            self.assertIn(metric, p_px)
            for p in ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]:
                self.assertIn(p, p_px[metric])

        # Area percentiles must be monotonically increasing
        self.assertLessEqual(p_px["area"]["P1"], p_px["area"]["P50"])
        self.assertLessEqual(p_px["area"]["P50"], p_px["area"]["P99"])

        # Size categories sum up to total bounding boxes (18)
        cats = stats["size_categories"]["pixel"]
        total_in_cats = cats["tiny"]["count"] + cats["small"]["count"] + cats["medium"]["count"] + cats["large"]["count"]
        self.assertEqual(total_in_cats, 18)

    def test_c4_tiny_objects_detection(self):
        """Verify C4 detects tiny objects (<1024 px^2) and identifies affected sequences."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        tiny = self.forensics.analyze_tiny_objects(ann_by_split, meta_by_split)

        # Exactly 1 tiny bus was created in MVI_20011 (20x20 = 400 px^2 < 1024)
        self.assertEqual(tiny["tiny_objects_count"], 1)
        self.assertAlmostEqual(tiny["tiny_percentage_overall"], (1 / 18) * 100, places=2)
        self.assertEqual(tiny["breakdown_by_class"]["bus"]["count"], 1)
        self.assertIn("MVI_20011", tiny["affected_sequences"])

    def test_c5_crowded_frames(self):
        """Verify C5 calculates density per frame, identifying max crowded frame and percentiles."""
        _, meta_by_split = self.forensics._collect_annotations_and_meta()
        crowded = self.forensics.analyze_crowded_frames(meta_by_split)

        # Max crowded frame is f00003 of MVI_20011 with 10 cars
        self.assertEqual(crowded["max_density"], 10)
        self.assertIn("percentiles", crowded)
        self.assertIn("top_crowded_sequences", crowded)
        self.assertEqual(crowded["top_crowded_sequences"][0]["sequence_id"], "MVI_20011")

    def test_c6_edge_objects(self):
        """Verify C6 flags bounding boxes touching or within margin px of image borders."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        edge = self.forensics.analyze_edge_objects(ann_by_split, meta_by_split)

        # In train f2, we placed an edge box with x=0.005, w=0.010 -> x1 = 0.0 <= margin
        self.assertGreater(edge["edge_objects_count"], 0)
        self.assertGreater(edge["border_breakdown"]["left"], 0)

    def test_c7_per_sequence_distribution(self):
        """Verify C7 groups metrics per sequence accurately."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        seq_stats = self.forensics.analyze_per_sequence(ann_by_split, meta_by_split)

        self.assertEqual(seq_stats["total_sequences"], 4)  # MVI_20011, MVI_20012, MVI_40131, MVI_63521
        self.assertIn("MVI_20011", seq_stats["sequences"])
        mvi_11 = seq_stats["sequences"]["MVI_20011"]
        self.assertEqual(mvi_11["frame_count"], 3)
        self.assertEqual(mvi_11["total_objects"], 13)
        self.assertEqual(mvi_11["split"], "train")

    def test_c8_split_comparison(self):
        """Verify C8 evaluates distribution shift and class balance across splits."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        c2 = self.forensics.analyze_class_distribution(ann_by_split, meta_by_split)
        c3 = self.forensics.analyze_bbox_sizes(ann_by_split)
        c4 = self.forensics.analyze_tiny_objects(ann_by_split, meta_by_split)
        c6 = self.forensics.analyze_edge_objects(ann_by_split, meta_by_split)
        c5 = self.forensics.analyze_crowded_frames(meta_by_split)

        comp = self.forensics.compare_splits(ann_by_split, meta_by_split, c2, c3, c4, c6, c5)
        self.assertIn("splits", comp)
        self.assertIn("train", comp["splits"])
        self.assertIn("val", comp["splits"])
        self.assertIn("test", comp["splits"])
        self.assertEqual(comp["splits"]["train"]["frames"], 4)
        self.assertEqual(comp["splits"]["val"]["frames"], 2)
        self.assertEqual(comp["splits"]["test"]["frames"], 1)

    def test_c9_rejected_annotations(self):
        """Verify C9 loads and accounts for rejected annotations from the dataset manifest."""
        manifest_p = self.test_dir / "dataset_manifest.json"
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        rejected = self.forensics.analyze_rejected_annotations(manifest_data)
        self.assertEqual(rejected["total_rejected"], 2)
        self.assertEqual(rejected["breakdown_by_class"]["car"], 1)
        self.assertEqual(rejected["breakdown_by_class"]["bus"], 1)
        self.assertEqual(rejected["breakdown_by_sequence"]["MVI_20011"], 2)

    def test_c10_visual_qa_montage(self):
        """Verify C10 generates contact sheet and sample bounding box overlays."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        vqa = self.forensics.generate_visual_qa_montage(meta_by_split, ann_by_split, num_montage_items=4)

        self.assertIn("montage_path", vqa)
        montage_file = Path(vqa["montage_path"])
        self.assertTrue(montage_file.is_file())
        self.assertGreater(montage_file.stat().st_size, 0)
        self.assertGreater(len(vqa["sample_overlays"]), 0)

    def test_c10_regression_fewer_frames_than_num_montage_items(self):
        """Regression test for infinite loop bug: bounded sampling terminates when dataset has fewer frames than requested."""
        ann_by_split, meta_by_split = self.forensics._collect_annotations_and_meta()
        total_available_frames = sum(len(f) for f in meta_by_split.values())
        self.assertEqual(total_available_frames, 7)

        # Request 16 items on a dataset with only 7 frames (previously hung indefinitely)
        vqa = self.forensics.generate_visual_qa_montage(
            meta_by_split, ann_by_split, num_montage_items=16
        )

        self.assertIn("montage_path", vqa)
        montage_file = Path(vqa["montage_path"])
        self.assertTrue(montage_file.is_file())
        self.assertGreater(montage_file.stat().st_size, 0)
        self.assertGreater(len(vqa["sample_overlays"]), 0)
        self.assertLessEqual(len(vqa["sample_overlays"]), 7)
        self.assertLessEqual(vqa["samples_count"], 7)

    def test_c11_and_c12_reports_and_recommendations(self):
        """Verify C11 facts vs inferences separation and C12 concrete YOLO hyperparameters."""
        results = self.forensics.run_full_forensics(generate_visual_qa=True)

        c11 = results["c11_bias_and_risk"]
        self.assertIn("observed_facts", c11)
        self.assertIn("inferences_and_risks", c11)
        self.assertGreater(len(c11["observed_facts"]), 0)

        c12 = results["c12_training_recommendations"]
        self.assertIn("imgsz", c12)
        self.assertIn("batch_size", c12)
        self.assertIn("loss_weights", c12)
        self.assertIn("augmentation_policy", c12)

        # Check that artifacts exist on disk
        self.assertTrue((self.report_dir / "DATASET_FORENSIC_REPORT.md").is_file())
        self.assertTrue((self.report_dir / "dataset_summary.json").is_file())
        self.assertTrue((self.report_dir / "class_distribution.json").is_file())
        self.assertTrue((self.report_dir / "bbox_percentiles.json").is_file())


if __name__ == "__main__":
    unittest.main()
