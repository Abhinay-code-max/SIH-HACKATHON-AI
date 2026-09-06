"""
Comprehensive Test Suite for UA-DETRAC Dataset Ingestion & Pipeline Integrity.
Verifies all bug fixes, boundary clipping, class mappings, multi-object frame grouping,
leakage-free splitting, and real-image provider integration.
"""

from pathlib import Path
import shutil
import sys
import unittest

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.dataset_generator import DataQualityGate, DatasetGenerator, load_ua_detrac_split_config
from training_lab.engine.dataset_importer import DatasetFormat, DatasetImporter, UADetracFrameProvider
from training_lab.engine.leakage_detector import DataLeakageDetector

UA_DETRAC_ROOT = Path(r"C:\Users\Abhinay Kandrika\OneDrive\Desktop\test dataset")
UA_DETRAC_XML = UA_DETRAC_ROOT / "DETRAC-Train-Annotations" / "DETRAC-Train-Annotations-XML" / "MVI_20011.xml"
UA_DETRAC_IMAGES = UA_DETRAC_ROOT / "DETRAC-Images"


class TestUADetracPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = ROOT_DIR / "training_lab" / "datasets" / "_tmp_test_ua_detrac"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.importer = DatasetImporter(datasets_dir=cls.test_dir)
        cls.generator = DatasetGenerator(datasets_dir=cls.test_dir)
        cls.leakage_detector = DataLeakageDetector(datasets_dir=cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_class_mapping(self):
        """Verify UA-DETRAC vehicle classes map correctly to master classes."""
        self.assertEqual(self.importer.map_class("car", DatasetFormat.UA_DETRAC), "car")
        self.assertEqual(self.importer.map_class("van", DatasetFormat.UA_DETRAC), "car")
        self.assertEqual(self.importer.map_class("bus", DatasetFormat.UA_DETRAC), "bus")
        self.assertEqual(self.importer.map_class("others", DatasetFormat.UA_DETRAC), "car")
        self.assertEqual(self.importer.map_class("truck", DatasetFormat.UA_DETRAC), "truck")

    def test_02_unknown_class_rejection(self):
        """Verify unmapped/unknown classes are NOT silently converted to person (class 0)."""
        # 1. Importer mapping returns 'unknown'
        mapped = self.importer.map_class("alien_uav", DatasetFormat.UA_DETRAC)
        self.assertEqual(mapped, "unknown")

        # 2. Generator resolve_class_id raises ValueError
        with self.assertRaises(ValueError):
            self.generator.resolve_class_id("alien_uav")

        # 3. DataQualityGate rejects unknown class
        is_valid, issues, _ = DataQualityGate.audit_and_clean_annotation(
            bbox=[100, 100, 200, 200],
            image_shape=(540, 960, 3),
            class_name="alien_uav",
            allowed_classes=self.generator.classes,
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("class" in issue.lower() for issue in issues))

    def test_03_default_image_dimensions(self):
        """Verify UA-DETRAC default resolution is 960x540 (width 960, height 540)."""
        xml_dummy = """<sequence name="MVI_TEST">
            <frame num="1">
                <target id="1">
                    <box left="10" top="20" width="30" height="40"/>
                    <attribute vehicle_type="car"/>
                </target>
            </frame>
        </sequence>"""
        cands = self.importer.parse_ua_detrac(xml_dummy)
        self.assertEqual(len(cands), 1)
        # Shape is (height, width, channels)
        h, w, c = cands[0]["image_shape"]
        self.assertEqual(h, 540)
        self.assertEqual(w, 960)
        self.assertEqual(c, 3)

    def test_04_boundary_bbox_clipping(self):
        """Verify bounding boxes exceeding bounds (e.g. y2=541 on 540h) are clipped and kept valid."""
        raw_box = [800.0, 450.0, 961.0, 541.0]
        shape = (540, 960, 3)

        is_valid, issues, cleaned_box = DataQualityGate.audit_and_clean_annotation(
            bbox=raw_box,
            image_shape=shape,
            class_name="car",
            allowed_classes=self.generator.classes,
            clip_to_bounds=True,
        )
        self.assertTrue(is_valid)
        self.assertEqual(cleaned_box[2], 960.0)
        self.assertEqual(cleaned_box[3], 540.0)

        # Convert to YOLO format and verify normalization strictly <= 1.0
        yolo_str = self.generator.bbox_to_yolo_format(cleaned_box, 960, 540, "car")
        parts = yolo_str.split()
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        self.assertEqual(cls_id, 1)  # car
        self.assertLessEqual(xc + w / 2.0, 1.000001)
        self.assertLessEqual(yc + h / 2.0, 1.000001)
        self.assertGreaterEqual(xc - w / 2.0, -0.000001)
        self.assertGreaterEqual(yc - h / 2.0, -0.000001)

    def test_05_video_aware_frame_identity(self):
        """Verify frames with same frame_idx across different videos do not collide."""
        cands = [
            {
                "video_id": "MVI_20011",
                "frame_idx": 1,
                "bbox": [100, 100, 200, 200],
                "class_name": "car",
                "image_shape": (540, 960, 3),
            },
            {
                "video_id": "MVI_20012",
                "frame_idx": 1,
                "bbox": [300, 300, 400, 400],
                "class_name": "bus",
                "image_shape": (540, 960, 3),
            },
            {
                "video_id": "MVI_20013",
                "frame_idx": 1,
                "bbox": [50, 50, 150, 150],
                "class_name": "truck",
                "image_shape": (540, 960, 3),
            },
        ]
        ds_name = "test_ds_identity"
        manifest = self.generator.create_dataset_version(
            dataset_version=ds_name,
            candidates=cands,
            allow_synthetic_fallback=True,
        )
        self.assertEqual(manifest["total_frames"], 3)
        self.assertEqual(manifest["total_annotations"], 3)

        ds_path = self.test_dir / ds_name
        all_imgs = list(ds_path.rglob("*.jpg"))
        self.assertEqual(len(all_imgs), 3)

        img_names = [p.name for p in all_imgs]
        self.assertTrue(any("MVI_20011" in n for n in img_names))
        self.assertTrue(any("MVI_20012" in n for n in img_names))
        self.assertTrue(any("MVI_20013" in n for n in img_names))

    def test_06_importer_to_generator_integration(self):
        """Verify DatasetImporter.import_and_generate works seamlessly."""
        cands = [
            {
                "scenario_id": "SCN_DETRAC",
                "video_id": "MVI_20011",
                "camera_id": "CAM_DETRAC",
                "frame_idx": 1,
                "bbox": [50, 50, 120, 120],
                "class_name": "car",
            }
        ]
        manifest = self.importer.import_and_generate("test_ds_import_gen", cands)
        self.assertIn("dataset_version", manifest)
        self.assertIn("imported_at", manifest)
        self.assertEqual(manifest["total_annotations"], 1)

    def test_07_real_image_provider(self):
        """Verify UADetracFrameProvider loads real UA-DETRAC images from disk if present."""
        if not UA_DETRAC_IMAGES.is_dir():
            self.skipTest("UA-DETRAC images directory not found on host.")

        provider = UADetracFrameProvider(UA_DETRAC_IMAGES)
        img = provider.get_frame("MVI_20011", 1)
        self.assertIsNotNone(img, "Failed to load MVI_20011 img00001.jpg")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (540, 960, 3))

    def test_08_multiple_objects_per_frame_grouping(self):
        """Verify multiple bounding boxes in the same frame write 1 image file with multi-line label."""
        cands = [
            {
                "video_id": "MVI_MULTI",
                "frame_idx": 1,
                "bbox": [50, 50, 100, 100],
                "class_name": "car",
            },
            {
                "video_id": "MVI_MULTI",
                "frame_idx": 1,
                "bbox": [120, 120, 180, 180],
                "class_name": "bus",
            },
            {
                "video_id": "MVI_MULTI",
                "frame_idx": 1,
                "bbox": [200, 200, 300, 300],
                "class_name": "car",
            },
        ]
        ds_name = "test_ds_multi_obj"
        manifest = self.generator.create_dataset_version(ds_name, cands)
        self.assertEqual(manifest["total_frames"], 1)
        self.assertEqual(manifest["total_annotations"], 3)

        ds_path = self.test_dir / ds_name
        label_files = list(ds_path.rglob("*.txt"))
        yolo_labels = [p for p in label_files if "labels" in str(p)]
        self.assertEqual(len(yolo_labels), 1)

        lines = [l.strip() for l in yolo_labels[0].read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 3)

    def test_09_leak_free_split_partitioning(self):
        """Verify video/sequence-level splitting guarantees zero video overlap across train/val/test."""
        videos = [f"VID_{i:02d}" for i in range(10)]
        cands = []
        for v in videos:
            for f in range(1, 5):
                cands.append({
                    "video_id": v,
                    "frame_idx": f,
                    "bbox": [50, 50, 150, 150],
                    "class_name": "car",
                })

        ds_name = "test_ds_leak_free"
        manifest = self.generator.create_dataset_version(ds_name, cands)
        self.assertEqual(manifest["split_strategy"], "VIDEO_SCENARIO_LEVEL")

        train_vids = set(manifest["videos_per_split"]["train"])
        val_vids = set(manifest["videos_per_split"]["val"])
        test_vids = set(manifest["videos_per_split"]["test"])

        self.assertEqual(train_vids.intersection(val_vids), set())
        self.assertEqual(train_vids.intersection(test_vids), set())
        self.assertEqual(val_vids.intersection(test_vids), set())
        self.assertEqual(train_vids.union(val_vids).union(test_vids), set(videos))

    def test_10_synthetic_fallback_control(self):
        """Verify that allow_synthetic_fallback=False raises FileNotFoundError when real frame is missing."""
        cands = [
            {
                "video_id": "NON_EXISTENT_VIDEO_999",
                "frame_idx": 999,
                "bbox": [50, 50, 150, 150],
                "class_name": "car",
            }
        ]
        with self.assertRaises(FileNotFoundError):
            self.generator.create_dataset_version(
                dataset_version="test_ds_fail_synth",
                candidates=cands,
                allow_synthetic_fallback=False,
            )

    def test_11_real_data_smoke_test(self):
        """Parse first 5 frames of real MVI_20011 with real images and verify YOLO dataset creation."""
        if not UA_DETRAC_XML.is_file() or not UA_DETRAC_IMAGES.is_dir():
            self.skipTest("UA-DETRAC real data not found on host.")

        # 1. Parse XML with max_frames=5
        cands = self.importer.parse_ua_detrac(
            xml_content=UA_DETRAC_XML,
            images_dir=UA_DETRAC_IMAGES,
            max_frames=5,
        )
        self.assertGreaterEqual(len(cands), 5)
        self.assertEqual(cands[0]["image_shape"][:2], (540, 960))

        # 2. Ingest with real frame provider and allow_synthetic_fallback=False
        provider = UADetracFrameProvider(UA_DETRAC_IMAGES)
        ds_name = "test_ua_detrac_smoke"
        manifest = self.importer.import_and_generate(
            dataset_version=ds_name,
            candidates=cands,
            frame_provider_func=provider,
            allow_synthetic_fallback=False,
        )

        self.assertEqual(manifest["total_frames"], 5)
        self.assertGreater(manifest["total_annotations"], 0)

        ds_path = self.test_dir / ds_name
        yaml_file = ds_path / "dataset.yaml"
        self.assertTrue(yaml_file.is_file())

        # Verify real written image properties and byte-identity
        img_files = list(ds_path.rglob("*.jpg"))
        self.assertEqual(len(img_files), 5)

        src_img1 = UA_DETRAC_IMAGES / "MVI_20011" / "img00001.jpg"
        f1_matches = [p for p in img_files if "f000001.jpg" in p.name]
        self.assertEqual(len(f1_matches), 1)
        self.assertEqual(src_img1.read_bytes(), f1_matches[0].read_bytes())

        for img_p in img_files:
            img = cv2.imread(str(img_p))
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (540, 960, 3))

        # Verify labels exist and coordinates are normalized [0.0, 1.0]
        lbl_files = list(ds_path.rglob("*.txt"))
        yolo_lbls = [p for p in lbl_files if "labels" in str(p)]
        self.assertGreaterEqual(len(yolo_lbls), 1)
        for lbl_p in yolo_lbls:
            for line in lbl_p.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                self.assertIn(cls_id, [0, 1, 2, 3, 4, 5, 6, 7, 8])
                for c in coords:
                    self.assertGreaterEqual(c, 0.0)
                    self.assertLessEqual(c, 1.0)

    def test_12_exact_explicit_split_respected(self):
        """Verify that explicit video_splits mapping is strictly respected."""
        cands = [
            {"video_id": "VID_A", "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"},
            {"video_id": "VID_B", "frame_idx": 1, "bbox": [20, 20, 60, 60], "class_name": "car"},
            {"video_id": "VID_C", "frame_idx": 1, "bbox": [30, 30, 70, 70], "class_name": "bus"},
        ]
        # Test Dict[str, str] mapping
        explicit_map = {"VID_A": "test", "VID_B": "train", "VID_C": "val"}
        manifest = self.generator.create_dataset_version(
            dataset_version="test_ds_explicit_map",
            candidates=cands,
            video_splits=explicit_map,
        )
        self.assertEqual(manifest["split_strategy"], "EXPLICIT_SEQUENCE_LEVEL")
        self.assertEqual(manifest["videos_per_split"]["train"], ["VID_B"])
        self.assertEqual(manifest["videos_per_split"]["val"], ["VID_C"])
        self.assertEqual(manifest["videos_per_split"]["test"], ["VID_A"])

        # Also test Dict[str, List[str]] grouped mapping
        grouped_map = {"train": ["VID_A", "VID_C"], "val": [], "test": ["VID_B"]}
        manifest2 = self.generator.create_dataset_version(
            dataset_version="test_ds_explicit_grouped",
            candidates=cands,
            video_splits=grouped_map,
        )
        self.assertEqual(manifest2["videos_per_split"]["train"], ["VID_A", "VID_C"])
        self.assertEqual(manifest2["videos_per_split"]["val"], [])
        self.assertEqual(manifest2["videos_per_split"]["test"], ["VID_B"])

    def test_13_canonical_splits_definition(self):
        """Verify canonical 32/7/7 split configuration covers all 46 usable sequences without overlap."""
        splits_map = load_ua_detrac_split_config()
        self.assertEqual(len(splits_map), 46)

        train_seqs = [k for k, v in splits_map.items() if v == "train"]
        val_seqs = [k for k, v in splits_map.items() if v == "val"]
        test_seqs = [k for k, v in splits_map.items() if v == "test"]

        self.assertEqual(len(train_seqs), 32)
        self.assertEqual(len(val_seqs), 7)
        self.assertEqual(len(test_seqs), 7)

        # Mutually disjoint
        s_train, s_val, s_test = set(train_seqs), set(val_seqs), set(test_seqs)
        self.assertEqual(s_train & s_val, set())
        self.assertEqual(s_train & s_test, set())
        self.assertEqual(s_val & s_test, set())

        # Exact sequence inventory check
        self.assertIn("MVI_20011", s_train)
        self.assertIn("MVI_40201", s_val)
        self.assertIn("MVI_40244", s_test)

    def test_14_missing_sequence_rejected(self):
        """Verify ValueError is raised if any sequence in candidates is omitted from video_splits."""
        cands = [
            {"video_id": "VID_A", "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"},
            {"video_id": "VID_B", "frame_idx": 1, "bbox": [20, 20, 60, 60], "class_name": "car"},
            {"video_id": "VID_C", "frame_idx": 1, "bbox": [30, 30, 70, 70], "class_name": "bus"},
        ]
        incomplete_splits = {"VID_A": "train", "VID_B": "val"}  # VID_C missing
        with self.assertRaises(ValueError) as ctx:
            self.generator.create_dataset_version(
                dataset_version="test_ds_missing_seq",
                candidates=cands,
                video_splits=incomplete_splits,
            )
        self.assertIn("Missing sequence(s)", str(ctx.exception))
        self.assertIn("VID_C", str(ctx.exception))

    def test_15_unknown_sequence_rejected(self):
        """Verify ValueError is raised if video_splits references a sequence not present in candidates."""
        cands = [
            {"video_id": "VID_A", "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"},
            {"video_id": "VID_B", "frame_idx": 1, "bbox": [20, 20, 60, 60], "class_name": "car"},
        ]
        spurious_splits = {"VID_A": "train", "VID_B": "val", "VID_UNKNOWN": "test"}
        with self.assertRaises(ValueError) as ctx:
            self.generator.create_dataset_version(
                dataset_version="test_ds_unknown_seq",
                candidates=cands,
                video_splits=spurious_splits,
            )
        self.assertIn("Unknown sequence(s)", str(ctx.exception))
        self.assertIn("VID_UNKNOWN", str(ctx.exception))

    def test_16_invalid_split_name_rejected(self):
        """Verify ValueError is raised if a split name other than train/val/test is passed."""
        cands = [
            {"video_id": "VID_A", "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"},
        ]
        invalid_name_splits = {"VID_A": "holdout_custom"}
        with self.assertRaises(ValueError) as ctx:
            self.generator.create_dataset_version(
                dataset_version="test_ds_invalid_name",
                candidates=cands,
                video_splits=invalid_name_splits,
            )
        self.assertIn("Invalid split name", str(ctx.exception))

    def test_17_duplicate_conflicting_assignment_rejected(self):
        """Verify ValueError is raised if a sequence is assigned to multiple splits."""
        cands = [
            {"video_id": "VID_A", "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"},
            {"video_id": "VID_B", "frame_idx": 1, "bbox": [20, 20, 60, 60], "class_name": "car"},
        ]
        conflicting_splits = {"train": ["VID_A", "VID_B"], "val": ["VID_A"]}
        with self.assertRaises(ValueError) as ctx:
            self.generator.create_dataset_version(
                dataset_version="test_ds_conflict",
                candidates=cands,
                video_splits=conflicting_splits,
            )
        self.assertIn("Duplicate/conflicting split assignment", str(ctx.exception))

    def test_18_automatic_split_preservation(self):
        """Verify automatic sequence-level split works when video_splits=None."""
        videos = [f"AUTO_VID_{i}" for i in range(5)]
        cands = [{"video_id": v, "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"} for v in videos]
        manifest = self.generator.create_dataset_version(
            dataset_version="test_ds_auto_split",
            candidates=cands,
            video_splits=None,
        )
        self.assertEqual(manifest["split_strategy"], "VIDEO_SCENARIO_LEVEL")
        self.assertGreater(len(manifest["videos_per_split"]["train"]), 0)
        self.assertGreater(len(manifest["videos_per_split"]["val"]), 0)
        self.assertGreater(len(manifest["videos_per_split"]["test"]), 0)

    def test_19_leakage_protection_with_explicit_split(self):
        """Verify DataLeakageDetector confirms zero leakage on explicitly partitioned dataset."""
        cands = [
            {"video_id": "LEAK_A", "frame_idx": 1, "bbox": [10, 10, 50, 50], "class_name": "car"},
            {"video_id": "LEAK_B", "frame_idx": 1, "bbox": [20, 20, 60, 60], "class_name": "car"},
            {"video_id": "LEAK_C", "frame_idx": 1, "bbox": [30, 30, 70, 70], "class_name": "bus"},
        ]
        explicit_splits = {"LEAK_A": "train", "LEAK_B": "val", "LEAK_C": "test"}
        ds_name = "test_ds_leak_audit_explicit"
        manifest = self.generator.create_dataset_version(
            dataset_version=ds_name,
            candidates=cands,
            video_splits=explicit_splits,
        )
        self.assertEqual(manifest["split_strategy"], "EXPLICIT_SEQUENCE_LEVEL")

        report = self.leakage_detector.audit_dataset(ds_name)
        self.assertFalse(report["leakage_detected"])
        self.assertEqual(report["status"], "LEAKAGE_FREE")
        self.assertEqual(len(report["violations"]), 0)


if __name__ == "__main__":
    unittest.main()


