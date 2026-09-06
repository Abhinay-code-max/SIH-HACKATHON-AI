"""
Unit tests for Border Sentinel Dataset Merge and Class Remapping Tool.
Tests verify class ID remapping, unmapped/DROP class filtering,
background negative sample handling, split preservation, deterministic splitting,
safety guards, and report generation.
"""

from pathlib import Path
import shutil
import tempfile
import pytest
import yaml

from dataset.tools.merge_datasets import DatasetMerger, ROOT_DIR


@pytest.fixture(scope="module")
def fixture_dir():
    f_dir = ROOT_DIR / "tests" / "fixtures" / "synthetic_datasets"
    if not (f_dir / "dataset_alpha").is_dir() or not (f_dir / "dataset_beta").is_dir():
        # Generate fixtures if missing
        from tests.fixtures.generate_fixtures import generate_all_fixtures
        generate_all_fixtures()
    return f_dir


@pytest.fixture
def temp_output_dir():
    tmp = tempfile.mkdtemp(prefix="sih_merge_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def base_config_dict(fixture_dir):
    return {
        "version": "3.0.0-test",
        "target_classes": {
            0: "person",
            1: "car",
            2: "truck",
            3: "bus",
            4: "motorcycle",
            5: "bicycle",
            6: "animal",
            7: "backpack",
            8: "bag",
            9: "weapon",
            10: "drone",
            11: "fire",
            12: "smoke",
        },
        "settings": {
            "keep_empty_as_background": True,
            "copy_mode": "copy",
            "prefix_filenames": True,
            "random_split": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
                "seed": 42,
            },
        },
        "datasets": [
            {
                "name": "fixture_alpha",
                "enabled": True,
                "path": str(fixture_dir / "dataset_alpha"),
                "mapping": {
                    "pedestrian": "person",
                    "car": "car",
                    "tricycle": "DROP",
                },
            },
            {
                "name": "fixture_beta",
                "enabled": True,
                "path": str(fixture_dir / "dataset_beta"),
                "mapping": {
                    "drone": "drone",
                    "unknown_noise": "DROP",
                },
            },
        ],
    }


def write_test_config(config_dict: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def test_safety_check_v2_protection(temp_output_dir, base_config_dict):
    """Verifies that the merger strictly prevents writing into dataset/releases/v2."""
    cfg_file = temp_output_dir / "config.yaml"
    write_test_config(base_config_dict, cfg_file)

    v2_path = ROOT_DIR / "dataset" / "releases" / "v2"
    with pytest.raises(PermissionError) as excinfo:
        DatasetMerger(config_path=cfg_file, output_dir=v2_path)
    assert "SAFETY VIOLATION" in str(excinfo.value)

    v2_subpath = ROOT_DIR / "dataset" / "releases" / "v2" / "nested_dir"
    with pytest.raises(PermissionError) as excinfo_sub:
        DatasetMerger(config_path=cfg_file, output_dir=v2_subpath)
    assert "SAFETY VIOLATION" in str(excinfo_sub.value)


def test_class_id_remapping(temp_output_dir, base_config_dict):
    """Verifies source class IDs are remapped correctly to target class IDs."""
    cfg_file = temp_output_dir / "config.yaml"
    out_dir = temp_output_dir / "merged"
    write_test_config(base_config_dict, cfg_file)

    merger = DatasetMerger(config_path=cfg_file, output_dir=out_dir)
    summary = merger.run()

    # Alpha: pedestrian -> person (0), car -> car (1)
    # Beta: drone -> drone (10)
    assert summary["final_target_class_counts"]["person"] == 2
    assert summary["final_target_class_counts"]["car"] == 2
    assert summary["final_target_class_counts"]["drone"] == 8

    # Inspect a remapped train label file
    train_labels = list((out_dir / "labels" / "train").glob("*.txt"))
    assert len(train_labels) > 0

    alpha_001_lbl = out_dir / "labels" / "train" / "fixture_alpha_alpha_001.txt"
    assert alpha_001_lbl.is_file()
    lines = alpha_001_lbl.read_text().splitlines()
    assert len(lines) == 2  # pedestrian and car kept, tricycle dropped
    assert lines[0].startswith("0 ")  # person target ID is 0
    assert lines[1].startswith("1 ")  # car target ID is 1


def test_drop_unmapped_classes_with_logging(temp_output_dir, base_config_dict):
    """Verifies unmapped and DROP classes are excluded from output labels and logged in audit records."""
    # Modify config to leave 'unknown_noise' unmapped instead of explicit DROP
    cfg_dict = base_config_dict.copy()
    cfg_dict["datasets"][1]["mapping"] = {"drone": "drone"}  # 'unknown_noise' is unmapped

    cfg_file = temp_output_dir / "config.yaml"
    out_dir = temp_output_dir / "merged"
    write_test_config(cfg_dict, cfg_file)

    merger = DatasetMerger(config_path=cfg_file, output_dir=out_dir)
    summary = merger.run()

    # tricycle explicitly dropped (2) + unknown_noise unmapped dropped (2) = 4 dropped boxes
    assert summary["total_boxes_dropped"] == 4
    reasons = [r["reason"] for r in merger.audit_records]
    assert "explicitly_dropped" in reasons
    assert "unmapped_class_dropped" in reasons


def test_negative_image_handling_keep_as_background(temp_output_dir, base_config_dict):
    """Verifies that keep_empty_as_background=True preserves empty images as background samples."""
    cfg_file = temp_output_dir / "config.yaml"
    out_dir = temp_output_dir / "merged"
    write_test_config(base_config_dict, cfg_file)

    merger = DatasetMerger(
        config_path=cfg_file,
        output_dir=out_dir,
        keep_empty_as_background=True,
    )
    summary = merger.run()

    # 4 images with 0 annotations: alpha_003_neg, alpha_val_002_drop, beta_004, beta_003
    assert summary["total_negative_images"] == 4
    assert summary["total_dropped_empty_images"] == 0
    assert summary["total_images"] == 15

    # Check that empty label file was created for negative image
    neg_lbl = out_dir / "labels" / "train" / "fixture_alpha_alpha_003_neg.txt"
    assert neg_lbl.is_file()
    assert neg_lbl.read_text().strip() == ""


def test_negative_image_handling_drop_empty(temp_output_dir, base_config_dict):
    """Verifies that keep_empty_as_background=False discards images with 0 annotations."""
    cfg_file = temp_output_dir / "config.yaml"
    out_dir = temp_output_dir / "merged"
    write_test_config(base_config_dict, cfg_file)

    merger = DatasetMerger(
        config_path=cfg_file,
        output_dir=out_dir,
        keep_empty_as_background=False,
    )
    summary = merger.run()

    # 4 empty images dropped
    assert summary["total_negative_images"] == 0
    assert summary["total_dropped_empty_images"] == 4
    assert summary["total_images"] == 11

    # Verify dropped negative image is absent from output images
    all_images = [p.name for p in (out_dir / "images").rglob("*.jpg")]
    assert "fixture_alpha_alpha_003_neg.jpg" not in all_images
    assert "fixture_alpha_alpha_val_002_drop.jpg" not in all_images


def test_preservation_of_existing_splits(temp_output_dir, base_config_dict):
    """Verifies that source datasets with existing train/val splits preserve their assignments."""
    # Only test fixture_alpha (which has 3 train and 2 val)
    cfg_dict = base_config_dict.copy()
    cfg_dict["datasets"] = [cfg_dict["datasets"][0]]

    cfg_file = temp_output_dir / "config.yaml"
    out_dir = temp_output_dir / "merged"
    write_test_config(cfg_dict, cfg_file)

    merger = DatasetMerger(config_path=cfg_file, output_dir=out_dir)
    summary = merger.run()

    assert summary["split_distribution"]["train"] == 3
    assert summary["split_distribution"]["val"] == 2
    assert summary["split_distribution"]["test"] == 0


def test_reproducible_random_split(temp_output_dir, base_config_dict):
    """Verifies that un-split datasets are split deterministically across runs with the same seed."""
    cfg_dict = base_config_dict.copy()
    cfg_dict["datasets"] = [cfg_dict["datasets"][1]]  # fixture_beta only (10 images)

    cfg_file = temp_output_dir / "config.yaml"
    write_test_config(cfg_dict, cfg_file)

    out1 = temp_output_dir / "out1"
    out2 = temp_output_dir / "out2"

    merger1 = DatasetMerger(config_path=cfg_file, output_dir=out1, random_seed=123)
    sum1 = merger1.run()

    merger2 = DatasetMerger(config_path=cfg_file, output_dir=out2, random_seed=123)
    sum2 = merger2.run()

    # Both runs should yield identical split counts
    assert sum1["split_distribution"] == sum2["split_distribution"]

    # Both runs should allocate the exact same files to train, val, test
    for s in ("train", "val", "test"):
        files1 = sorted([p.name for p in (out1 / "images" / s).glob("*.jpg")])
        files2 = sorted([p.name for p in (out2 / "images" / s).glob("*.jpg")])
        assert files1 == files2


def test_data_yaml_and_report_generation(temp_output_dir, base_config_dict):
    """Verifies that data.yaml, merge_summary.json, and merge_summary.md are correctly formatted."""
    cfg_file = temp_output_dir / "config.yaml"
    out_dir = temp_output_dir / "merged"
    write_test_config(base_config_dict, cfg_file)

    merger = DatasetMerger(config_path=cfg_file, output_dir=out_dir)
    merger.run()

    # Verify data.yaml
    data_yaml_path = out_dir / "data.yaml"
    assert data_yaml_path.is_file()
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        dy = yaml.safe_load(f)
    assert dy["nc"] == 13
    assert dy["train"] == "images/train"
    assert dy["val"] == "images/val"
    assert dy["test"] == "images/test"
    assert dy["names"][0] == "person"
    assert dy["names"][10] == "drone"

    # Verify JSON summary
    json_path = out_dir / "merge_summary.json"
    assert json_path.is_file()

    # Verify Markdown summary
    md_path = out_dir / "merge_summary.md"
    assert md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "# Dataset Merge & Class-Remap Summary Report" in md_text
    assert "fixture_alpha" in md_text
    assert "fixture_beta" in md_text
