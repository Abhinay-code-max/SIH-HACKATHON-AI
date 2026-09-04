"""
Dataset Foundation Manager for Milestone v0.2.
Validates master class schemas, initializes directory trees, and manages dataset versioning.
"""

from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class DatasetManager:
    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            self.config_path = ROOT_DIR / "config" / "classes.yaml"
        else:
            self.config_path = Path(config_path)

        self.dataset_root = ROOT_DIR / "dataset"
        self.dirs = {
            "raw_footage": self.dataset_root / "raw_footage",
            "extracted_frames": self.dataset_root / "extracted_frames",
            "pre_annotations": self.dataset_root / "pre_annotations",
            "verified_annotations": self.dataset_root / "verified_annotations",
            "failure_cases": self.dataset_root / "failure_cases",
            "releases": self.dataset_root / "releases",
        }

        self.config = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Master class config missing at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_classes(self) -> dict:
        """Returns dict of class_id -> class_name."""
        return self.config.get("classes", {})

    def get_class_names_list(self) -> list:
        classes = self.get_classes()
        return [classes[k] for k in sorted(classes.keys())]

    def init_directory_structure(self) -> dict:
        """Creates and validates the dataset directory layout."""
        created = []
        for name, p in self.dirs.items():
            p.mkdir(parents=True, exist_ok=True)
            marker = p / ".keep"
            if not marker.exists():
                marker.touch()
            created.append(str(p.relative_to(ROOT_DIR)))

        # Also initialize standard failure case taxonomy
        failure_categories = [
            "missed_person",
            "false_person",
            "tiny_objects",
            "night",
            "vegetation",
            "occlusion",
            "blur",
            "hard_negatives",
        ]
        for cat in failure_categories:
            cat_dir = self.dirs["failure_cases"] / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            (cat_dir / ".keep").touch()

        return {
            "directories_initialized": created,
            "failure_taxonomy": failure_categories,
            "classes_loaded": self.get_classes(),
        }


if __name__ == "__main__":
    try:
        mgr = DatasetManager()
        res = mgr.init_directory_structure()
        print("\n--- Dataset Foundation Initialization ---")
        print(f"Master Classes ({len(res['classes_loaded'])}): {res['classes_loaded']}")
        print(f"Failure Case Taxonomy: {res['failure_taxonomy']}")
        print(f"Dataset Root: {mgr.dataset_root}")
        print("Status: DATASET ARCHITECTURE INITIALIZED SUCCESSFULLY")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
