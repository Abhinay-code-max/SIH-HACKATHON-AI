"""
Border Sentinel (SIH-HACKATHON-AI)
Dataset Merge & Class-Remapping Tool

Merges multiple external YOLO-format datasets with different class taxonomies
into a unified dataset matching the target class schema. Rewrites label files,
manages train/val/test splits, handles hard negatives, and generates audit reports.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Protected production paths that must never be overwritten
PROTECTED_RELEASE_PATHS = [
    ROOT_DIR / "dataset" / "releases" / "v2",
    ROOT_DIR / "dataset" / "releases" / "v1",
]

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_datasets")


class DatasetMergeError(Exception):
    """Custom exception for dataset merge errors."""
    pass


class DatasetMerger:
    """
    Handles merging and remapping of multiple YOLO datasets.
    """

    def __init__(
        self,
        config_path: Path | str,
        output_dir: Optional[Path | str] = None,
        keep_empty_as_background: Optional[bool] = None,
        copy_mode: Optional[str] = None,
        random_seed: Optional[int] = None,
        prefix_filenames: Optional[bool] = None,
        dry_run: bool = False,
    ):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.config = self._load_config(self.config_path)

        # Settings resolution
        settings = self.config.get("settings", {})
        
        # Output directory
        raw_output = output_dir or settings.get("output_dir", "dataset/releases/v3-staging")
        self.output_dir = Path(raw_output)
        if not self.output_dir.is_absolute():
            self.output_dir = (ROOT_DIR / self.output_dir).resolve()
        else:
            self.output_dir = self.output_dir.resolve()

        # Enforce safety check
        self._verify_safety(self.output_dir)

        # Transfer mode: 'copy' or 'symlink'
        self.copy_mode = copy_mode or settings.get("copy_mode", "copy")
        if self.copy_mode not in ("copy", "symlink"):
            raise ValueError(f"Invalid copy_mode: {self.copy_mode}. Must be 'copy' or 'symlink'.")

        # Hard negative retention
        if keep_empty_as_background is not None:
            self.keep_empty_as_background = keep_empty_as_background
        else:
            self.keep_empty_as_background = settings.get("keep_empty_as_background", True)

        # Random split settings
        split_cfg = settings.get("random_split", {})
        self.split_ratios = {
            "train": split_cfg.get("train", 0.70),
            "val": split_cfg.get("val", 0.15),
            "test": split_cfg.get("test", 0.15),
        }
        self.random_seed = random_seed if random_seed is not None else split_cfg.get("seed", 42)

        # Filename prefixing
        if prefix_filenames is not None:
            self.prefix_filenames = prefix_filenames
        else:
            self.prefix_filenames = settings.get("prefix_filenames", True)

        self.dry_run = dry_run

        # Parse target taxonomy
        self.target_classes, self.target_name_to_id = self._parse_target_classes()

        # Audit & summary tracking
        self.audit_records: List[Dict[str, Any]] = []
        self.source_class_counts: Dict[str, Dict[str, int]] = {}
        self.target_class_counts: Dict[str, int] = {name: 0 for name in self.target_classes.values()}
        self.split_image_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.dataset_image_counts: Dict[str, int] = {}
        self.negative_images_count = 0
        self.dropped_empty_images_count = 0
        self.total_boxes_dropped = 0

    @staticmethod
    def _verify_safety(target_dir: Path):
        """Ensures target directory is not an existing protected production dataset."""
        resolved = target_dir.resolve()
        for protected in PROTECTED_RELEASE_PATHS:
            prot_res = protected.resolve()
            if resolved == prot_res or prot_res in resolved.parents:
                raise PermissionError(
                    f"SAFETY VIOLATION: Output directory '{resolved}' attempts to overwrite or write inside "
                    f"protected production dataset '{prot_res}'!"
                )

    @staticmethod
    def _load_config(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    def _parse_target_classes(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        raw_classes = self.config.get("target_classes", {})
        if not raw_classes:
            raise ValueError("Configuration missing 'target_classes' definition.")

        id_to_name: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}

        for k, v in raw_classes.items():
            cid = int(k)
            name = str(v).strip()
            id_to_name[cid] = name
            name_to_id[name] = cid

        return id_to_name, name_to_id

    def _discover_source_classes(self, dataset_path: Path, dataset_cfg: dict) -> Dict[int, str]:
        """
        Resolves source class ID -> name from:
        1. Explicit 'classes' dict in dataset config
        2. data.yaml / dataset.yaml in dataset directory
        3. classes.txt in dataset directory
        """
        # 1. Config explicit override
        if "classes" in dataset_cfg and dataset_cfg["classes"]:
            raw = dataset_cfg["classes"]
            if isinstance(raw, dict):
                return {int(k): str(v).strip() for k, v in raw.items()}
            elif isinstance(raw, list):
                return {idx: str(v).strip() for idx, v in enumerate(raw)}

        # 2. Look for data.yaml or dataset.yaml
        for yaml_name in ("data.yaml", "dataset.yaml"):
            y_path = dataset_path / yaml_name
            if y_path.is_file():
                try:
                    with open(y_path, "r", encoding="utf-8") as f:
                        y_data = yaml.safe_load(f) or {}
                    names = y_data.get("names")
                    if isinstance(names, dict):
                        return {int(k): str(v).strip() for k, v in names.items()}
                    elif isinstance(names, list):
                        return {idx: str(v).strip() for idx, v in enumerate(names)}
                except Exception as e:
                    logger.warning(f"Failed to read classes from {y_path}: {e}")

        # 3. Look for classes.txt
        classes_txt = dataset_path / "classes.txt"
        if classes_txt.is_file():
            try:
                lines = [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
                return {idx: name for idx, name in enumerate(lines)}
            except Exception as e:
                logger.warning(f"Failed to read {classes_txt}: {e}")

        logger.warning(
            f"No class definition found for dataset at '{dataset_path}'. "
            "Will fall back to matching mapping keys directly or assuming IDs."
        )
        return {}

    def _discover_images_and_labels(self, dataset_path: Path) -> List[Tuple[Path, Optional[Path], Optional[str]]]:
        """
        Scans a dataset directory and pairs images with label files and pre-existing splits.
        Returns list of (image_path, label_path_or_none, detected_split_or_none).
        """
        items: List[Tuple[Path, Optional[Path], Optional[str]]] = []
        splits = ["train", "val", "test"]

        # Check Layout A: images/{train,val,test} and labels/{train,val,test}
        layout_a = False
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        if images_dir.is_dir():
            for s in splits:
                s_img_dir = images_dir / s
                s_lbl_dir = labels_dir / s
                if s_img_dir.is_dir():
                    layout_a = True
                    for img_file in s_img_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                            lbl_file = s_lbl_dir / f"{img_file.stem}.txt" if s_lbl_dir.is_dir() else None
                            items.append((img_file, lbl_file if (lbl_file and lbl_file.is_file()) else None, s))

        if layout_a and items:
            return items

        # Check Layout B: {train,val,test}/images and {train,val,test}/labels
        layout_b = False
        for s in splits:
            s_img_dir = dataset_path / s / "images"
            s_lbl_dir = dataset_path / s / "labels"
            if s_img_dir.is_dir():
                layout_b = True
                for img_file in s_img_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        lbl_file = s_lbl_dir / f"{img_file.stem}.txt" if s_lbl_dir.is_dir() else None
                        items.append((img_file, lbl_file if (lbl_file and lbl_file.is_file()) else None, s))

        if layout_b and items:
            return items

        # Check Layout C: flat images/ and labels/ (no split)
        if images_dir.is_dir():
            for img_file in images_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    lbl_file = labels_dir / f"{img_file.stem}.txt" if labels_dir.is_dir() else None
                    items.append((img_file, lbl_file if (lbl_file and lbl_file.is_file()) else None, None))
            if items:
                return items

        # Fallback: scan whole directory recursively for images
        for img_file in dataset_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                # Check if path contains train, val, or test
                path_parts = [p.lower() for p in img_file.parts]
                det_split = None
                for s in splits:
                    if s in path_parts:
                        det_split = s
                        break
                
                # Check adjacent or labels directory for matching .txt
                lbl_file = None
                possible_lbls = [
                    img_file.with_suffix(".txt"),
                    img_file.parent.parent / "labels" / f"{img_file.stem}.txt",
                    dataset_path / "labels" / f"{img_file.stem}.txt",
                ]
                for pl in possible_lbls:
                    if pl.is_file():
                        lbl_file = pl
                        break

                items.append((img_file, lbl_file, det_split))

        return items

    def _allocate_splits(
        self,
        items: List[Tuple[Path, Optional[Path], Optional[str]]],
        dataset_name: str,
    ) -> List[Tuple[Path, Optional[Path], str]]:
        """
        Allocates items to train/val/test splits:
        - If split was detected from directory structure, preserves it.
        - If item has no split, assigns via seeded deterministic random split.
        """
        result: List[Tuple[Path, Optional[Path], str]] = []
        unassigned: List[Tuple[Path, Optional[Path], Optional[str]]] = []

        for img_p, lbl_p, det_split in items:
            if det_split in ("train", "val", "test"):
                result.append((img_p, lbl_p, det_split))
            else:
                unassigned.append((img_p, lbl_p, det_split))

        if unassigned:
            # Sort unassigned by file path for cross-platform determinism before shuffling
            unassigned.sort(key=lambda x: str(x[0]))
            rng = random.Random(self.random_seed)
            rng.shuffle(unassigned)

            total = len(unassigned)
            train_count = int(total * self.split_ratios["train"])
            val_count = int(total * self.split_ratios["val"])

            for idx, (img_p, lbl_p, _) in enumerate(unassigned):
                if idx < train_count:
                    split = "train"
                elif idx < train_count + val_count:
                    split = "val"
                else:
                    split = "test"
                result.append((img_p, lbl_p, split))

            logger.info(
                f"Dataset '{dataset_name}' allocated un-split items ({total}) using seed {self.random_seed}: "
                f"train={train_count}, val={val_count}, test={total - train_count - val_count}"
            )

        return result

    def _remap_label_file(
        self,
        label_path: Optional[Path],
        source_classes: Dict[int, str],
        dataset_mapping: Dict[str, str],
        dataset_name: str,
        image_name: str,
    ) -> Tuple[List[str], int, int]:
        """
        Remaps label lines according to explicit mapping.
        Returns:
            (remapped_lines, valid_boxes_count, dropped_boxes_count)
        """
        if label_path is None or not label_path.is_file():
            return [], 0, 0

        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            return [], 0, 0

        remapped_lines: List[str] = []
        dropped_count = 0

        # Create lookup mappings for fast matching
        # Mapping can be keyed by source class name or source class id (as str or int)
        for line_num, line in enumerate(content.splitlines(), start=1):
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            if not parts:
                continue

            # Class ID is first token
            try:
                src_id = int(parts[0])
            except ValueError:
                self.audit_records.append({
                    "dataset": dataset_name,
                    "image": image_name,
                    "line": line_num,
                    "raw": line_str,
                    "reason": "invalid_class_id_format",
                })
                dropped_count += 1
                continue

            # Resolve source class name
            src_name = source_classes.get(src_id, str(src_id))

            # Record source class instance count
            if dataset_name not in self.source_class_counts:
                self.source_class_counts[dataset_name] = {}
            self.source_class_counts[dataset_name][src_name] = (
                self.source_class_counts[dataset_name].get(src_name, 0) + 1
            )

            # Determine target class from mapping
            # Try name match first, then str(id) match, then int match
            target_class_spec = None
            if src_name in dataset_mapping:
                target_class_spec = dataset_mapping[src_name]
            elif str(src_id) in dataset_mapping:
                target_class_spec = dataset_mapping[str(src_id)]
            elif src_id in dataset_mapping:
                target_class_spec = dataset_mapping[src_id]

            # Case: Not found in mapping
            if target_class_spec is None:
                self.audit_records.append({
                    "dataset": dataset_name,
                    "image": image_name,
                    "line": line_num,
                    "source_class_id": src_id,
                    "source_class_name": src_name,
                    "reason": "unmapped_class_dropped",
                })
                dropped_count += 1
                continue

            target_class_str = str(target_class_spec).strip()

            # Case: Explicit DROP
            if target_class_str.upper() == "DROP":
                self.audit_records.append({
                    "dataset": dataset_name,
                    "image": image_name,
                    "line": line_num,
                    "source_class_id": src_id,
                    "source_class_name": src_name,
                    "reason": "explicitly_dropped",
                })
                dropped_count += 1
                continue

            # Case: Map to target class
            if target_class_str not in self.target_name_to_id:
                # Specified target class does not exist in target taxonomy!
                self.audit_records.append({
                    "dataset": dataset_name,
                    "image": image_name,
                    "line": line_num,
                    "source_class_id": src_id,
                    "source_class_name": src_name,
                    "target_class_requested": target_class_str,
                    "reason": "target_class_not_in_taxonomy_dropped",
                })
                dropped_count += 1
                logger.warning(
                    f"[{dataset_name}] Target class '{target_class_str}' is not in target taxonomy! Dropping line {line_num} in {image_name}."
                )
                continue

            target_id = self.target_name_to_id[target_class_str]
            parts[0] = str(target_id)
            remapped_line = " ".join(parts)
            remapped_lines.append(remapped_line)

            # Update count
            self.target_class_counts[target_class_str] += 1

        return remapped_lines, len(remapped_lines), dropped_count

    def _transfer_file(self, src: Path, dst: Path):
        """Copies or symlinks file based on transfer mode."""
        if self.dry_run:
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()

        if self.copy_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
                return
            except OSError as e:
                logger.warning(f"Symlink failed for '{src}' -> '{dst}': {e}. Falling back to copy.")

        # Default copy
        shutil.copy2(src, dst)

    def process_dataset(self, dataset_cfg: dict) -> Dict[str, Any]:
        """Processes a single source dataset."""
        dataset_name = dataset_cfg.get("name", "unnamed_dataset")
        raw_path = dataset_cfg.get("path")
        if not raw_path:
            raise ValueError(f"Dataset '{dataset_name}' missing 'path'.")

        ds_path = Path(raw_path)
        if not ds_path.is_absolute():
            ds_path = (ROOT_DIR / ds_path).resolve()
        else:
            ds_path = ds_path.resolve()

        if not ds_path.is_dir():
            logger.warning(f"Dataset path '{ds_path}' for '{dataset_name}' does not exist or is not a directory. Skipping.")
            return {"dataset": dataset_name, "status": "skipped", "reason": "directory_not_found"}

        mapping = dataset_cfg.get("mapping", {})
        if not mapping:
            logger.warning(f"No mapping defined for dataset '{dataset_name}'. All classes will be dropped!")

        source_classes = self._discover_source_classes(ds_path, dataset_cfg)
        logger.info(f"Processing dataset '{dataset_name}' ({ds_path}) with source classes: {source_classes}")

        raw_items = self._discover_images_and_labels(ds_path)
        if not raw_items:
            logger.warning(f"No images found in dataset '{dataset_name}' at '{ds_path}'.")
            return {"dataset": dataset_name, "status": "empty", "images_found": 0}

        allocated_items = self._allocate_splits(raw_items, dataset_name)

        ds_stats = {
            "total_images": len(allocated_items),
            "kept_images": 0,
            "dropped_images": 0,
            "negative_images": 0,
            "valid_boxes": 0,
            "dropped_boxes": 0,
            "splits": {"train": 0, "val": 0, "test": 0},
        }

        for img_path, lbl_path, split in allocated_items:
            orig_filename = img_path.name
            if self.prefix_filenames:
                out_img_name = f"{dataset_name}_{orig_filename}"
                out_lbl_name = f"{dataset_name}_{img_path.stem}.txt"
            else:
                out_img_name = orig_filename
                out_lbl_name = f"{img_path.stem}.txt"

            remapped_lines, valid_cnt, drop_cnt = self._remap_label_file(
                lbl_path,
                source_classes,
                mapping,
                dataset_name,
                orig_filename,
            )

            ds_stats["valid_boxes"] += valid_cnt
            ds_stats["dropped_boxes"] += drop_cnt
            self.total_boxes_dropped += drop_cnt

            is_negative = (valid_cnt == 0)

            if is_negative and not self.keep_empty_as_background:
                # Drop negative sample
                ds_stats["dropped_images"] += 1
                self.dropped_empty_images_count += 1
                self.audit_records.append({
                    "dataset": dataset_name,
                    "image": orig_filename,
                    "reason": "empty_negative_dropped_by_policy",
                })
                continue

            if is_negative:
                ds_stats["negative_images"] += 1
                self.negative_images_count += 1

            # Prepare destinations
            dst_img = self.output_dir / "images" / split / out_img_name
            dst_lbl = self.output_dir / "labels" / split / out_lbl_name

            # Transfer image
            self._transfer_file(img_path, dst_img)

            # Write label file (empty if negative)
            if not self.dry_run:
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    if remapped_lines:
                        f.write("\n".join(remapped_lines) + "\n")

            ds_stats["kept_images"] += 1
            ds_stats["splits"][split] += 1
            self.split_image_counts[split] += 1

        self.dataset_image_counts[dataset_name] = ds_stats["kept_images"]
        logger.info(
            f"Dataset '{dataset_name}' merged: kept={ds_stats['kept_images']}, "
            f"dropped_images={ds_stats['dropped_images']}, negative_samples={ds_stats['negative_images']}, "
            f"valid_boxes={ds_stats['valid_boxes']}, dropped_boxes={ds_stats['dropped_boxes']}"
        )
        return ds_stats

    def _write_data_yaml(self):
        """Generates Ultralytics-compatible data.yaml."""
        if self.dry_run:
            return

        yaml_content = {
            "path": str(self.output_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.target_classes),
            "names": {int(k): v for k, v in self.target_classes.items()},
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Generated data.yaml at {yaml_path}")

    def _write_summary_reports(self) -> Dict[str, Any]:
        """Generates machine-readable JSON and human-readable Markdown summary reports."""
        total_kept_images = sum(self.split_image_counts.values())

        summary = {
            "version": self.config.get("version", "3.0.0"),
            "output_directory": str(self.output_dir).replace("\\", "/"),
            "target_classes": {str(k): v for k, v in self.target_classes.items()},
            "keep_empty_as_background": self.keep_empty_as_background,
            "copy_mode": self.copy_mode,
            "total_images": total_kept_images,
            "total_negative_images": self.negative_images_count,
            "total_dropped_empty_images": self.dropped_empty_images_count,
            "total_boxes_dropped": self.total_boxes_dropped,
            "split_distribution": self.split_image_counts,
            "dataset_image_counts": self.dataset_image_counts,
            "source_class_counts": self.source_class_counts,
            "final_target_class_counts": self.target_class_counts,
            "audit_records_count": len(self.audit_records),
            "sample_audit_records": self.audit_records[:50],  # Include first 50 dropped records
        }

        if not self.dry_run:
            # 1. JSON Report
            json_path = self.output_dir / "merge_summary.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            # 2. Markdown Report
            md_path = self.output_dir / "merge_summary.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self._build_markdown_report(summary))

            logger.info(f"Generated summary reports at {json_path} and {md_path}")

        return summary

    def _build_markdown_report(self, summary: dict) -> str:
        """Constructs a clean Markdown summary table."""
        lines = [
            "# Dataset Merge & Class-Remap Summary Report",
            "",
            f"- **Output Directory:** `{summary['output_directory']}`",
            f"- **Keep Hard Negatives (Background):** `{summary['keep_empty_as_background']}`",
            f"- **File Transfer Mode:** `{summary['copy_mode']}`",
            f"- **Total Merged Images:** **{summary['total_images']}** (Train: {summary['split_distribution']['train']}, Val: {summary['split_distribution']['val']}, Test: {summary['split_distribution']['test']})",
            f"- **Negative Samples Retained:** {summary['total_negative_images']}",
            f"- **Empty Images Dropped:** {summary['total_dropped_empty_images']}",
            f"- **Total Bounding Boxes Dropped:** {summary['total_boxes_dropped']}",
            "",
            "## 1. Images per Source Dataset",
            "| Source Dataset | Final Images Merged |",
            "| :--- | :--- |",
        ]
        for ds, count in summary["dataset_image_counts"].items():
            lines.append(f"| `{ds}` | {count} |")

        lines.extend([
            "",
            "## 2. Final Target Class Distribution",
            "| Class ID | Target Class Name | Final Instance Count |",
            "| :--- | :--- | :--- |",
        ])
        for cid, name in self.target_classes.items():
            cnt = self.target_class_counts.get(name, 0)
            lines.append(f"| {cid} | `{name}` | {cnt} |")

        lines.extend([
            "",
            "## 3. Source Class Instances Discovered",
            "| Dataset | Source Class | Count |",
            "| :--- | :--- | :--- |",
        ])
        for ds, cls_dict in summary["source_class_counts"].items():
            for s_cls, cnt in cls_dict.items():
                lines.append(f"| `{ds}` | `{s_cls}` | {cnt} |")

        lines.extend([
            "",
            "## 4. Dropped / Excluded Annotations Audit (Sample)",
            "| Dataset | Image | Source Class | Reason |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for r in summary["sample_audit_records"][:20]:
            s_cls = r.get("source_class_name", "N/A")
            img = r.get("image", "N/A")
            ds = r.get("dataset", "N/A")
            reason = r.get("reason", "N/A")
            lines.append(f"| `{ds}` | `{img}` | `{s_cls}` | `{reason}` |")

        if summary["audit_records_count"] > 20:
            lines.append(f"\n*...and {summary['audit_records_count'] - 20} more audit records in merge_summary.json.*")

        lines.append("")
        return "\n".join(lines)

    def run(self) -> Dict[str, Any]:
        """Executes the full dataset merge pipeline."""
        logger.info(f"Starting dataset merge pipeline. Target output: {self.output_dir}")
        datasets = self.config.get("datasets", [])
        if not datasets:
            raise DatasetMergeError("No datasets configured under 'datasets'.")

        # Create split folders if not dry run
        if not self.dry_run:
            for s in ("train", "val", "test"):
                (self.output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
                (self.output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

        processed_count = 0
        for ds_cfg in datasets:
            if not ds_cfg.get("enabled", True):
                logger.info(f"Dataset '{ds_cfg.get('name')}' is disabled. Skipping.")
                continue

            self.process_dataset(ds_cfg)
            processed_count += 1

        if processed_count == 0:
            logger.warning("No enabled datasets were processed!")

        # Write data.yaml and summaries
        self._write_data_yaml()
        summary = self._write_summary_reports()

        logger.info(
            f"Merge complete! Merged {summary['total_images']} images into '{self.output_dir}'. "
            f"Audit log has {summary['audit_records_count']} dropped items."
        )
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Unified YOLO Dataset Merge and Class-Remapping Tool."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "dataset" / "tools" / "merge_config.yaml"),
        help="Path to merge configuration YAML file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output directory for merged dataset (e.g. dataset/releases/v3-staging).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        default=None,
        help="Retain images with 0 annotations as hard negative/background samples.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        default=None,
        help="Discard images with 0 annotations.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying image files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic train/val/test splitting of un-split datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and validate mappings without writing or copying files.",
    )

    args = parser.parse_args()

    keep_empty = None
    if args.drop_empty:
        keep_empty = False
    elif args.keep_empty:
        keep_empty = True

    copy_mode = "symlink" if args.symlink else "copy"

    try:
        merger = DatasetMerger(
            config_path=args.config,
            output_dir=args.output,
            keep_empty_as_background=keep_empty,
            copy_mode=copy_mode,
            random_seed=args.seed,
            dry_run=args.dry_run,
        )
        summary = merger.run()
        print("\n--- YOLO Dataset Merge Completed Successfully ---")
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Total Merged Images: {summary['total_images']}")
        print(f"Split Distribution: {summary['split_distribution']}")
        print(f"Hard Negatives Retained: {summary['total_negative_images']}")
        print(f"Empty Images Dropped: {summary['total_dropped_empty_images']}")
        print(f"Boxes Dropped: {summary['total_boxes_dropped']}")
        print(f"Target Class Counts: {summary['final_target_class_counts']}")
    except PermissionError as pe:
        logger.error(f"SAFETY ERROR: {pe}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"MERGE FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
