"""
Deterministic Derived Dataset View Engine.
Creates isolated, reproducible, experiment-specific derived dataset representations
with on-the-fly or materialized label remapping and zero-copy directory junctions
for image assets. Ensures canonical datasets remain strictly immutable and byte-for-byte untouched.
"""

import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file for immutability auditing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def link_directory_zero_copy(source_dir: Path, target_link: Path) -> None:
    """
    Creates a zero-copy directory junction on Windows (or symlink on POSIX).
    Does NOT require administrator privileges or Developer Mode on Windows.
    Consumes 0 bytes of disk space for images.
    """
    source_dir = source_dir.resolve()
    target_link = target_link.resolve()

    if target_link.exists() or target_link.is_symlink():
        if target_link.is_dir() and not target_link.is_symlink():
            try:
                os.rmdir(str(target_link))
            except OSError:
                shutil.rmtree(str(target_link))
        else:
            try:
                target_link.unlink(missing_ok=True)
            except OSError:
                os.rmdir(str(target_link))

    target_link.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        import _winapi

        try:
            _winapi.CreateJunction(str(source_dir), str(target_link))
            return
        except Exception:
            pass

    # Fallback to standard symlink
    try:
        os.symlink(str(source_dir), str(target_link), target_is_directory=True)
    except Exception as e:
        raise OSError(
            f"Failed to create zero-copy directory link from {source_dir} to {target_link}: {e}"
        )


def remap_label_line(
    line: str,
    class_mapping: Dict[int, int],
    source_file: str = "",
) -> str:
    """
    Transforms a single YOLO annotation row.
    Accepts only classes present in class_mapping.
    Raises ValueError loudly for any unsupported canonical class ID.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""

    parts = stripped.split()
    if len(parts) < 5:
        raise ValueError(
            f"Malformed YOLO annotation row in '{source_file}': '{line}'. Expected at least 5 tokens."
        )

    try:
        canonical_cls = int(parts[0])
    except ValueError:
        raise ValueError(
            f"Non-integer class ID '{parts[0]}' in '{source_file}'."
        )

    if canonical_cls not in class_mapping:
        raise ValueError(
            f"Unsupported canonical class ID {canonical_cls} in '{source_file}'. "
            f"Permitted canonical classes for this mapping are: {sorted(list(class_mapping.keys()))}. "
            f"Row content: '{line}'"
        )

    target_cls = class_mapping[canonical_cls]
    return f"{target_cls} " + " ".join(parts[1:])


def remap_label_file(
    src_file: Path,
    dst_file: Path,
    class_mapping: Dict[int, int],
) -> int:
    """
    Reads a canonical label file, validates and remaps its class IDs,
    and writes the remapped content to dst_file.
    Returns count of written annotations.
    """
    content = src_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()

    remapped_lines: List[str] = []
    for line in lines:
        remapped = remap_label_line(line, class_mapping, source_file=str(src_file))
        if remapped:
            remapped_lines.append(remapped)

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if remapped_lines:
        dst_file.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
    else:
        dst_file.write_text("", encoding="utf-8")

    return len(remapped_lines)


class DerivedDatasetView:
    """
    Engine to build and manage isolated derived dataset views with remapped labels
    and zero-copy image assets.
    """

    DEFAULT_VEHICLE_2CLASS_MAPPING = {1: 0, 3: 1}
    DEFAULT_VEHICLE_2CLASS_NAMES = {0: "car", 1: "bus"}

    @classmethod
    def create_view(
        cls,
        source_dataset_dir: Union[str, Path],
        target_view_dir: Union[str, Path],
        class_mapping: Optional[Dict[int, int]] = None,
        names: Optional[Dict[int, str]] = None,
        splits: Optional[List[str]] = None,
        selected_files: Optional[Dict[str, List[str]]] = None,
        max_workers: int = 8,
    ) -> Path:
        """
        Constructs a complete derived dataset view.

        Args:
            source_dataset_dir: Root directory of canonical dataset (e.g. dataset_ua_detrac_v001).
            target_view_dir: Destination directory for the isolated derived view.
            class_mapping: Canonical-to-derived class ID mapping (e.g. {1: 0, 3: 1}).
            names: Derived class names dict (e.g. {0: 'car', 1: 'bus'}).
            splits: Splits to include (default: ['train', 'val']).
            selected_files: Optional dict mapping split name to list of specific filenames to include.
            max_workers: ThreadPool workers for fast parallel label remapping.

        Returns:
            Path to the generated data.yaml inside target_view_dir.
        """
        src_root = Path(source_dataset_dir).resolve()
        dst_root = Path(target_view_dir).resolve()
        dst_root.mkdir(parents=True, exist_ok=True)

        mapping = class_mapping if class_mapping is not None else cls.DEFAULT_VEHICLE_2CLASS_MAPPING
        target_names = names if names is not None else cls.DEFAULT_VEHICLE_2CLASS_NAMES
        target_splits = splits or ["train", "val"]

        print(f"[DerivedDatasetView] Initializing derived view at: {dst_root}")
        print(f"  Source canonical dataset: {src_root}")
        print(f"  Class mapping:            {mapping}")
        print(f"  Target names:             {target_names}")
        print(f"  Included splits:          {target_splits}")

        # 1. Setup zero-copy images directory
        dst_images = dst_root / "images"
        dst_images.mkdir(parents=True, exist_ok=True)

        for split in target_splits:
            src_split_img = src_root / "images" / split
            dst_split_img = dst_images / split

            if not src_split_img.is_dir():
                raise FileNotFoundError(f"Source image split not found: {src_split_img}")

            dst_split_img.mkdir(parents=True, exist_ok=True)

            if selected_files and split in selected_files:
                img_names = selected_files[split]
            else:
                img_names = [f.name for f in src_split_img.iterdir() if f.is_file()]

            print(f"  Linking {len(img_names):,} zero-copy image assets for split '{split}'...")

            def _link_img(name: str) -> None:
                sf = src_split_img / name
                df = dst_split_img / name
                if sf.is_file() and not df.exists():
                    try:
                        os.link(str(sf), str(df))
                    except Exception:
                        shutil.copy2(str(sf), str(df))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_link_img, img_names))

        # 2. Materialize remapped labels
        dst_labels = dst_root / "labels"
        dst_labels.mkdir(parents=True, exist_ok=True)

        total_files = 0
        total_boxes = 0

        for split in target_splits:
            src_split_lbl = src_root / "labels" / split
            dst_split_lbl = dst_labels / split
            dst_split_lbl.mkdir(parents=True, exist_ok=True)

            if not src_split_lbl.is_dir():
                raise FileNotFoundError(f"Source label split not found: {src_split_lbl}")

            if selected_files and split in selected_files:
                lbl_names = [
                    Path(f).stem + ".txt" for f in selected_files[split]
                ]
                lbl_files = [src_split_lbl / name for name in lbl_names if (src_split_lbl / name).is_file()]
            else:
                lbl_files = list(src_split_lbl.glob("*.txt"))

            print(f"  Processing {len(lbl_files)} label files for split '{split}'...")

            # Process files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        remap_label_file,
                        src_f,
                        dst_split_lbl / src_f.name,
                        mapping,
                    ): src_f
                    for src_f in lbl_files
                }
                for f in concurrent.futures.as_completed(futures):
                    box_count = f.result()
                    total_boxes += box_count
                    total_files += 1

        print(f"[DerivedDatasetView] Remapped {total_files:,} label files ({total_boxes:,} boxes).")

        # 3. Generate isolated dataset.yaml
        yaml_path = dst_root / "dataset.yaml"
        yaml_content = {
            "path": str(dst_root).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": target_names,
        }
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yaml.safe_dump(yaml_content, yf, sort_keys=False)

        print(f"[DerivedDatasetView] Generated YAML config: {yaml_path}")
        return yaml_path
