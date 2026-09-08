"""
Targeted COCO Train2017 Downloader for Weak Classes (Backpack, Bag, Bicycle, Truck).
Downloads 600 official COCO Train2017 images containing high-density annotations
for U1's critically weak classes directly from official COCO image servers.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "training_lab" / "raw_data"
DEST_DIR = RAW_DIR / "coco_train_targeted"
DEST_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = RAW_DIR / "annotations_trainval2017.zip"


def main():
    print(f"Reading instances_train2017.json from {ZIP_PATH}...")
    zf = zipfile.ZipFile(ZIP_PATH)
    with zf.open("annotations/instances_train2017.json") as f:
        data = json.load(f)

    cat_to_name = {c["id"]: c["name"] for c in data["categories"]}
    name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    target_cats = {"backpack", "handbag", "suitcase", "bicycle", "truck"}
    target_cat_ids = {name_to_id[c] for c in target_cats}

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in target_cat_ids and ann.get("iscrowd", 0) == 0:
            w, h = ann["bbox"][2], ann["bbox"][3]
            if w >= 15 and h >= 15:
                anns_by_img[ann["image_id"]].append(ann)

    scored_images = []
    for img_id, anns in anns_by_img.items():
        score = sum(3 if cat_to_name[a["category_id"]] == "backpack" else 2 for a in anns)
        scored_images.append((score, img_id, anns))

    scored_images.sort(key=lambda x: x[0], reverse=True)
    selected = scored_images[:600]

    urls_to_fetch = []
    for score, img_id, anns in selected:
        img_meta = images_by_id[img_id]
        fname = img_meta["file_name"]
        dest_path = DEST_DIR / fname
        url = f"http://images.cocodataset.org/train2017/{fname}"
        urls_to_fetch.append((url, dest_path, img_meta, anns))

    print(f"Starting parallel download of {len(urls_to_fetch)} targeted images...")

    def download_one(item):
        url, dest, meta, anns = item
        if not dest.is_file() or dest.stat().st_size == 0:
            urllib.request.urlretrieve(url, dest)
        return dest

    downloaded = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(download_one, item) for item in urls_to_fetch]
        for f in as_completed(futures):
            f.result()
            downloaded += 1

    print(f"Downloaded {downloaded} images to {DEST_DIR}.")

    # Save targeted annotations subset
    subset_data = {
        "images": [x[2] for x in urls_to_fetch],
        "annotations": [ann for x in urls_to_fetch for ann in x[3]],
        "categories": data["categories"],
    }
    meta_path = RAW_DIR / "coco_train_targeted_annotations.json"
    with open(meta_path, "w", encoding="utf-8") as sf:
        json.dump(subset_data, sf)

    print(f"Saved targeted metadata with {len(subset_data['images'])} images and {len(subset_data['annotations'])} annotations to {meta_path}.")


if __name__ == "__main__":
    main()
