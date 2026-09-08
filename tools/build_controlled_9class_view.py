import os, sys, shutil, re
from pathlib import Path
from collections import defaultdict
import yaml

ROOT = Path('.').resolve()
ua_root = ROOT / 'training_lab/datasets/dataset_ua_detrac_v001'
uni_root = ROOT / 'training_lab/datasets/dataset_unified_v001'
v2_root = ROOT / 'dataset/releases/v2'

out_root = ROOT / 'training_lab/runs/controlled_9class_dataset_view'
out_root.mkdir(parents=True, exist_ok=True)

TAXONOMY = {
    0: 'person',
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'motorcycle',
    5: 'bicycle',
    6: 'animal',
    7: 'backpack',
    8: 'bag',
}

def link_file(src, dst):
    if not dst.exists():
        try:
            os.link(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))

print("1. Adding dataset_unified_v001 (multi-class core)...")
for split in ['train', 'val']:
    src_img = uni_root / 'images' / split
    src_lbl = uni_root / 'labels' / split
    dst_img = out_root / 'images' / split
    dst_lbl = out_root / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    for f in os.scandir(src_img):
        if f.name.lower().endswith(('.jpg', '.png')):
            stem = f.name.rsplit('.', 1)[0]
            link_file(f.path, dst_img / f'uni_{f.name}')
            
            lbl_p = src_lbl / f'{stem}.txt'
            out_lbl_p = dst_lbl / f'uni_{stem}.txt'
            if lbl_p.is_file():
                out_lbl_p.write_text(lbl_p.read_text(encoding='utf-8'), encoding='utf-8')
            else:
                out_lbl_p.write_text('', encoding='utf-8')

print("2. Adding operational v2 (CCTV threat samples)...")
for split in ['train', 'val']:
    src_img = v2_root / 'images' / split
    src_lbl = v2_root / 'labels' / split
    dst_img = out_root / 'images' / split
    dst_lbl = out_root / 'labels' / split
    
    for f in os.scandir(src_img):
        if f.name.lower().endswith(('.jpg', '.png')):
            stem = f.name.rsplit('.', 1)[0]
            link_file(f.path, dst_img / f'v2_{f.name}')
            
            lbl_p = src_lbl / f'{stem}.txt'
            out_lbl_p = dst_lbl / f'v2_{stem}.txt'
            if lbl_p.is_file():
                out_lbl_p.write_text(lbl_p.read_text(encoding='utf-8'), encoding='utf-8')
            else:
                out_lbl_p.write_text('', encoding='utf-8')

print("3. Adding strided UA-DETRAC (stride=10 for vehicle balance)...")
for split in ['train', 'val']:
    src_img = ua_root / 'images' / split
    src_lbl = ua_root / 'labels' / split
    dst_img = out_root / 'images' / split
    dst_lbl = out_root / 'labels' / split
    
    for f in os.scandir(src_img):
        if f.name.endswith('.jpg'):
            try:
                fnum = int(f.name.split('_f')[-1].split('.')[0])
                if fnum % 10 != 1:
                    continue
            except Exception:
                continue
                
            stem = f.name[:-4]
            link_file(f.path, dst_img / f.name)
            
            lbl_p = src_lbl / f'{stem}.txt'
            out_lbl_p = dst_lbl / f'{stem}.txt'
            out_lines = []
            if lbl_p.is_file():
                for line in lbl_p.read_text(encoding='utf-8').splitlines():
                    p = line.strip().split()
                    if len(p) >= 5:
                        raw_cid = int(p[0])
                        if raw_cid == 1:
                            out_lines.append(f'1 ' + ' '.join(p[1:]))
                        elif raw_cid == 3:
                            out_lines.append(f'3 ' + ' '.join(p[1:]))
            out_lbl_p.write_text('\n'.join(out_lines) + ('\n' if out_lines else ''), encoding='utf-8')

# Write dataset.yaml
yaml_cfg = {
    'path': str(out_root.resolve()).replace('\\', '/'),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 9,
    'names': TAXONOMY,
}
yaml_path = out_root / 'dataset.yaml'
with open(yaml_path, 'w', encoding='utf-8') as yf:
    yaml.safe_dump(yaml_cfg, yf, sort_keys=False)
print(f"Wrote dataset.yaml to: {yaml_path}")

# Calculate Class Statistics across Train and Val
for split in ['train', 'val']:
    lbl_dir = out_root / 'labels' / split
    class_boxes = {c: 0 for c in range(9)}
    class_imgs = {c: set() for c in range(9)}
    total_imgs = len(list((out_root / 'images' / split).glob('*.*')))
    total_boxes = 0
    
    for entry in os.scandir(lbl_dir):
        if not entry.name.endswith('.txt'):
            continue
        stem = entry.name[:-4]
        content = open(entry.path, 'r', encoding='utf-8').read().strip()
        if not content:
            continue
        for line in content.splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                cid = int(p[0])
                class_boxes[cid] += 1
                class_imgs[cid].add(stem)
                total_boxes += 1
                
    print(f"\n=== {split.upper()} CLASS DISTRIBUTION (Total Images: {total_imgs}, Total Boxes: {total_boxes}) ===")
    print("| Class ID | Class Name | Annotations | Images | Percentage |")
    print("|---|---|---|---|---|")
    for cid in range(9):
        cname = TAXONOMY[cid]
        cnt = class_boxes[cid]
        img_cnt = len(class_imgs[cid])
        pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0.0
        print(f"| {cid} | {cname:10s} | {cnt:11d} | {img_cnt:6d} | {pct:9.2f}% |")

print("\nView construction and audit COMPLETE!")
