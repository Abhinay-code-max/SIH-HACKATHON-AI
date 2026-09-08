import gc, json, os, sys, time, yaml
from pathlib import Path
from typing import Any, Dict, List
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.dataset_view import DerivedDatasetView
from training_lab.engine.training_manager import format_weights_path

def select_smoke_dataset_files(canonical_root: Path) -> Dict[str, List[str]]:
    train_files = sorted([f.name for f in (canonical_root / 'images' / 'train').glob('dataset_ua_detrac_v001_train_MVI_20034_*.jpg')])
    mvi_20012 = sorted([f.name for f in (canonical_root / 'images' / 'train').glob('dataset_ua_detrac_v001_train_MVI_20012_*.jpg')])
    train_files.extend(mvi_20012[:700])
    val_files = sorted([f.name for f in (canonical_root / 'images' / 'val').glob('dataset_ua_detrac_v001_val_MVI_40204_f*.jpg') if 400 <= int(f.stem.split('_f')[1]) <= 650])
    val_files.extend(sorted([f.name for f in (canonical_root / 'images' / 'val').glob('dataset_ua_detrac_v001_val_MVI_40243_f*.jpg') if 300 <= int(f.stem.split('_f')[1]) <= 450]))
    return {'train': train_files, 'val': val_files}

def run_smoke_training() -> Dict[str, Any]:
    print('=' * 80)
    print('PHASE D.3: PART 1 -- FAST REAL TRAINING SMOKE TEST (1-EPOCH YOLOv8s)')
    print('=' * 80)
    canonical_root = (ROOT_DIR / 'training_lab' / 'datasets' / 'dataset_ua_detrac_v001').resolve()
    assert canonical_root.is_dir(), f'Canonical dataset missing: {canonical_root}'
    selected = select_smoke_dataset_files(canonical_root)
    print(f'Selected train frames: {len(selected["train"]):,} (MVI_20034, MVI_20012)')
    print(f'Selected val frames:   {len(selected["val"]):,} (MVI_40204, MVI_40243)')
    smoke_run_dir = ROOT_DIR / 'training_lab' / 'runs' / 'D3_smoke_test'
    smoke_view_dir = smoke_run_dir / 'dataset_view'
    print(f'Materializing isolated derived smoke view: {smoke_view_dir}')
    data_yaml_path = DerivedDatasetView.create_view(
        source_dataset_dir=canonical_root,
        target_view_dir=smoke_view_dir,
        class_mapping={1: 0, 3: 1},
        names={0: 'car', 1: 'bus'},
        splits=['train', 'val'],
        selected_files=selected,
    )
    cfg_data = check_det_dataset(str(data_yaml_path))
    val_resolved = str(Path(cfg_data['val']).resolve())
    assert str(smoke_view_dir.resolve()).lower() in val_resolved.lower(), f'Validation path escaped smoke view: {val_resolved}'
    print(f'Verified path isolation: val resolved to {val_resolved}')
    base_model_path = ROOT_DIR / 'yolov8s.pt'
    assert base_model_path.is_file(), f'Base weights yolov8s.pt missing: {base_model_path}'
    model = YOLO(str(base_model_path))
    t_start = time.perf_counter()
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=1,
        batch=16,
        imgsz=640,
        project=str((ROOT_DIR / 'training_lab' / 'runs').resolve()),
        name='D3_smoke_test',
        device='0',
        workers=0,
        plots=True,
        save=True,
        verbose=True,
        exist_ok=True,
    )
    t_train = time.perf_counter() - t_start
    weights_dir = smoke_run_dir / 'weights'
    best_pt = weights_dir / 'best.pt'
    assert best_pt.is_file(), f'Checkpoint best.pt not found: {best_pt}'
    print(f'Checkpoint verified: {best_pt} ({best_pt.stat().st_size / (1024**2):.1f} MB)')
    print('Inspecting validator metrics on trained checkpoint...')
    val_model = YOLO(str(best_pt))
    val_results = val_model.val(
        data=str(data_yaml_path.resolve()),
        split='val',
        batch=16,
        device='0',
        plots=False,
        verbose=True,
    )
    names = val_results.names
    ap_classes = getattr(val_results.box, 'ap_class_index', None)
    all_map50 = float(val_results.results_dict.get('metrics/mAP50(B)', 0.0))
    all_map50_95 = float(val_results.results_dict.get('metrics/mAP50-95(B)', 0.0))
    all_prec = float(val_results.results_dict.get('metrics/precision(B)', 0.0))
    all_rec = float(val_results.results_dict.get('metrics/recall(B)', 0.0))

    per_class_metrics = {}
    if ap_classes is not None:
        for i, c in enumerate(ap_classes):
            c_name = names.get(c, str(c))
            cp, cr, cap50, cap = val_results.box.class_result(i)
            per_class_metrics[c_name] = {
                'precision': round(float(cp) * 100, 2),
                'recall': round(float(cr) * 100, 2),
                'mAP50': round(float(cap50) * 100, 2),
                'mAP50_95': round(float(cap) * 100, 2),
            }

    print(f'Validator Model Names:  {names}')
    print(f'Validator ap_classes:   {ap_classes}')
    print(f'Overall mAP50:          {all_map50 * 100:.2f}%')
    print(f'Overall mAP50-95:       {all_map50_95 * 100:.2f}%')
    for cn, cm in per_class_metrics.items():
        print(f'Class {cn:5s} -> P: {cm["precision"]}%, R: {cm["recall"]}%, mAP50: {cm["mAP50"]}%, mAP50-95: {cm["mAP50_95"]}%')

    val_cache_file = smoke_view_dir / 'labels' / 'val.cache'
    assert val_cache_file.is_file(), f'Expected cache in smoke view: {val_cache_file}'
    cache_data = np.load(str(val_cache_file), allow_pickle=True).item()
    val_targets = {0: 0, 1: 0}
    for lb in cache_data.get('labels', []):
        cls_arr = lb.get('cls')
        if cls_arr is not None and len(cls_arr) > 0:
            for c in cls_arr.flatten().tolist():
                val_targets[int(c)] += 1
    print(f'Validation Target Counts: Car={val_targets[0]:,}, Bus={val_targets[1]:,}')

    summary_record = {
        'experiment': 'D3_smoke_test',
        'base_model': 'yolov8s.pt',
        'epochs': 1,
        'batch': 16,
        'imgsz': 640,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'train_time_seconds': round(t_train, 2),
        'smoke_dataset': {
            'train_frames': len(selected['train']),
            'val_frames': len(selected['val']),
            'train_sequences': ['MVI_20034', 'MVI_20012'],
            'val_sequences': ['MVI_40204', 'MVI_40243'],
            'val_targets': val_targets,
        },
        'weights_path': format_weights_path(best_pt),
        'metrics': {
            'all': {'precision': round(all_prec * 100, 2), 'recall': round(all_rec * 100, 2), 'mAP50': round(all_map50 * 100, 2), 'mAP50_95': round(all_map50_95 * 100, 2)},
            'per_class': per_class_metrics,
        },
        'validator': {
            'names': names,
            'ap_class_index': [int(x) for x in ap_classes] if ap_classes is not None else [],
        },
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    summary_file = smoke_run_dir / 'training_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_record, f, indent=2)
    print(f'Training summary saved to {summary_file}')
    return summary_record

def run_throughput_benchmark(worker_counts: List[int] = [0, 4, 8]) -> Dict[str, Any]:
    print('=' * 80)
    print('PHASE D.3: PART 2 -- DATA-LOADING THROUGHPUT BENCHMARK')
    print('=' * 80)
    data_yaml_path = ROOT_DIR / 'training_lab' / 'runs' / 'D3_smoke_test' / 'dataset_view' / 'dataset.yaml'
    assert data_yaml_path.is_file(), f'Smoke view dataset.yaml missing: {data_yaml_path}'
    with open(data_yaml_path, 'r', encoding='utf-8') as yf:
        data_cfg = yaml.safe_load(yf)
    smoke_images_train = str(data_yaml_path.parent / 'images' / 'train')
    batch_size = 16
    n_benchmark_batches = 50
    results_table = []
    for workers in worker_counts:
        print(f'Testing workers={workers}...')
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        dataset = YOLODataset(
            img_path=smoke_images_train,
            imgsz=640,
            data=data_cfg,
            task='detect',
            augment=True,
            batch_size=batch_size,
        )
        dataloader = build_dataloader(
            dataset=dataset,
            batch=batch_size,
            workers=workers,
            shuffle=True,
        )
        it = iter(dataloader)
        for _ in range(3):
            _ = next(it)
        t0 = time.perf_counter()
        batches_done = 0
        for _ in range(n_benchmark_batches):
            try:
                batch = next(it)
                if torch.cuda.is_available():
                    _ = batch['img'].to('cuda', non_blocking=True)
                batches_done += 1
            except StopIteration:
                break
        t_elapsed = time.perf_counter() - t0
        sec_per_iter = t_elapsed / batches_done if batches_done > 0 else 0.0
        batches_per_sec = batches_done / t_elapsed if t_elapsed > 0 else 0.0
        images_per_sec = (batches_done * batch_size) / t_elapsed if t_elapsed > 0 else 0.0
        gpu_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
        row = {
            'workers': workers,
            'batches_done': batches_done,
            'images_done': batches_done * batch_size,
            'total_time_sec': round(t_elapsed, 3),
            'sec_per_iter': round(sec_per_iter, 4),
            'batches_per_sec': round(batches_per_sec, 2),
            'images_per_sec': round(images_per_sec, 2),
            'peak_gpu_mem_mb': round(gpu_mem_mb, 1),
        }
        results_table.append(row)
        print(f'  workers={workers}: {sec_per_iter*1000:.1f} ms/iter | {batches_per_sec:.2f} batches/s | {images_per_sec:.1f} img/s | peak VRAM={gpu_mem_mb:.1f} MB | total={t_elapsed:.2f}s')
    print('Probing OneDrive raw image read latency...')
    src_images = list((ROOT_DIR / 'training_lab' / 'datasets' / 'dataset_ua_detrac_v001' / 'images' / 'train').glob('*.jpg'))[:200]
    t0_io = time.perf_counter()
    total_bytes = 0
    for f in src_images:
        b = f.read_bytes()
        total_bytes += len(b)
    t_io = time.perf_counter() - t0_io
    io_throughput_mb_s = (total_bytes / (1024**2)) / t_io if t_io > 0 else 0.0
    ms_per_file = (t_io / len(src_images)) * 1000 if src_images else 0.0
    print(f'  Direct File Read Speed: {io_throughput_mb_s:.1f} MB/s ({ms_per_file:.2f} ms/image across {len(src_images)} images)')
    benchmark_record = {
        'benchmark_batches': n_benchmark_batches,
        'batch_size': batch_size,
        'results': results_table,
        'direct_io': {
            'read_mb_per_sec': round(io_throughput_mb_s, 2),
            'ms_per_image': round(ms_per_file, 2),
        },
    }
    bench_file = ROOT_DIR / 'training_lab' / 'runs' / 'D3_smoke_test' / 'throughput_benchmark.json'
    with open(bench_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_record, f, indent=2)
    print(f'Benchmark results saved to {bench_file}')
    return benchmark_record

if __name__ == '__main__':
    smoke_summary = run_smoke_training()
    benchmark_summary = run_throughput_benchmark()
    print('PHASE D.3 EXECUTION COMPLETE')
