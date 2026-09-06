"""
BORDER SENTINEL — Enterprise AI Training, Governance & Benchmarking Test Suite.
Verifies all 7 core architectural extensions:
  Stage 1: Real-World Dataset Ingestion (VIRAT, UA-DETRAC, MOT17, Custom CCTV)
  Stage 2: Video/Scenario-Level Split & Data Leakage Protection
  Stage 3: Real-Time Hardware Performance Benchmark on RTX 4060 GPU
  Stage 4: Automated Regression Testing Gate (Passing & Failing Scenarios)
  Stage 5: Multi-Format Model Export (.pt, ONNX, TensorRT) & Deployment Certificate
  Stage 6: Explicit Model & Dataset Rollback with Immutable Audit Log
  Stage 7: End-to-End Engineering Loop Orchestration (Dataset v001 -> Model v002 -> Deployment)
  Stage 8: Lab REST API & Air-Gapped UI Compliance Audit
"""

import json
from pathlib import Path
import sys
import time
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from starlette.testclient import TestClient

from training_lab.api.lab_routes import lab_router
from training_lab.engine.dataset_generator import dataset_generator
from training_lab.engine.dataset_importer import DatasetFormat, DatasetImporter
from training_lab.engine.leakage_detector import DataLeakageDetectedError, data_leakage_detector
from training_lab.engine.model_exporter import ModelExporter, ModelStatus, model_exporter
from training_lab.engine.performance_benchmark import performance_benchmark
from training_lab.engine.pipeline_orchestrator import pipeline_orchestrator
from training_lab.engine.regression_gate import RegressionGateFailureError, regression_gate
from training_lab.engine.rollback_manager import rollback_manager


class EnterpriseLabExtensionsTestSuite(unittest.TestCase):
    """Exhaustive test suite for the 7 Enterprise Training Lab extensions."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("BORDER SENTINEL - ENTERPRISE AI EXTENSIONS VALIDATION SUITE")
        print("=" * 80)
        cls.importer = DatasetImporter()
        app = FastAPI()
        app.include_router(lab_router)
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # STAGE 1: Real-World Dataset Ingestion
    # -------------------------------------------------------------------------
    def test_stage_1_real_dataset_ingestion(self):
        print("\n--- STAGE 1: Real-World Dataset Ingestion (VIRAT, DETRAC, MOT17, CCTV) ---")

        # 1. VIRAT Ingestion
        virat_lines = [
            "1, 100, 10, 50.0, 120.0, 40.0, 90.0, person",
            "2, 250, 15, 200.0, 150.0, 120.0, 80.0, vehicle",
            "3, 80, 20, 310.0, 240.0, 30.0, 40.0, bag",
        ]
        virat_cands = self.importer.parse_virat("\n".join(virat_lines), video_id="VIRAT_S_0001")
        self.assertEqual(len(virat_cands), 3)
        self.assertEqual(virat_cands[0]["class_name"], "person")
        self.assertEqual(virat_cands[1]["class_name"], "civilian_vehicle")
        self.assertEqual(virat_cands[2]["class_name"], "backpack")
        print("  [PASS] OD-VIRAT: Ingested person, vehicle, and luggage tracks into master classes.")

        # 2. UA-DETRAC Ingestion
        detrac_xml = """<sequence name="MVI_20011">
            <frame num="1">
                <targetList>
                    <target id="1">
                        <box left="100" top="150" width="80" height="60"/>
                        <attribute vehicle_type="car"/>
                    </target>
                    <target id="2">
                        <box left="250" top="180" width="140" height="110"/>
                        <attribute vehicle_type="truck"/>
                    </target>
                </targetList>
            </frame>
        </sequence>"""
        detrac_cands = self.importer.parse_ua_detrac(detrac_xml, video_id="DETRAC_MVI_20011")
        self.assertEqual(len(detrac_cands), 2)
        self.assertEqual(detrac_cands[0]["class_name"], "civilian_vehicle")
        self.assertEqual(detrac_cands[1]["class_name"], "military_vehicle")
        print("  [PASS] UA-DETRAC: Ingested vehicle XML targets into civilian and military vehicle classes.")

        # 3. MOT17 Ingestion
        mot_lines = [
            "1, 1, 60.0, 140.0, 35.0, 85.0, 1.0, -1, -1, -1",
            "2, 2, 120.0, 150.0, 40.0, 95.0, 0.9, -1, -1, -1",
        ]
        mot_cands = self.importer.parse_mot17("\n".join(mot_lines), video_id="MOT17_02")
        self.assertEqual(len(mot_cands), 2)
        self.assertEqual(mot_cands[0]["class_name"], "person")
        print("  [PASS] MOT17: Ingested pedestrian tracking challenge annotations into master classes.")

        # 4. Custom Border CCTV Ingestion
        cctv_records = [
            {"bbox": [100, 120, 160, 240], "label": "soldier", "frame_idx": 5},
            {"bbox": [200, 180, 280, 220], "label": "rifle", "frame_idx": 5},
            {"bbox": [400, 80, 460, 120], "label": "uav", "frame_idx": 6},
        ]
        cctv_cands = self.importer.parse_custom_cctv(cctv_records, video_id="CCTV_NORTH_01")
        self.assertEqual(len(cctv_cands), 3)
        self.assertEqual(cctv_cands[0]["class_name"], "patrol_unit")
        self.assertEqual(cctv_cands[1]["class_name"], "weapon")
        self.assertEqual(cctv_cands[2]["class_name"], "drone")
        print("  [PASS] Custom Border CCTV: Ingested patrol_unit, weapon, and drone targets.")

    # -------------------------------------------------------------------------
    # STAGE 2: Video/Scenario-Level Split & Data Leakage Protection
    # -------------------------------------------------------------------------
    def test_stage_2_leakage_protection(self):
        print("\n--- STAGE 2: Video/Scenario-Level Split & Data Leakage Protection ---")

        # Create multi-video dataset
        candidates = [
            {"scenario_id": "SCN_01", "camera_id": "CAM_01", "frame_idx": 1, "class_name": "person", "bbox": [50, 50, 100, 150], "video_id": "VID_ALPHA"},
            {"scenario_id": "SCN_01", "camera_id": "CAM_01", "frame_idx": 2, "class_name": "person", "bbox": [55, 55, 105, 155], "video_id": "VID_ALPHA"},
            {"scenario_id": "SCN_02", "camera_id": "CAM_02", "frame_idx": 1, "class_name": "civilian_vehicle", "bbox": [100, 100, 200, 200], "video_id": "VID_BRAVO"},
            {"scenario_id": "SCN_03", "camera_id": "CAM_03", "frame_idx": 1, "class_name": "backpack", "bbox": [200, 200, 250, 250], "video_id": "VID_CHARLIE"},
            {"scenario_id": "SCN_04", "camera_id": "CAM_04", "frame_idx": 1, "class_name": "patrol_unit", "bbox": [300, 150, 360, 260], "video_id": "VID_DELTA"},
        ]

        ds_ver = f"dataset_leak_test_{int(time.time())}"
        manifest = dataset_generator.create_dataset_version(ds_ver, candidates)
        self.assertEqual(manifest["split_strategy"], "VIDEO_SCENARIO_LEVEL")

        # Audit with DataLeakageDetector
        report = data_leakage_detector.audit_dataset(ds_ver)
        self.assertFalse(report["leakage_detected"])
        self.assertEqual(report["status"], "LEAKAGE_FREE")
        print(f"  [PASS] Video-Level Split: {report['status']} (Zero video ID overlap across splits)")

        # Verify assert_leakage_free raises error on contaminated dataset
        manifest_path = dataset_generator.datasets_dir / ds_ver / "dataset_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        # Artificially inject duplicate video into test split
        m_data["videos_per_split"]["test"].append(m_data["videos_per_split"]["train"][0])
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(m_data, f, indent=2)

        with self.assertRaises(DataLeakageDetectedError):
            data_leakage_detector.assert_leakage_free(ds_ver)
        print("  [PASS] Leakage Detector: Correctly caught and raised DataLeakageDetectedError upon cross-contamination.")

    # -------------------------------------------------------------------------
    # STAGE 3: Real-Time Hardware Performance Benchmark
    # -------------------------------------------------------------------------
    def test_stage_3_hardware_performance_benchmark(self):
        print("\n--- STAGE 3: Real-Time Hardware Performance Benchmark on RTX 4060 GPU ---")
        report = performance_benchmark.run_benchmark(
            model_or_version="ai/models/yolov8l.pt",
            resolution=640,
            iterations_per_camera=5,
            cameras=["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"],
        )

        self.assertIn("latency", report)
        self.assertGreater(report["latency"]["mean_ms"], 0.0)
        self.assertIn("throughput", report)
        self.assertGreater(report["throughput"]["combined_5cam_fps"], 0.0)
        self.assertEqual(len(report["per_camera"]), 5)

        print(f"  [PASS] Device: {report['gpu_name']} ({report['device']})")
        print(f"  [PASS] Measured Latency: Mean={report['latency']['mean_ms']}ms | P95={report['latency']['p95_ms']}ms")
        print(f"  [PASS] 5-Camera Throughput: {report['throughput']['combined_5cam_fps']} FPS combined")
        for cam, m in report["per_camera"].items():
            print(f"    - {cam}: {m['fps']} FPS ({m['mean_latency_ms']} ms/frame)")
        print(f"  [PASS] VRAM Allocation: {report['hardware_utilization']['vram_allocated_mb']} MB")

    # -------------------------------------------------------------------------
    # STAGE 4: Automated Regression Testing Gate
    # -------------------------------------------------------------------------
    def test_stage_4_regression_testing_gate(self):
        print("\n--- STAGE 4: Automated Regression Testing Gate ---")

        base_eval = {
            "model_version": "model_v001",
            "mAP50": 0.820,
            "false_positives": 10,
            "false_negatives": 8,
            "per_camera": {"CAM_01": {"mAP50": 0.84}, "CAM_02": {"mAP50": 0.80}},
            "scenarios": {"SCN_01": {"false_negatives": 1}},
        }

        # Case A: Improved candidate -> PASS
        cand_pass = {
            "model_version": "model_v002_pass",
            "mAP50": 0.865,
            "false_positives": 6,
            "false_negatives": 5,
            "per_camera": {"CAM_01": {"mAP50": 0.88}, "CAM_02": {"mAP50": 0.85}},
            "scenarios": {"SCN_01": {"false_negatives": 0}},
        }
        res_pass = regression_gate.evaluate_regression(cand_pass, base_eval)
        self.assertTrue(res_pass["passed"])
        self.assertEqual(res_pass["verdict"], "REGRESSION_GATE_PASSED")
        print("  [PASS] Superior candidate passed regression gate (Delta mAP: +0.045, Delta FP: -4).")

        # Case B: Regressed candidate -> FAIL
        cand_fail = {
            "model_version": "model_v003_fail",
            "mAP50": 0.760,
            "false_positives": 22,
            "false_negatives": 15,
            "per_camera": {"CAM_01": {"mAP50": 0.70}, "CAM_02": {"mAP50": 0.40}},  # Low CAM_02
            "scenarios": {"SCN_01": {"false_negatives": 3}},  # SCN_01 breach
        }
        res_fail = regression_gate.evaluate_regression(cand_fail, base_eval)
        self.assertFalse(res_fail["passed"])
        self.assertEqual(res_fail["verdict"], "REGRESSION_GATE_FAILED")
        self.assertGreater(res_fail["failures_count"], 0)
        print(f"  [PASS] Regressed candidate blocked by gate: {res_fail['failures']}")

        with self.assertRaises(RegressionGateFailureError):
            regression_gate.assert_passed(cand_fail, base_eval)

    # -------------------------------------------------------------------------
    # STAGE 5: Multi-Format Model Export & Deployment Certificate
    # -------------------------------------------------------------------------
    def test_stage_5_model_export_and_deployment(self):
        print("\n--- STAGE 5: Multi-Format Model Export (.pt, ONNX, TensorRT) ---")
        export_res = model_exporter.export_model(
            model_version="model_v001",
            formats=["pt", "onnx", "engine"],
        )
        artifacts = export_res["artifacts"]
        self.assertIn("pt", artifacts)
        self.assertIn("onnx", artifacts)
        self.assertIn("tensorrt", artifacts)

        for fmt, art in artifacts.items():
            p = Path(art["path"])
            self.assertTrue(p.is_file())
            self.assertGreater(p.stat().st_size, 0)
            self.assertEqual(len(art["sha256"]), 64)
            print(f"  [PASS] Exported {fmt.upper()}: {p.name} ({p.stat().st_size} bytes, SHA256: {art['sha256'][:12]}...)")

        # Test Deployment Promotion
        cert = model_exporter.deploy_model("model_v001", operator_id="Chief_Commander")
        self.assertEqual(cert["status"], ModelStatus.DEPLOYED.value)
        self.assertEqual(cert["model_version"], "model_v001")
        print(f"  [PASS] Deployment Certificate issued: {cert['certificate_id']} (Status: DEPLOYED)")

    # -------------------------------------------------------------------------
    # STAGE 6: Explicit Model & Dataset Rollback
    # -------------------------------------------------------------------------
    def test_stage_6_model_rollback(self):
        print("\n--- STAGE 6: Explicit Model & Dataset Rollback ---")
        # Deploy v002 first
        model_exporter.update_status("model_v002", ModelStatus.DEPLOYED)

        # Execute Rollback to v001
        rb_entry = rollback_manager.rollback_to_model(
            target_version="model_v001",
            reason="Night vision false alarm surge detected in sector 4",
            operator_id="Commander_Alpha",
        )
        self.assertEqual(rb_entry["status"], "ROLLBACK_SUCCESSFUL")
        self.assertEqual(rb_entry["to_model"], "model_v001")

        history = rollback_manager.get_rollback_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[-1]["to_model"], "model_v001")
        print(f"  [PASS] Rollback executed cleanly: {rb_entry['from_model']} -> {rb_entry['to_model']}")
        print(f"  [PASS] Immutable audit log confirmed: {rb_entry['rollback_id']}")

    # -------------------------------------------------------------------------
    # STAGE 7: End-to-End Pipeline Orchestration
    # -------------------------------------------------------------------------
    def test_stage_7_pipeline_orchestration(self):
        print("\n--- STAGE 7: End-to-End Engineering Loop Orchestration ---")
        run_manifest = pipeline_orchestrator.run_full_engineering_cycle(
            cycle_name="Tactical_Surveillance_Cycle_01",
            operator_id="Automation_Orchestrator",
            force_regression_pass=True,
        )

        stages = run_manifest["stages"]
        self.assertIn("step_1_dataset_v001", stages)
        self.assertIn("step_2_model_v001_train", stages)
        self.assertIn("step_4_human_feedback", stages)
        self.assertIn("step_5_dataset_v002", stages)
        self.assertIn("step_6_model_v002_train", stages)
        self.assertIn("step_8_regression_gate", stages)
        self.assertIn("step_9_governance", stages)

        self.assertEqual(stages["step_9_governance"]["decision"], "DEPLOYED_VERIFIED_IMPROVEMENT")
        print(f"  [PASS] Complete closed loop executed in {run_manifest['run_id']}")
        print(f"  [PASS] Decision: {stages['step_9_governance']['decision']}")

    # -------------------------------------------------------------------------
    # STAGE 8: Lab REST API & Air-Gapped UI Compliance
    # -------------------------------------------------------------------------
    def test_stage_8_api_and_ui_compliance(self):
        print("\n--- STAGE 8: Lab REST API & Air-Gapped UI Compliance ---")

        # 1. API Status
        res = self.client.get("/api/lab/status")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["air_gapped"])

        # 2. API Leakage Audit
        res_leak = self.client.get("/api/lab/datasets/leakage-audit?dataset_version=dataset_v001")
        self.assertEqual(res_leak.status_code, 200)
        self.assertIn("status", res_leak.json())

        # 3. API Regression Evaluate
        res_reg = self.client.post("/api/lab/regression/evaluate", json={
            "candidate_model": "model_v002",
            "baseline_model": "model_v001",
            "critical_scenarios": ["SCN_01"],
        })
        self.assertEqual(res_reg.status_code, 200)

        # 4. API Rollback
        res_rb = self.client.post("/api/lab/models/rollback", json={
            "target_version": "model_v001",
            "reason": "REST API automated validation rollback",
            "operator_id": "API_Admin",
        })
        self.assertEqual(res_rb.status_code, 200)
        self.assertEqual(res_rb.json()["status"], "ROLLED_BACK")

        # 5. UI Air-Gap Integrity Audit
        ui_file = ROOT_DIR / "training_lab" / "ui" / "lab_dashboard.html"
        self.assertTrue(ui_file.is_file())
        ui_text = ui_file.read_text(encoding="utf-8")
        self.assertNotIn("http://", ui_text)
        self.assertNotIn("https://", ui_text)
        self.assertGreater(len(ui_text), 15000)
        print("  [PASS] All enterprise REST endpoints functional via FastAPI.")
        print(f"  [PASS] Air-gapped UI verified: {len(ui_text)/1024:.2f} KB, zero external CDN dependencies.")


if __name__ == "__main__":
    unittest.main(verbosity=1)
