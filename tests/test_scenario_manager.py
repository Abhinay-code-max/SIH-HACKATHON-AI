"""
Comprehensive Test Suite for Scenario Manager & Test Bank (Scenarios 01-15).

Validates:
1. Seed 15 Standard Benchmark Scenarios (disk serialization and 5-camera binding).
2. Custom Scenario Creation (SCN_16 Drone Surveillance disk persistence).
3. Category & Difficulty Query Filtering.
4. Atomic Scenario Updates on Disk.
5. Schema Integrity & Scenario Repository Summary Export.
"""

from pathlib import Path
import sys
import json
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.scenario_manager import (
    CameraBinding,
    DifficultyLevel,
    LightingCondition,
    PerimeterZone,
    Scenario,
    ScenarioCategory,
    ScenarioManager,
    ThreatRule,
    TripwireLine,
    WeatherCondition,
    get_default_5_camera_bindings,
    scenario_manager,
)


def test_stage_1_seed_15_scenarios():
    print("\n[Stage 1/5] Testing 15 Master Benchmark Scenarios Seeding & Persistence...")
    seeded = scenario_manager.seed_default_scenarios()
    assert len(seeded) == 15, f"Expected 15 seeded scenarios, got {len(seeded)}"

    # Check files on disk
    scenarios_dir = ROOT_DIR / "training_lab" / "scenarios"
    for i in range(1, 16):
        scn_id = f"SCN_{i:02d}"
        file_path = scenarios_dir / f"{scn_id}.json"
        assert file_path.is_file(), f"Missing scenario file: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["scenario_id"] == scn_id
        assert len(data["assigned_cameras"]) == 5, f"Scenario {scn_id} missing 5-camera bindings"
        for cam_idx in range(1, 6):
            cam_key = f"CAM_{cam_idx:02d}"
            assert cam_key in data["assigned_cameras"], f"Missing {cam_key} in {scn_id}"

    print("  -> Verified all 15 scenarios exist on disk with complete 5-camera bindings.")
    print("  --> PASS: Stage 1 Master Scenarios Seeding Verified.")


def test_stage_2_custom_scenario_creation():
    print("\n[Stage 2/5] Testing Custom Scenario Creation & Disk Persistence...")
    scn_16 = Scenario(
        scenario_id="SCN_16",
        name="Airborne Drone Incursion",
        description="Low-altitude quadcopter breaching perimeter airspace.",
        category=ScenarioCategory.PERIMETER_DEFENSE,
        environment="OPEN_TERRAIN",
        lighting=LightingCondition.DAY_CLEAR,
        weather=WeatherCondition.CLEAR,
        difficulty=DifficultyLevel.HARD,
        assigned_cameras=get_default_5_camera_bindings(),
        threat_rules=[
            ThreatRule(
                rule_id="R_16A",
                name="Airborne Drone Threat",
                event_type="ZONE_INTRUSION",
                severity="CRITICAL",
                points=50,
                target_classes=["drone", "aircraft"],
            )
        ],
    )

    created = scenario_manager.create_scenario(scn_16)
    assert created.scenario_id == "SCN_16"

    # Verify retrieval from fresh manager instance reading disk
    fresh_mgr = ScenarioManager()
    retrieved = fresh_mgr.get_scenario("SCN_16")
    assert retrieved is not None, "Failed to retrieve SCN_16 from disk"
    assert retrieved.name == "Airborne Drone Incursion"
    assert retrieved.threat_rules[0].points == 50
    print(f"  -> Successfully created and verified SCN_16 on disk ({retrieved.name}).")
    print("  --> PASS: Stage 2 Custom Scenario Creation Verified.")


def test_stage_3_filtering_and_querying():
    print("\n[Stage 3/5] Testing Category & Difficulty Query Filtering...")
    # Query Perimeter Defense scenarios
    defense_scenarios = scenario_manager.list_scenarios(category=ScenarioCategory.PERIMETER_DEFENSE)
    assert len(defense_scenarios) >= 4, f"Expected at least 4 perimeter defense scenarios, got {len(defense_scenarios)}"
    for s in defense_scenarios:
        assert s.category == ScenarioCategory.PERIMETER_DEFENSE

    print(f"  -> Found {len(defense_scenarios)} PERIMETER_DEFENSE scenarios: {[s.scenario_id for s in defense_scenarios]}")

    # Query Extreme Difficulty scenarios
    extreme_scenarios = scenario_manager.list_scenarios(difficulty=DifficultyLevel.EXTREME)
    assert len(extreme_scenarios) >= 3, f"Expected at least 3 extreme scenarios, got {len(extreme_scenarios)}"
    for s in extreme_scenarios:
        assert s.difficulty == DifficultyLevel.EXTREME

    print(f"  -> Found {len(extreme_scenarios)} EXTREME difficulty scenarios: {[s.scenario_id for s in extreme_scenarios]}")
    print("  --> PASS: Stage 3 Filtering & Querying Verified.")


def test_stage_4_scenario_updates():
    print("\n[Stage 4/5] Testing Atomic Scenario Updates on Disk...")
    # Update SCN_05 with additional customized restricted zone on CAM_01
    custom_zone = PerimeterZone(
        zone_id="ZONE_VAULT_DOOR",
        name="Vault High-Security Door",
        zone_type="CRITICAL",
        polygon_normalized=[[0.01, 0.01], [0.25, 0.01], [0.25, 0.40], [0.01, 0.40]],
        severity="CRITICAL",
        restricted_classes=["person"],
    )

    orig_scn = scenario_manager.get_scenario("SCN_05")
    assert orig_scn is not None
    updated_cameras = dict(orig_scn.assigned_cameras)
    cam_01_binding = updated_cameras["CAM_01"]
    cam_01_binding.zones.append(custom_zone)
    updated_cameras["CAM_01"] = cam_01_binding

    scenario_manager.update_scenario("SCN_05", {"assigned_cameras": updated_cameras, "description": "Updated Vault Ingress Test."})

    # Reload from disk and verify
    fresh_mgr = ScenarioManager()
    reloaded = fresh_mgr.get_scenario("SCN_05")
    assert reloaded is not None
    assert reloaded.description == "Updated Vault Ingress Test."
    assert any(z.zone_id == "ZONE_VAULT_DOOR" for z in reloaded.assigned_cameras["CAM_01"].zones)

    print("  -> Verified atomic update to SCN_05 (custom vault zone added and persisted).")
    print("  --> PASS: Stage 4 Atomic Updates Verified.")


def test_stage_5_summary_export():
    print("\n[Stage 5/5] Testing Schema Integrity & Summary Export...")
    summary = scenario_manager.export_summary()
    assert summary["total_scenarios"] >= 16, f"Expected at least 16 scenarios, got {summary['total_scenarios']}"
    assert "PERIMETER_DEFENSE" in summary["categories"]
    assert "EASY" in summary["difficulties"]
    assert "EXTREME" in summary["difficulties"]
    assert len(summary["scenario_ids"]) == summary["total_scenarios"]

    print("  -> Scenario Summary Export:")
    print(f"     Total Scenarios: {summary['total_scenarios']}")
    print(f"     Categories:      {summary['categories']}")
    print(f"     Difficulties:    {summary['difficulties']}")

    # Clean up custom SCN_16 so test suite is idempotent
    scenario_manager.delete_scenario("SCN_16")
    fresh_summary = scenario_manager.export_summary()
    assert fresh_summary["total_scenarios"] == 15
    print("  -> Cleaned up custom SCN_16; restored 15 standard master scenarios.")
    print("  --> PASS: Stage 5 Summary Export & Cleanup Verified.")


def run_all_scenario_tests():
    print("=" * 75)
    print("AI TRAINING LAB — SCENARIO MANAGER & MULTI-SCENARIO TEST BANK SUITE")
    print("=" * 75)

    test_stage_1_seed_15_scenarios()
    test_stage_2_custom_scenario_creation()
    test_stage_3_filtering_and_querying()
    test_stage_4_scenario_updates()
    test_stage_5_summary_export()

    print("\n" + "=" * 75)
    print("ALL 5 SCENARIO MANAGER TEST STAGES PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    try:
        run_all_scenario_tests()
        print("\nStatus: SCENARIO MANAGER TEST SUITE SUCCESSFUL")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nStatus: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
