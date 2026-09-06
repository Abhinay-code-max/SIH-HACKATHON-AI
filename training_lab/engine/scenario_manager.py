"""
Scenario Manager Engine & Multi-Scenario Test Bank.
Manages surveillance simulation scenarios, 5-camera stream bindings,
environmental condition parameters, and perimeter threat evaluation rules.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCENARIOS_DIR = ROOT_DIR / "training_lab" / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)


class ScenarioCategory(str, Enum):
    PERIMETER_DEFENSE = "PERIMETER_DEFENSE"
    NIGHT_SURVEILLANCE = "NIGHT_SURVEILLANCE"
    VEHICLE_SURVEILLANCE = "VEHICLE_SURVEILLANCE"
    ANIMAL_INTRUSION = "ANIMAL_INTRUSION"
    CROWD_MONITORING = "CROWD_MONITORING"
    ADVERSE_ENVIRONMENT = "ADVERSE_ENVIRONMENT"
    FALSE_POSITIVE_AUDIT = "FALSE_POSITIVE_AUDIT"
    MIXED_TACTICAL = "MIXED_TACTICAL"


class DifficultyLevel(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    EXTREME = "EXTREME"


class LightingCondition(str, Enum):
    DAY_CLEAR = "DAY_CLEAR"
    DAY_OVERCAST = "DAY_OVERCAST"
    LOW_LIGHT = "LOW_LIGHT"
    NIGHT_IR = "NIGHT_IR"
    HARSH_SHADOWS = "HARSH_SHADOWS"


class WeatherCondition(str, Enum):
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    FOG = "FOG"
    DUST_WIND = "DUST_WIND"
    UNKNOWN = "UNKNOWN"


class PerimeterZone(BaseModel):
    zone_id: str
    name: str
    zone_type: str  # SAFE, WARNING, RESTRICTED, CRITICAL
    polygon_normalized: List[List[float]]  # [[x, y], ...]
    severity: str = "HIGH"
    restricted_classes: List[str] = Field(default_factory=lambda: ["person", "car", "truck", "bus"])


class TripwireLine(BaseModel):
    wire_id: str
    name: str
    line_start: List[float]  # [x1, y1]
    line_end: List[float]    # [x2, y2]
    severity: str = "HIGH"
    target_classes: List[str] = Field(default_factory=lambda: ["person", "car", "truck"])


class CameraBinding(BaseModel):
    camera_id: str  # CAM_01, CAM_02, CAM_03, CAM_04, CAM_05
    stream_name: str
    video_file: Optional[str] = None
    zones: List[PerimeterZone] = Field(default_factory=list)
    tripwires: List[TripwireLine] = Field(default_factory=list)


class ThreatRule(BaseModel):
    rule_id: str
    name: str
    event_type: str
    severity: str = "HIGH"
    points: int = 25
    dwell_threshold_seconds: Optional[float] = None
    target_classes: List[str] = Field(default_factory=list)


class Scenario(BaseModel):
    scenario_id: str  # e.g. SCN_01, SCN_02
    name: str
    description: str
    category: ScenarioCategory
    environment: str  # BORDER_CHECKPOINT, PERIMETER_FENCE, ROADWAY, OPEN_TERRAIN
    lighting: LightingCondition = LightingCondition.DAY_CLEAR
    weather: WeatherCondition = WeatherCondition.CLEAR
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    assigned_cameras: Dict[str, CameraBinding] = Field(default_factory=dict)  # CAM_01 to CAM_05
    threat_rules: List[ThreatRule] = Field(default_factory=list)
    model_version: Optional[str] = "YOLO-L-v002"
    dataset_version: Optional[str] = "v2"
    results_summary: Optional[Dict[str, Any]] = None  # None indicates "NOT TESTED"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def get_default_5_camera_bindings() -> Dict[str, CameraBinding]:
    """Generates standard 5-camera bindings with default zones and tripwires."""
    return {
        "CAM_01": CameraBinding(
            camera_id="CAM_01",
            stream_name="Concourse Main Gateway",
            video_file="training_lab/videos/CAM_01_gateway.mp4",
            zones=[
                PerimeterZone(
                    zone_id="ZONE_GATE_RESTRICTED",
                    name="Perimeter Gateway Corridor",
                    zone_type="RESTRICTED",
                    polygon_normalized=[[0.05, 0.05], [0.45, 0.05], [0.45, 0.65], [0.05, 0.65]],
                    severity="CRITICAL",
                    restricted_classes=["person", "car", "truck"],
                )
            ],
            tripwires=[
                TripwireLine(
                    wire_id="WIRE_GATE_ENTRY",
                    name="Main Ingress Threshold",
                    line_start=[0.30, 0.10],
                    line_end=[0.30, 0.90],
                    severity="HIGH",
                    target_classes=["person", "car"],
                )
            ],
        ),
        "CAM_02": CameraBinding(
            camera_id="CAM_02",
            stream_name="Gate 1 Vehicle Checkpoint",
            video_file="training_lab/videos/CAM_02_checkpoint.mp4",
            zones=[
                PerimeterZone(
                    zone_id="ZONE_VEHICLE_RESTRICTED",
                    name="Vehicle Inspection Bay",
                    zone_type="CRITICAL",
                    polygon_normalized=[[0.20, 0.20], [0.80, 0.20], [0.80, 0.85], [0.20, 0.85]],
                    severity="CRITICAL",
                    restricted_classes=["truck", "bus", "unauthorized_vehicle"],
                )
            ],
            tripwires=[
                TripwireLine(
                    wire_id="WIRE_VEHICLE_STOP_LINE",
                    name="Security Barrier Stop Line",
                    line_start=[0.50, 0.20],
                    line_end=[0.50, 0.85],
                    severity="CRITICAL",
                    target_classes=["car", "truck", "bus"],
                )
            ],
        ),
        "CAM_03": CameraBinding(
            camera_id="CAM_03",
            stream_name="Perimeter Command Outpost",
            video_file="training_lab/videos/CAM_03_perimeter.mp4",
            zones=[
                PerimeterZone(
                    zone_id="ZONE_PERIMETER_NORTH",
                    name="North Border Buffer Zone",
                    zone_type="WARNING",
                    polygon_normalized=[[0.10, 0.10], [0.90, 0.10], [0.90, 0.40], [0.10, 0.40]],
                    severity="HIGH",
                    restricted_classes=["person", "car"],
                )
            ],
            tripwires=[
                TripwireLine(
                    wire_id="WIRE_BORDER_FENCE_NORTH",
                    name="Physical Border Fence",
                    line_start=[0.10, 0.35],
                    line_end=[0.90, 0.35],
                    severity="CRITICAL",
                    target_classes=["person", "animal"],
                )
            ],
        ),
        "CAM_04": CameraBinding(
            camera_id="CAM_04",
            stream_name="Secondary Fence North",
            video_file="training_lab/videos/CAM_04_north_fence.mp4",
            zones=[
                PerimeterZone(
                    zone_id="ZONE_OUTER_FENCE",
                    name="Outer Secondary Buffer",
                    zone_type="RESTRICTED",
                    polygon_normalized=[[0.05, 0.15], [0.95, 0.15], [0.95, 0.75], [0.05, 0.75]],
                    severity="HIGH",
                    restricted_classes=["person", "vehicle"],
                )
            ],
            tripwires=[
                TripwireLine(
                    wire_id="WIRE_OUTER_FENCE_01",
                    name="Secondary Tripwire Line",
                    line_start=[0.05, 0.50],
                    line_end=[0.95, 0.50],
                    severity="HIGH",
                    target_classes=["person"],
                )
            ],
        ),
        "CAM_05": CameraBinding(
            camera_id="CAM_05",
            stream_name="Restricted Logistics Depot",
            video_file="training_lab/videos/CAM_05_logistics.mp4",
            zones=[
                PerimeterZone(
                    zone_id="ZONE_DEPOT_RESTRICTED",
                    name="Ammunition & Fuel Storage Bay",
                    zone_type="CRITICAL",
                    polygon_normalized=[[0.15, 0.15], [0.85, 0.15], [0.85, 0.90], [0.15, 0.90]],
                    severity="CRITICAL",
                    restricted_classes=["person", "car", "truck"],
                )
            ],
            tripwires=[
                TripwireLine(
                    wire_id="WIRE_DEPOT_GATE",
                    name="Depot Inner Perimeter Line",
                    line_start=[0.15, 0.50],
                    line_end=[0.85, 0.50],
                    severity="CRITICAL",
                    target_classes=["person", "truck"],
                )
            ],
        ),
    }


class ScenarioManager:
    """
    Manages loading, updating, listing, and persisting simulation scenarios.
    """

    def __init__(self, scenarios_dir: Optional[Path] = None):
        self.scenarios_dir = Path(scenarios_dir) if scenarios_dir else SCENARIOS_DIR
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Scenario] = {}
        self.reload_from_disk()

    def reload_from_disk(self):
        """Loads all scenario JSON files into memory cache."""
        self._cache.clear()
        for p in self.scenarios_dir.glob("SCN_*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scn = Scenario(**data)
                self._cache[scn.scenario_id] = scn
            except Exception:
                pass

    def create_scenario(self, scenario: Scenario) -> Scenario:
        """Saves a scenario to disk and memory cache."""
        scenario.updated_at = datetime.now(timezone.utc).isoformat()
        filepath = self.scenarios_dir / f"{scenario.scenario_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scenario.model_dump(), f, indent=2)
        self._cache[scenario.scenario_id] = scenario
        return scenario

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Retrieves scenario by ID."""
        if scenario_id in self._cache:
            return self._cache[scenario_id]
        filepath = self.scenarios_dir / f"{scenario_id}.json"
        if filepath.is_file():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    scn = Scenario(**json.load(f))
                self._cache[scenario_id] = scn
                return scn
            except Exception:
                return None
        return None

    def list_scenarios(
        self,
        category: Optional[ScenarioCategory] = None,
        difficulty: Optional[DifficultyLevel] = None,
    ) -> List[Scenario]:
        """Lists scenarios with optional category and difficulty filtering."""
        result = list(self._cache.values())
        if category:
            result = [s for s in result if s.category == category]
        if difficulty:
            result = [s for s in result if s.difficulty == difficulty]
        return sorted(result, key=lambda s: s.scenario_id)

    def update_scenario(self, scenario_id: str, updates: Dict[str, Any]) -> Scenario:
        """Updates specific fields of an existing scenario atomically."""
        scn = self.get_scenario(scenario_id)
        if not scn:
            raise KeyError(f"Scenario not found: {scenario_id}")
        data = scn.model_dump()
        data.update(updates)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated_scn = Scenario(**data)
        return self.create_scenario(updated_scn)

    def delete_scenario(self, scenario_id: str) -> bool:
        """Deletes scenario file from disk and cache."""
        self._cache.pop(scenario_id, None)
        filepath = self.scenarios_dir / f"{scenario_id}.json"
        if filepath.is_file():
            filepath.unlink()
            return True
        return False

    def export_summary(self) -> Dict[str, Any]:
        """Aggregates summary statistics of the scenario repository."""
        scenarios = list(self._cache.values())
        category_counts: Dict[str, int] = {}
        difficulty_counts: Dict[str, int] = {}
        tested_count = 0

        for s in scenarios:
            cat = s.category.value
            diff = s.difficulty.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            if s.results_summary is not None:
                tested_count += 1

        return {
            "total_scenarios": len(scenarios),
            "tested_scenarios": tested_count,
            "untested_scenarios": len(scenarios) - tested_count,
            "categories": category_counts,
            "difficulties": difficulty_counts,
            "scenario_ids": [s.scenario_id for s in sorted(scenarios, key=lambda x: x.scenario_id)],
        }

    def seed_default_scenarios(self) -> List[Scenario]:
        """
        Pre-seeds all 15 master benchmark scenarios with 5-camera configurations.
        """
        raw_defs = [
            (
                "SCN_01",
                "Daytime Person Detection",
                "Clear daytime baseline detection of individuals entering monitored sector.",
                ScenarioCategory.PERIMETER_DEFENSE,
                "BORDER_CHECKPOINT",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.EASY,
                [ThreatRule(rule_id="R_01", name="Person Intrusion", event_type="ZONE_INTRUSION", severity="HIGH", points=25, target_classes=["person"])],
            ),
            (
                "SCN_02",
                "Nighttime Surveillance",
                "Infrared nocturnal perimeter surveillance under complete darkness.",
                ScenarioCategory.NIGHT_SURVEILLANCE,
                "PERIMETER_FENCE",
                LightingCondition.NIGHT_IR,
                WeatherCondition.CLEAR,
                DifficultyLevel.HARD,
                [ThreatRule(rule_id="R_02", name="Night Intrusion", event_type="ZONE_INTRUSION", severity="CRITICAL", points=35, target_classes=["person"])],
            ),
            (
                "SCN_03",
                "Multiple People Approaching Perimeter",
                "Concurrent multi-subject movement toward restricted border sector.",
                ScenarioCategory.PERIMETER_DEFENSE,
                "OPEN_TERRAIN",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.MEDIUM,
                [ThreatRule(rule_id="R_03", name="Coordinated Ingress", event_type="CROWD_DENSITY", severity="HIGH", points=30, target_classes=["person"])],
            ),
            (
                "SCN_04",
                "Vehicle Approaching Restricted Area",
                "High-speed or suspicious vehicle approaching perimeter access gate.",
                ScenarioCategory.VEHICLE_SURVEILLANCE,
                "ROADWAY",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.MEDIUM,
                [ThreatRule(rule_id="R_04", name="Unauthorized Vehicle", event_type="UNAUTHORIZED_VEHICLE", severity="HIGH", points=30, target_classes=["car", "truck"])],
            ),
            (
                "SCN_05",
                "Person Entering Restricted Zone",
                "Individual breaching polygon geofence in restricted door/barrier zone.",
                ScenarioCategory.PERIMETER_DEFENSE,
                "BORDER_CHECKPOINT",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.MEDIUM,
                [ThreatRule(rule_id="R_05", name="Restricted Zone Breach", event_type="ZONE_INTRUSION", severity="CRITICAL", points=35, target_classes=["person"])],
            ),
            (
                "SCN_06",
                "Person Leaving Monitored Zone",
                "Target exiting the perimeter area through verified exit corridor.",
                ScenarioCategory.PERIMETER_DEFENSE,
                "BORDER_CHECKPOINT",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.EASY,
                [ThreatRule(rule_id="R_06", name="Egress Tracking", event_type="OBJECT_DETECTED", severity="LOW", points=10, target_classes=["person"])],
            ),
            (
                "SCN_07",
                "Multiple Vehicles",
                "Heavy multi-vehicle traffic and congestion at border gate.",
                ScenarioCategory.VEHICLE_SURVEILLANCE,
                "ROADWAY",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.HARD,
                [ThreatRule(rule_id="R_07", name="Vehicle Congestion Alert", event_type="UNAUTHORIZED_VEHICLE", severity="HIGH", points=25, target_classes=["truck", "bus", "car"])],
            ),
            (
                "SCN_08",
                "Animal Near Perimeter",
                "Wildlife/animal false-alarm discrimination near fence boundary.",
                ScenarioCategory.ANIMAL_INTRUSION,
                "OPEN_TERRAIN",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.MEDIUM,
                [ThreatRule(rule_id="R_08", name="Animal Proximity", event_type="OBJECT_DETECTED", severity="LOW", points=5, target_classes=["animal"])],
            ),
            (
                "SCN_09",
                "Low-Light CCTV",
                "Degraded low-light CCTV camera feed with optical noise.",
                ScenarioCategory.ADVERSE_ENVIRONMENT,
                "PERIMETER_FENCE",
                LightingCondition.LOW_LIGHT,
                WeatherCondition.FOG,
                DifficultyLevel.HARD,
                [ThreatRule(rule_id="R_09", name="Low-Light Threat", event_type="ZONE_INTRUSION", severity="HIGH", points=30, target_classes=["person"])],
            ),
            (
                "SCN_10",
                "Crowded Environment",
                "High density gathering or crowd surge scenario in public concourse.",
                ScenarioCategory.CROWD_MONITORING,
                "BORDER_CHECKPOINT",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.HARD,
                [ThreatRule(rule_id="R_10", name="Crowd Surge Alert", event_type="CROWD_DENSITY", severity="HIGH", points=30, target_classes=["person"])],
            ),
            (
                "SCN_11",
                "Occluded Person Behind Obstacle",
                "Target partially occluded by barricades, containers, or vegetation.",
                ScenarioCategory.ADVERSE_ENVIRONMENT,
                "PERIMETER_FENCE",
                LightingCondition.DAY_OVERCAST,
                WeatherCondition.RAIN,
                DifficultyLevel.EXTREME,
                [ThreatRule(rule_id="R_11", name="Occluded Infiltration", event_type="ZONE_INTRUSION", severity="CRITICAL", points=40, target_classes=["person"])],
            ),
            (
                "SCN_12",
                "Long-Distance Target (<30px BBox)",
                "Distant perimeter targets appearing smaller than 30 pixels.",
                ScenarioCategory.ADVERSE_ENVIRONMENT,
                "OPEN_TERRAIN",
                LightingCondition.HARSH_SHADOWS,
                WeatherCondition.DUST_WIND,
                DifficultyLevel.EXTREME,
                [ThreatRule(rule_id="R_12", name="Distant Intrusion", event_type="ZONE_INTRUSION", severity="HIGH", points=35, target_classes=["person"])],
            ),
            (
                "SCN_13",
                "Fast-Moving Vehicle (>45 px/step)",
                "High velocity vehicular flight or ramming breach scenario.",
                ScenarioCategory.VEHICLE_SURVEILLANCE,
                "ROADWAY",
                LightingCondition.DAY_CLEAR,
                WeatherCondition.CLEAR,
                DifficultyLevel.HARD,
                [ThreatRule(rule_id="R_13", name="High-Velocity Vehicle Breach", event_type="UNAUTHORIZED_VEHICLE", severity="CRITICAL", points=45, target_classes=["car", "truck"])],
            ),
            (
                "SCN_14",
                "False-Positive Audit (Vegetation/Shadows)",
                "Windblown vegetation, moving shadows, and lighting glare test.",
                ScenarioCategory.FALSE_POSITIVE_AUDIT,
                "PERIMETER_FENCE",
                LightingCondition.HARSH_SHADOWS,
                WeatherCondition.DUST_WIND,
                DifficultyLevel.HARD,
                [ThreatRule(rule_id="R_14", name="False Alarm Suppression Audit", event_type="OBJECT_DETECTED", severity="LOW", points=0, target_classes=[])],
            ),
            (
                "SCN_15",
                "Mixed-Object Tactical Surveillance",
                "Compound tactical scenario with persons, vehicles, and baggage.",
                ScenarioCategory.MIXED_TACTICAL,
                "BORDER_CHECKPOINT",
                LightingCondition.LOW_LIGHT,
                WeatherCondition.FOG,
                DifficultyLevel.EXTREME,
                [
                    ThreatRule(rule_id="R_15A", name="Compound Intrusion", event_type="ZONE_INTRUSION", severity="CRITICAL", points=40, target_classes=["person"]),
                    ThreatRule(rule_id="R_15B", name="Vehicle Threat", event_type="UNAUTHORIZED_VEHICLE", severity="CRITICAL", points=35, target_classes=["truck"]),
                ],
            ),
        ]

        seeded = []
        for (
            scn_id,
            name,
            desc,
            cat,
            env,
            light,
            wth,
            diff,
            rules,
        ) in raw_defs:
            scn = Scenario(
                scenario_id=scn_id,
                name=name,
                description=desc,
                category=cat,
                environment=env,
                lighting=light,
                weather=wth,
                difficulty=diff,
                assigned_cameras=get_default_5_camera_bindings(),
                threat_rules=rules,
                model_version="YOLO-L-v002",
                dataset_version="v2",
                results_summary=None,
            )
            self.create_scenario(scn)
            seeded.append(scn)

        return seeded


scenario_manager = ScenarioManager()
