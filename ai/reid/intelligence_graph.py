"""
Surveillance Intelligence Graph Engine.
Represents multi-camera surveillance grids as a zero-dependency directed graph.
Models Entities (Subjects/Vehicles), Physical Sensors (Cameras), and Sightings,
with transit edges connecting camera sectors, journey path tracing,
and graph-based topological anomaly detection (checkpoint skipping, teleportation, cyclic loitering).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_GRAPH_STORAGE = ROOT_DIR / "data" / "sample-events" / "intelligence_graph.json"


@dataclass
class CameraNode:
    camera_id: str
    name: str
    sector: str
    location: Dict[str, float] = field(default_factory=lambda: {"lat": 17.3850, "lon": 78.4867})
    node_type: str = "camera"


@dataclass
class EntityNode:
    entity_id: str
    class_name: str
    first_seen: float
    last_seen: float
    total_sightings: int = 1
    node_type: str = "entity"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationNode:
    obs_id: str
    entity_id: str
    camera_id: str
    timestamp: float
    bbox: List[float]
    confidence: float
    node_type: str = "observation"


@dataclass
class TransitEdge:
    edge_id: str
    entity_id: str
    from_camera: str
    to_camera: str
    transit_duration_sec: float
    similarity_score: float
    timestamp: float
    edge_type: str = "TRANSIT_EDGE"


class SurveillanceIntelligenceGraph:
    """
    Zero-dependency pure-Python directed graph representation of the surveillance grid.
    Stores cameras, entities, and temporal transit edges.
    """

    # Sector topologies and intermediate mandatory checkpoints
    MANDATORY_CHECKPOINTS = {
        ("CAM_01", "CAM_03"): ["CAM_02"],  # Must pass Gate 1 (CAM_02) when going Concourse -> Perimeter
        ("CAM_03", "CAM_01"): ["CAM_02"],
    }

    # Minimum physically possible transit durations in seconds (Teleportation rejection)
    MIN_TRANSIT_TIMES = {
        ("CAM_01", "CAM_02"): 1.0,
        ("CAM_02", "CAM_01"): 1.0,
        ("CAM_02", "CAM_03"): 1.5,
        ("CAM_03", "CAM_02"): 1.5,
        ("CAM_01", "CAM_03"): 2.5,
        ("CAM_03", "CAM_01"): 2.5,
    }

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_GRAPH_STORAGE

        self.cameras: Dict[str, CameraNode] = {}
        self.entities: Dict[str, EntityNode] = {}
        self.observations: List[ObservationNode] = []
        self.transits: List[TransitEdge] = []

        # (entity_id) -> list of camera sighting records sorted by timestamp
        self.entity_sighting_history: Dict[str, List[Dict[str, Any]]] = {}

        self._init_default_cameras()
        self.load_graph()

    def _init_default_cameras(self):
        defaults = [
            CameraNode("CAM_01", "Concourse Main Gateway", "Sector Alpha (Entry)", {"lat": 17.4435, "lon": 78.3765}),
            CameraNode("CAM_02", "Gate 1 Checkpoint", "Sector Bravo (Checkpoint)", {"lat": 17.4442, "lon": 78.3778}),
            CameraNode("CAM_03", "Perimeter Command Outpost", "Sector Charlie (Perimeter)", {"lat": 17.4455, "lon": 78.3792}),
        ]
        for c in defaults:
            self.cameras[c.camera_id] = c

    def add_camera(self, camera_id: str, name: str, sector: str, location: Optional[Dict[str, float]] = None):
        self.cameras[camera_id] = CameraNode(
            camera_id=camera_id,
            name=name,
            sector=sector,
            location=location or {"lat": 17.4435, "lon": 78.3765},
        )

    def add_sighting(
        self,
        subject_id: str,
        camera_id: str,
        timestamp: float,
        bbox: List[float],
        class_name: str = "person",
        confidence: float = 0.90,
    ) -> ObservationNode:
        """Records a timestamped sighting of an entity on a specific camera."""
        if camera_id not in self.cameras:
            self.add_camera(camera_id, f"Camera {camera_id}", "Sector General")

        if subject_id not in self.entities:
            self.entities[subject_id] = EntityNode(
                entity_id=subject_id,
                class_name=class_name,
                first_seen=timestamp,
                last_seen=timestamp,
                total_sightings=1,
            )
        else:
            ent = self.entities[subject_id]
            ent.last_seen = max(ent.last_seen, timestamp)
            ent.total_sightings += 1

        obs_id = f"OBS_{len(self.observations) + 1:06d}"
        obs = ObservationNode(
            obs_id=obs_id,
            entity_id=subject_id,
            camera_id=camera_id,
            timestamp=timestamp,
            bbox=bbox,
            confidence=confidence,
        )
        self.observations.append(obs)

        if subject_id not in self.entity_sighting_history:
            self.entity_sighting_history[subject_id] = []
        self.entity_sighting_history[subject_id].append({
            "obs_id": obs_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "bbox": bbox,
            "confidence": confidence,
        })

        return obs

    def add_transit(
        self,
        from_camera: str,
        to_camera: str,
        subject_id: str,
        transit_sec: float,
        similarity: float,
        timestamp: Optional[float] = None,
    ) -> TransitEdge:
        """Creates a directed transit edge between two cameras for a specific entity."""
        ts = timestamp if timestamp is not None else time.time()
        edge_id = f"TR_{len(self.transits) + 1:04d}"

        edge = TransitEdge(
            edge_id=edge_id,
            entity_id=subject_id,
            from_camera=from_camera,
            to_camera=to_camera,
            transit_duration_sec=round(transit_sec, 2),
            similarity_score=round(similarity, 3),
            timestamp=ts,
        )
        self.transits.append(edge)
        return edge

    def trace_journey(self, subject_id: str) -> List[Dict[str, Any]]:
        """
        Extracts the chronological journey sequence for an entity across camera sectors.
        Calculates dwell times and transitions.
        """
        history = self.entity_sighting_history.get(subject_id, [])
        if not history:
            return []

        # Sort chronologically
        sorted_sightings = sorted(history, key=lambda x: x["timestamp"])

        journey: List[Dict[str, Any]] = []
        current_step: Optional[Dict[str, Any]] = None

        for s in sorted_sightings:
            cam_id = s["camera_id"]
            ts = s["timestamp"]

            if current_step is None:
                current_step = {
                    "camera_id": cam_id,
                    "camera_name": self.cameras[cam_id].name if cam_id in self.cameras else cam_id,
                    "sector": self.cameras[cam_id].sector if cam_id in self.cameras else "Unknown",
                    "entry_time": ts,
                    "exit_time": ts,
                    "dwell_duration_sec": 0.0,
                    "sightings_count": 1,
                }
            elif current_step["camera_id"] == cam_id:
                current_step["exit_time"] = ts
                current_step["dwell_duration_sec"] = round(ts - current_step["entry_time"], 1)
                current_step["sightings_count"] += 1
            else:
                # Camera handoff!
                journey.append(current_step)
                current_step = {
                    "camera_id": cam_id,
                    "camera_name": self.cameras[cam_id].name if cam_id in self.cameras else cam_id,
                    "sector": self.cameras[cam_id].sector if cam_id in self.cameras else "Unknown",
                    "entry_time": ts,
                    "exit_time": ts,
                    "dwell_duration_sec": 0.0,
                    "sightings_count": 1,
                }

        if current_step:
            journey.append(current_step)

        return journey

    def detect_path_anomalies(self, subject_id: str) -> List[Dict[str, Any]]:
        """
        Analyzes an entity's movement sequence across cameras for topological anomalies:
        1. Speed / Teleportation Violations: Transit duration < physical minimum
        2. Checkpoint Skip / Perimeter Bypassing: Transition skips mandatory middle checkpoint
        3. Cyclic / Patrolling Loitering: Repetitive back-and-forth movement across sectors
        """
        anomalies: List[Dict[str, Any]] = []
        subject_transits = [t for t in self.transits if t.entity_id == subject_id]

        # -------------------------------------------------------------
        # 1. TELEPORTATION VIOLATION & CHECKPOINT SKIPPING
        # -------------------------------------------------------------
        for tr in subject_transits:
            cam_pair = (tr.from_camera, tr.to_camera)

            # Check min physical duration
            min_sec = self.MIN_TRANSIT_TIMES.get(cam_pair, 0.8)
            if tr.transit_duration_sec < min_sec:
                anomalies.append({
                    "anomaly_type": "TELEPORTATION_VIOLATION",
                    "severity": "CRITICAL",
                    "subject_id": subject_id,
                    "from_camera": tr.from_camera,
                    "to_camera": tr.to_camera,
                    "transit_duration_sec": tr.transit_duration_sec,
                    "min_physical_sec": min_sec,
                    "timestamp": tr.timestamp,
                    "description": (
                        f"Physical speed violation: Entity {subject_id} transitioned "
                        f"{tr.from_camera} -> {tr.to_camera} in {tr.transit_duration_sec:.1f}s "
                        f"(minimum possible: {min_sec:.1f}s)."
                    ),
                })

            # Check checkpoint skipping
            mandatory = self.MANDATORY_CHECKPOINTS.get(cam_pair, [])
            if mandatory:
                anomalies.append({
                    "anomaly_type": "CHECKPOINT_SKIP_BREACH",
                    "severity": "CRITICAL",
                    "subject_id": subject_id,
                    "from_camera": tr.from_camera,
                    "to_camera": tr.to_camera,
                    "skipped_checkpoints": mandatory,
                    "timestamp": tr.timestamp,
                    "description": (
                        f"Perimeter bypass detected: Entity {subject_id} moved directly from "
                        f"{tr.from_camera} to {tr.to_camera}, bypassing mandatory checkpoint(s): {', '.join(mandatory)}."
                    ),
                })

        # -------------------------------------------------------------
        # 2. CYCLIC / PATROLLING LOITERING
        # -------------------------------------------------------------
        if len(subject_transits) >= 3:
            # Check for oscillating transitions (A -> B -> A)
            for i in range(len(subject_transits) - 1):
                t1 = subject_transits[i]
                t2 = subject_transits[i + 1]
                if t1.from_camera == t2.to_camera and t1.to_camera == t2.from_camera:
                    dt = t2.timestamp - t1.timestamp
                    if dt <= 120.0:
                        anomalies.append({
                            "anomaly_type": "CIRCULAR_PATROLLING_LOITER",
                            "severity": "HIGH",
                            "subject_id": subject_id,
                            "camera_a": t1.from_camera,
                            "camera_b": t1.to_camera,
                            "delta_seconds": round(dt, 1),
                            "timestamp": t2.timestamp,
                            "description": (
                                f"Cyclic patrolling pattern: Entity {subject_id} bounced between "
                                f"{t1.from_camera} and {t1.to_camera} in {dt:.1f}s."
                            ),
                        })
                        break

        return anomalies

    def to_node_link_json(self) -> Dict[str, Any]:
        """
        Converts the entire graph into a standard D3/GIS node-link schema:
        {
          "nodes": [ { "id": "...", "label": "...", "type": "camera"|"entity" }, ... ],
          "links": [ { "source": "...", "target": "...", "type": "...", "weight": ... }, ... ]
        }
        """
        nodes = []
        links = []

        # Add camera nodes
        for c in self.cameras.values():
            nodes.append({
                "id": c.camera_id,
                "label": c.name,
                "sector": c.sector,
                "type": "camera",
                "location": c.location,
            })

        # Add entity nodes
        for e in self.entities.values():
            nodes.append({
                "id": e.entity_id,
                "label": f"{e.class_name.upper()} ({e.entity_id})",
                "class_name": e.class_name,
                "total_sightings": e.total_sightings,
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
                "type": "entity",
            })

        # Add transit links between cameras
        for t in self.transits:
            links.append({
                "id": t.edge_id,
                "source": t.from_camera,
                "target": t.to_camera,
                "entity_id": t.entity_id,
                "duration_sec": t.transit_duration_sec,
                "similarity": t.similarity_score,
                "timestamp": t.timestamp,
                "type": "transit",
            })

        return {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nodes_count": len(nodes),
            "links_count": len(links),
            "nodes": nodes,
            "links": links,
        }

    def save_graph(self, filepath: Optional[Path | str] = None) -> Path:
        """Saves current graph to air-gapped JSON storage."""
        target = Path(filepath or self.storage_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "cameras": [asdict(c) for c in self.cameras.values()],
            "entities": [asdict(e) for e in self.entities.values()],
            "transits": [asdict(t) for t in self.transits[-200:]],
            "observations_count": len(self.observations),
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return target

    def load_graph(self, filepath: Optional[Path | str] = None):
        """Loads graph state from local JSON storage."""
        target = Path(filepath or self.storage_path).resolve()
        if not target.is_file():
            return

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)

            for c_data in data.get("cameras", []):
                c = CameraNode(**c_data)
                self.cameras[c.camera_id] = c

            for e_data in data.get("entities", []):
                e = EntityNode(**e_data)
                self.entities[e.entity_id] = e

            for t_data in data.get("transits", []):
                t = TransitEdge(**t_data)
                self.transits.append(t)
        except Exception:
            pass


surveillance_graph = SurveillanceIntelligenceGraph()
