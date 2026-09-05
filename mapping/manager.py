"""
Offline Mapping Manager for Phase 1 Prototype.
Loads local GeoJSON layers (zones, cameras, roads), associates events with coordinates,
and generates offline map assets without requiring external tile servers or internet.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.models.contracts import CameraAsset, GeoLocation, SecurityEvent


class OfflineMapManager:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.geojson_dir = self.base_dir / "geojson"
        self.processed_dir = self.base_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.zones = self._load_json(self.geojson_dir / "zones.geojson")
        self.cameras_geojson = self._load_json(self.geojson_dir / "cameras.geojson")
        self.roads = self._load_json(self.geojson_dir / "roads.geojson")

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing required offline map asset: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_camera_locations(self) -> Dict[str, Dict[str, Any]]:
        """Returns dict of camera_id -> coordinate & details."""
        cam_map = {}
        for feat in self.cameras_geojson.get("features", []):
            cid = feat["properties"]["camera_id"]
            lon, lat = feat["geometry"]["coordinates"]
            cam_map[cid] = {
                "camera_id": cid,
                "name": feat["properties"]["name"],
                "lat": lat,
                "lon": lon,
                "zone_id": feat["properties"].get("zone_id"),
                "status": feat["properties"].get("status", "ONLINE"),
            }
        return cam_map

    def build_integrated_map_payload(self, events_file: Path | None = None) -> Dict[str, Any]:
        """
        Combines zones, roads, cameras, and live security events into a single
        offline-ready map payload for the frontend.
        """
        camera_dict = self.get_camera_locations()

        # Load live events if present
        if events_file is None:
            events_file = ROOT_DIR / "data" / "sample-events" / "live_events.json"

        event_markers = []
        if events_file.is_file():
            with open(events_file, "r", encoding="utf-8") as f:
                events_data = json.load(f)
            for ev in events_data:
                src_id = ev.get("source_id", "CAM_01")
                cam_info = camera_dict.get(src_id, {})
                lat = ev.get("location", {}).get("lat", cam_info.get("lat", 17.4435))
                lon = ev.get("location", {}).get("lon", cam_info.get("lon", 78.3765))

                event_markers.append({
                    "event_id": ev.get("event_id"),
                    "type": ev.get("event_type"),
                    "class": ev.get("class_name"),
                    "severity": ev.get("severity"),
                    "timestamp": ev.get("timestamp"),
                    "source_id": src_id,
                    "lat": lat,
                    "lon": lon,
                    "zone_name": ev.get("location", {}).get("zone_name", "Demonstration Perimeter"),
                })

        # Load cross-camera transit trails from Re-ID engine
        transits_list = []
        try:
            from ai.reid.manager import global_subject_manager
            transits_list = global_subject_manager.get_recent_transits(limit=10)
        except Exception:
            pass

        payload = {
            "center": [17.4445, 78.3780],
            "bounds": [
                [17.4410, 78.3740],
                [17.4490, 78.3820],
            ],
            "zones": self.zones,
            "roads": self.roads,
            "cameras": list(camera_dict.values()),
            "event_markers": event_markers,
            "transits": transits_list,
        }

        # Save processed bundle
        out_json = self.processed_dir / "map_overlay.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload

    def generate_offline_html_preview(self, payload: Dict[str, Any]) -> Path:
        """
        Generates a 100% self-contained offline SVG/HTML map preview.
        Loads ZERO external scripts, ZERO CDNs, ZERO external fonts or tiles.
        """
        html_file = self.processed_dir / "offline_map_preview.html"

        # Bounds for local coordinate transformation to SVG viewbox 0 0 1000 800
        min_lat, min_lon = 17.4410, 78.3740
        max_lat, max_lon = 17.4490, 78.3820

        def to_svg(lon, lat):
            x = (lon - min_lon) / (max_lon - min_lon) * 900 + 50
            y = 750 - (lat - min_lat) / (max_lat - min_lat) * 700
            return round(x, 1), round(y, 1)

        # SVG Polygons for Zones
        zones_svg = []
        for feat in payload["zones"]["features"]:
            coords = feat["geometry"]["coordinates"][0]
            pts = " ".join([f"{to_svg(p[0], p[1])[0]},{to_svg(p[0], p[1])[1]}" for p in coords])
            prop = feat["properties"]
            color = prop.get("color", "#3B82F6")
            name = prop.get("name")
            zones_svg.append(f"""
            <polygon points="{pts}" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="2">
                <title>{name}</title>
            </polygon>
            """)

        # SVG Lines for Roads
        roads_svg = []
        for feat in payload["roads"]["features"]:
            coords = feat["geometry"]["coordinates"]
            pts = " ".join([f"{to_svg(p[0], p[1])[0]},{to_svg(p[0], p[1])[1]}" for p in coords])
            roads_svg.append(f"""
            <polyline points="{pts}" stroke="#475569" stroke-width="6" stroke-linecap="round" fill="none" />
            <polyline points="{pts}" stroke="#94A3B8" stroke-width="2" stroke-dasharray="8,6" fill="none" />
            """)

        # SVG Points for Cameras
        cams_svg = []
        for cam in payload["cameras"]:
            x, y = to_svg(cam["lon"], cam["lat"])
            cams_svg.append(f"""
            <g transform="translate({x}, {y})">
                <circle r="12" fill="#1E293B" stroke="#38BDF8" stroke-width="3" />
                <circle r="4" fill="#38BDF8" />
                <text x="16" y="5" font-family="monospace" font-size="12" fill="#E2E8F0" font-weight="bold">{cam['camera_id']} ({cam['status']})</text>
            </g>
            """)

        # SVG Markers for Events
        evts_svg = []
        for ev in payload["event_markers"]:
            x, y = to_svg(ev["lon"], ev["lat"])
            sev = ev.get("severity", "low").lower()
            sev_color = "#EF4444" if sev == "critical" else ("#F59E0B" if sev in {"high", "medium"} else "#10B981")
            evts_svg.append(f"""
            <g transform="translate({x}, {y})">
                <circle r="18" fill="{sev_color}" fill-opacity="0.3">
                    <animate attributeName="r" values="12;24;12" dur="2s" repeatCount="indefinite"/>
                    <animate attributeName="fill-opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite"/>
                </circle>
                <circle r="8" fill="{sev_color}" stroke="#FFFFFF" stroke-width="2"/>
                <text x="14" y="-12" font-family="monospace" font-size="11" fill="{sev_color}" font-weight="bold">[{sev.upper()}] {ev['type']} ({ev['class']})</text>
            </g>
            """)

        # SVG Transit Vectors for Cross-Camera Re-ID Hand-offs
        transits_svg = []
        cams_coord_map = {cam["camera_id"]: (cam["lon"], cam["lat"]) for cam in payload.get("cameras", [])}
        for tr in payload.get("transits", []):
            from_c = tr.get("from_camera")
            to_c = tr.get("to_camera")
            if from_c in cams_coord_map and to_c in cams_coord_map and from_c != to_c:
                x1, y1 = to_svg(*cams_coord_map[from_c])
                x2, y2 = to_svg(*cams_coord_map[to_c])
                subj_name = tr.get("display_name", tr.get("subject_id", "TARGET"))
                dur = tr.get("transit_duration_sec", 0.0)
                mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                transits_svg.append(f"""
                <g>
                    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#F59E0B" stroke-width="4" stroke-dasharray="8,6" opacity="0.85">
                        <animate attributeName="stroke-dashoffset" values="28;0" dur="1.2s" repeatCount="indefinite" />
                    </line>
                    <circle cx="{mid_x}" cy="{mid_y}" r="6" fill="#F59E0B">
                        <animate attributeName="r" values="4;7;4" dur="1.5s" repeatCount="indefinite" />
                    </circle>
                    <rect x="{mid_x - 65}" y="{mid_y - 20}" width="130" height="18" rx="3" fill="#0F172A" stroke="#F59E0B" stroke-width="1" />
                    <text x="{mid_x}" y="{mid_y - 7}" font-family="monospace" font-size="10" fill="#FDE047" font-weight="bold" text-anchor="middle">
                        {subj_name[:12]} ({dur:.0f}s)
                    </text>
                </g>
                """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Offline Surveillance Map — Local Prototype</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background-color: #0F172A;
            color: #F8FAFC;
            font-family: system-ui, -apple-system, sans-serif;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .badge-offline {{
            background-color: #059669;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .map-container {{
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
        }}
        svg {{
            width: 100%;
            max-width: 1000px;
            height: auto;
            background-color: #090D16;
        }}
        .legend {{
            margin-top: 16px;
            display: flex;
            gap: 20px;
            font-size: 13px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h2 style="margin: 0 0 4px 0;">Local Security & Surveillance Grid</h2>
            <div style="font-size: 13px; color: #94A3B8;">Zero-Internet Offline Vector Map Engine | Area: HITEC Demonstration Perimeter</div>
        </div>
        <div class="badge-offline">NETWORK STATUS: 100% OFFLINE / LOCAL</div>
    </div>

    <div class="map-container">
        <svg viewBox="0 0 1000 800" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid Background -->
            <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#1E293B" stroke-width="1"/>
                </pattern>
            </defs>
            <rect width="1000" height="800" fill="url(#grid)" />

            <!-- Security Zones -->
            {"".join(zones_svg)}

            <!-- Road Corridors -->
            {"".join(roads_svg)}

            <!-- Camera Points -->
            {"".join(cams_svg)}

            <!-- Live Event Alerts -->
            {"".join(evts_svg)}

            <!-- Cross-Camera Re-ID Transit Trails -->
            {"".join(transits_svg)}
        </svg>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#3B82F6;"></div> Pedestrian Concourse</div>
        <div class="legend-item"><div class="legend-color" style="background:#F59E0B;"></div> Vehicle Checkpoint</div>
        <div class="legend-item"><div class="legend-color" style="background:#EF4444;"></div> Critical Command Perimeter</div>
        <div class="legend-item"><div class="legend-color" style="background:#38BDF8; border-radius: 50%;"></div> Cameras (CAM_01..03)</div>
        <div class="legend-item"><div class="legend-color" style="background:#F59E0B; border: 1px dashed #FFFFFF;"></div> Re-ID Transit Trail</div>
        <div class="legend-item"><div class="legend-color" style="background:#EF4444; border-radius: 50%;"></div> Critical Alert</div>
    </div>
</body>
</html>
"""
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_file


def run_mapping_verification():
    mgr = OfflineMapManager()
    payload = mgr.build_integrated_map_payload()
    preview_path = mgr.generate_offline_html_preview(payload)

    return {
        "zones_count": len(payload["zones"]["features"]),
        "roads_count": len(payload["roads"]["features"]),
        "cameras_count": len(payload["cameras"]),
        "event_markers_count": len(payload["event_markers"]),
        "map_overlay_json": str(mgr.processed_dir / "map_overlay.json"),
        "preview_html": str(preview_path),
    }


if __name__ == "__main__":
    try:
        res = run_mapping_verification()
        print("\n--- Offline Mapping Verification Summary ---")
        print(f"Zones Loaded: {res['zones_count']}")
        print(f"Road Segments: {res['roads_count']}")
        print(f"Cameras Registered: {res['cameras_count']}")
        print(f"Live Event Markers Mapped: {res['event_markers_count']}")
        print(f"Map Overlay JSON: {res['map_overlay_json']}")
        print(f"Self-Contained Offline Preview: {res['preview_html']}")
        print("Status: OFFLINE MAPPING VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
