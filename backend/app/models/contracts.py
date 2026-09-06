"""
Core Data Contracts and Pydantic Schemas for Phase 1 Prototype.
These contracts define the stable API between AI Engine, Backend, and Offline Frontend.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(str, Enum):
    OBJECT_DETECTED = "object_detected"
    CROWD_DENSITY = "crowd_density"
    UNAUTHORIZED_VEHICLE = "unauthorized_vehicle"
    ZONE_INTRUSION = "zone_intrusion"
    LOITERING = "loitering"


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class GeoLocation(BaseModel):
    lat: float = Field(..., description="Latitude coordinate", ge=-90.0, le=90.0)
    lon: float = Field(..., description="Longitude coordinate", ge=-180.0, le=180.0)
    zone_name: Optional[str] = Field(default="Demonstration Zone", description="Local region identifier")


class CameraStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


class CameraAsset(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier, e.g. 'CAM_01'")
    name: str = Field(..., description="Descriptive camera name, e.g. 'Gate 1 Entrance'")
    stream_source: str = Field(..., description="Local source: webcam index ('0') or local MP4 path")
    status: CameraStatus = Field(default=CameraStatus.ONLINE)
    location: GeoLocation = Field(..., description="Coordinates on local offline map")
    resolution: str = Field(default="640x480")
    target_fps: int = Field(default=30)


class RawDetection(BaseModel):
    detection_id: str
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] bounding box coordinates")
    normalized_center: Optional[List[float]] = None
    object_id: Optional[Union[int, str]] = Field(default=None, description="Assigned tracker ID or None before tracking")
    camera_id: Optional[str] = Field(default=None, description="Source camera identifier")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of detection")
    confirmed: Optional[bool] = Field(default=None, description="Whether detection has been confirmed across consecutive frames by ConfidenceTracker")


class SecurityEvent(BaseModel):
    event_id: str = Field(..., description="Unique event ID, e.g. 'evt_20260904_001'")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp"
    )
    source_id: str = Field(..., description="Camera or sensor ID, e.g. 'CAM_01'")
    event_type: EventType = Field(default=EventType.OBJECT_DETECTED)
    class_name: str = Field(..., description="Detected object class (e.g. person, car, bus)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates")
    severity: SeverityLevel = Field(default=SeverityLevel.LOW)
    location: GeoLocation = Field(..., description="Associated geographic coordinate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom local metadata")


class SystemHealth(BaseModel):
    status: str = Field(default="healthy", description="'healthy' | 'degraded' | 'error'")
    offline_mode: bool = Field(default=True, description="Strictly True for zero-cloud architecture")
    device: str = Field(..., description="'cuda' or 'cpu'")
    device_name: str = Field(..., description="GPU model name or CPU")
    active_model: str = Field(..., description="Loaded weights file name, e.g. 'yolov8l.pt'")
    vram_allocated_mb: Optional[float] = None
    cameras_active: int = Field(default=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
