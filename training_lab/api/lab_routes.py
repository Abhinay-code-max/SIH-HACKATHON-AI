"""
BORDER SENTINEL — AI Training, Validation & Simulation Lab REST API Router.
Provides endpoints under /api/lab for scenario telemetry, human validation,
missed detection capture, operator override, timeline audits, experiment dossiers,
and model lineage tracking.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.annotation_capture import annotation_capture
from training_lab.engine.experiment_manager import experiment_manager
from training_lab.engine.operator_override import event_timeline, operator_override_engine
from training_lab.engine.perimeter_editor import perimeter_engine
from training_lab.engine.scenario_manager import scenario_manager
from training_lab.engine.threat_validator import threat_validator
from training_lab.engine.training_manager import training_manager
from training_lab.engine.validation_engine import ErrorCategory, ValidationDecision, validation_engine

lab_router = APIRouter(prefix="/api/lab", tags=["AI Training Lab"])


# ---------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------
class ValidationSubmitRequest(BaseModel):
    scenario_id: str
    camera_id: str
    frame_idx: int
    decision: str  # "CORRECT", "INCORRECT", "CHANGE_CLASS", "IGNORE"
    human_verified_class: Optional[str] = None
    human_bbox: Optional[List[float]] = None
    operator_id: str = "Operator"
    error_category: Optional[str] = None
    notes: Optional[str] = ""


class AnnotationCaptureRequest(BaseModel):
    scenario_id: str
    camera_id: str
    frame_idx: int
    human_verified_class: str
    bbox: List[float]
    operator_id: str = "Operator"
    notes: Optional[str] = ""


class OverrideSubmitRequest(BaseModel):
    automatic_level: str
    override_level: str
    reason: str
    operator: str = "Commander"


class ExperimentApproveRequest(BaseModel):
    status: str  # "APPROVED" or "REJECTED"
    operator_id: str = "Commander"
    notes: Optional[str] = ""


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@lab_router.get("/status")
def get_lab_status() -> Dict[str, Any]:
    """
    Returns system status, CUDA hardware state, model registry counts,
    and currently deployed model version.
    """
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Fallback)"

    scenarios = scenario_manager.list_scenarios()
    val_stats = validation_engine.get_validation_statistics()
    genealogy = experiment_manager.get_genealogy_tree()
    models = training_manager.list_models()

    return {
        "status": "OPERATIONAL",
        "air_gapped": True,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "scenarios_count": len(scenarios),
        "models_count": len(models),
        "validations_count": val_stats.get("total_validations", 0),
        "experiments_count": genealogy.get("total_experiments", 0),
        "deployed_model": genealogy.get("deployed_model", "model_v001"),
    }


@lab_router.get("/scenarios")
def list_scenarios() -> List[Dict[str, Any]]:
    """Returns all 15 master benchmark surveillance scenarios."""
    scenarios = scenario_manager.list_scenarios()
    return [
        {
            "scenario_id": s.scenario_id,
            "name": s.name,
            "description": s.description,
            "category": s.category.value if hasattr(s.category, "value") else str(s.category),
            "threat_category": s.category.value if hasattr(s.category, "value") else str(s.category),
            "environment": s.environment.value if hasattr(s.environment, "value") else str(s.environment),
            "lighting": s.lighting.value if hasattr(s.lighting, "value") else str(s.lighting),
            "difficulty": s.difficulty.value if hasattr(s.difficulty, "value") else str(s.difficulty),
            "assigned_cameras": list(s.assigned_cameras.keys()),
        }
        for s in scenarios
    ]


@lab_router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> Dict[str, Any]:
    """Returns metadata for a specific scenario."""
    s = scenario_manager.get_scenario(scenario_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")
    return {
        "scenario_id": s.scenario_id,
        "name": s.name,
        "description": s.description,
        "category": s.category.value if hasattr(s.category, "value") else str(s.category),
        "threat_category": s.category.value if hasattr(s.category, "value") else str(s.category),
        "environment": s.environment.value if hasattr(s.environment, "value") else str(s.environment),
        "lighting": s.lighting.value if hasattr(s.lighting, "value") else str(s.lighting),
        "difficulty": s.difficulty.value if hasattr(s.difficulty, "value") else str(s.difficulty),
        "assigned_cameras": list(s.assigned_cameras.keys()),
    }


@lab_router.post("/validations/submit")
def submit_validation(req: ValidationSubmitRequest) -> Dict[str, Any]:
    """Records human validation feedback on a detection."""
    # Convert string decision to ValidationDecision enum
    try:
        dec = ValidationDecision(req.decision.upper())
    except ValueError:
        dec = ValidationDecision.CORRECT

    err_cat = None
    if req.error_category:
        try:
            err_cat = ErrorCategory(req.error_category.upper())
        except ValueError:
            err_cat = None

    rec = validation_engine.record_feedback(
        scenario_id=req.scenario_id,
        camera_id=req.camera_id,
        frame_idx=req.frame_idx,
        decision=dec,
        ai_prediction={"class_name": "target", "confidence": 0.85},
        human_verified_class=req.human_verified_class,
        human_bbox=req.human_bbox,
        error_category=err_cat,
        notes=req.notes or "",
    )
    return {"status": "SUCCESS", "validation_id": rec.record_id, "record": rec.model_dump()}


@lab_router.get("/validations/stats")
def get_validation_stats() -> Dict[str, Any]:
    """Returns error taxonomy and validation accuracy statistics."""
    return validation_engine.get_validation_statistics()


@lab_router.post("/annotations/capture")
def capture_missed_detection(req: AnnotationCaptureRequest) -> Dict[str, Any]:
    """Captures a manual human bounding box for a missed target."""
    rec = annotation_capture.capture_missed_detection(
        scenario_id=req.scenario_id,
        camera_id=req.camera_id,
        frame_idx=req.frame_idx,
        class_name=req.human_verified_class,
        bbox=req.bbox,
        notes=req.notes or "",
    )
    return {"status": "CAPTURED", "annotation_id": rec.record_id, "record": rec.model_dump()}


@lab_router.post("/override/submit")
def submit_override(req: OverrideSubmitRequest) -> Dict[str, Any]:
    """Applies a manual threat override by a human operator."""
    res = operator_override_engine.set_override(
        automatic_level=req.automatic_level,
        override_level=req.override_level,
        reason=req.reason,
        operator=req.operator,
    )
    event_timeline.log_event(
        event_type="OPERATOR_OVERRIDE",
        camera_id="ALL",
        description=f"Operator {req.operator} changed threat level to {req.override_level}: {req.reason}",
        metadata=res,
    )
    return {"status": "OVERRIDE_APPLIED", "override": res}


@lab_router.get("/override/effective")
def get_effective_override(current_ai_level: str = "LOW") -> Dict[str, Any]:
    """Returns the effective threat level distinguishing AI vs Human override."""
    return operator_override_engine.get_effective_threat(current_ai_level=current_ai_level)


@lab_router.post("/override/clear")
def clear_override() -> Dict[str, Any]:
    """Clears the active manual override restoring AI decision mode."""
    res = operator_override_engine.clear_override()
    event_timeline.log_event(
        event_type="OPERATOR_OVERRIDE",
        camera_id="ALL",
        description="Operator override cleared; restoring automated AI threat arbitration.",
    )
    return res


@lab_router.get("/timeline")
def get_timeline(limit: int = Query(default=100, ge=1, le=500)) -> List[Dict[str, Any]]:
    """Returns recent chronological security events from the unified timeline."""
    return event_timeline.get_timeline(limit=limit)


@lab_router.get("/models")
def list_models() -> List[Dict[str, Any]]:
    """Returns registered surveillance model checkpoints."""
    return training_manager.list_models()


@lab_router.get("/experiments")
def list_experiments() -> Dict[str, Any]:
    """Returns all recorded experiment dossiers and the model genealogy tree."""
    dossiers = experiment_manager.list_experiments()
    genealogy = experiment_manager.get_genealogy_tree()
    return {
        "dossiers": dossiers,
        "genealogy": genealogy,
    }


@lab_router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """Returns an experiment dossier by ID."""
    dossier = experiment_manager.get_experiment(experiment_id)
    if not dossier:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return dossier


@lab_router.post("/experiments/{experiment_id}/approve")
def approve_experiment(experiment_id: str, req: ExperimentApproveRequest) -> Dict[str, Any]:
    """Records human commander approval or rejection for a model experiment."""
    try:
        updated = experiment_manager.update_approval(
            experiment_id=experiment_id,
            status=req.status,
            operator_id=req.operator_id,
            notes=req.notes or "",
        )
        event_timeline.log_event(
            event_type="HUMAN_VALIDATION",
            camera_id="LAB",
            description=f"Commander {req.operator_id} marked {experiment_id} as {req.status}",
            metadata={"experiment_id": experiment_id, "status": req.status},
        )
        return {"status": "SUCCESS", "dossier": updated}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@lab_router.get("/experiments/{experiment_id}/dossier/markdown")
def get_experiment_markdown(experiment_id: str) -> Dict[str, str]:
    """Returns a standalone tactical markdown audit report for the experiment."""
    try:
        md = experiment_manager.export_dossier_markdown(experiment_id)
        return {"experiment_id": experiment_id, "markdown": md}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
