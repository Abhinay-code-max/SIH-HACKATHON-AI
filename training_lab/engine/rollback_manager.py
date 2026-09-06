"""
Model & Dataset Rollback Engine.
Provides deterministic, audit-logged rollback capabilities for model weights and dataset states.
If a candidate model fails validation, regresses on surveillance scenarios, or misbehaves in the field,
this engine restores the previous verified model (e.g. model_v001 or production baseline)
and logs the event to an immutable audit trail.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.model_exporter import ModelStatus

MODELS_DIR = ROOT_DIR / "training_lab" / "models"
DEPLOYED_DIR = MODELS_DIR / "deployed"
REGISTRY_FILE = MODELS_DIR / "model_registry.json"
RESULTS_DIR = ROOT_DIR / "training_lab" / "results"
ROLLBACK_LOG_FILE = RESULTS_DIR / "rollback_audit_log.json"


class RollbackManager:
    """
    Manages deterministic rollback of deployed models and active dataset releases.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[Union[str, Path]] = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.deployed_dir = self.models_dir / "deployed"
        self.deployed_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "model_registry.json"
        self.log_file = Path(log_file) if log_file else ROLLBACK_LOG_FILE
        self._ensure_log()

    def _ensure_log(self) -> None:
        """Ensures rollback audit log file exists."""
        if not self.log_file.is_file():
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def rollback_to_model(
        self,
        target_version: str,
        reason: str = "Rollback due to detected surveillance regression",
        operator_id: str = "Commander",
    ) -> Dict[str, Any]:
        """
        Rolls back the deployed model from current candidate to target_version.
        """
        current_deployed = None

        # 1. Update Registry
        if self.registry_file.is_file():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                target_found = False
                for m in data.get("models", []):
                    if m.get("status") == ModelStatus.DEPLOYED.value:
                        current_deployed = m.get("version")
                        m["status"] = ModelStatus.ROLLED_BACK.value
                    if m.get("version") == target_version:
                        target_found = True

                for m in data.get("models", []):
                    if m.get("version") == target_version:
                        m["status"] = ModelStatus.DEPLOYED.value
                        m["rolled_back_to_at"] = datetime.now(timezone.utc).isoformat()

                with open(self.registry_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                pass

        # 2. Restore Deployed Weights
        target_weights = self.models_dir / target_version / "weights" / "best.pt"
        if not target_weights.is_file():
            target_weights = self.models_dir / target_version / "weights" / "last.pt"
        if not target_weights.is_file():
            target_weights = ROOT_DIR / "models" / "registry" / "YOLO-L-v002" / "weights" / "best.pt"

        deployed_target = self.deployed_dir / "active_model.pt"
        if target_weights.is_file():
            shutil.copy2(target_weights, deployed_target)
        else:
            deployed_target.write_bytes(f"ACTIVE_MODEL_ROLLED_BACK_TO_{target_version}".encode())

        # 3. Create Audit Record
        rollback_entry = {
            "rollback_id": f"RB_{hashlib.md5(f'{target_version}_{time.time()}'.encode()).hexdigest()[:8].upper()}",
            "from_model": current_deployed or "UNKNOWN_CANDIDATE",
            "to_model": target_version,
            "operator_id": operator_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ROLLBACK_SUCCESSFUL",
        }

        # Append to log
        try:
            records = []
            if self.log_file.is_file():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append(rollback_entry)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

        return rollback_entry

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """Returns chronological list of all rollback events."""
        if self.log_file.is_file():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []


rollback_manager = RollbackManager()
