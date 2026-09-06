"""
Experiment Manager & Model Lineage Genealogy Tracker.
Maintains reproducible experiment dossiers (EXP_XXX.json), tracks parent-to-candidate
model evolution, dataset hash bindings, multi-scenario evaluation results, human commander
approvals, and exports tactical audit markdown reports.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EXPERIMENTS_DIR = ROOT_DIR / "training_lab" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = EXPERIMENTS_DIR / "experiment_registry.json"


class ExperimentManager:
    """
    Manages experiment dossiers, model versioning genealogy, and approval workflows.
    """

    VALID_APPROVAL_STATUSES = {"PENDING_REVIEW", "APPROVED", "REJECTED"}
    VALID_VERDICTS = {"IMPROVED", "REGRESSED", "NO_CHANGE", "INCONCLUSIVE"}

    def __init__(
        self,
        experiments_dir: Optional[Union[str, Path]] = None,
        registry_file: Optional[Union[str, Path]] = None,
    ):
        self.experiments_dir = Path(experiments_dir) if experiments_dir else EXPERIMENTS_DIR
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = Path(registry_file) if registry_file else (self.experiments_dir / "experiment_registry.json")
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def get_next_experiment_id(self) -> str:
        """Computes next auto-incremented experiment ID (EXP_001, EXP_002...)."""
        existing = []
        for p in self.experiments_dir.glob("EXP_*.json"):
            m = re.match(r"EXP_(\d+)\.json", p.name)
            if m:
                existing.append(int(m.group(1)))

        if not existing:
            # Also check registry
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                for item in reg:
                    eid = item.get("experiment_id", "")
                    m = re.match(r"EXP_(\d+)", eid)
                    if m:
                        existing.append(int(m.group(1)))
            except Exception:
                pass

        next_idx = max(existing) + 1 if existing else 1
        return f"EXP_{next_idx:03d}"

    def record_experiment(
        self,
        name: str,
        parent_model_version: str,
        candidate_model_version: str,
        dataset_version: str,
        dataset_hash: str,
        scenarios_tested: List[str],
        perimeter_rules_summary: Optional[Dict[str, Any]] = None,
        threat_rules_summary: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        comparison_delta: Optional[Dict[str, Any]] = None,
        verdict: str = "INCONCLUSIVE",
        human_approval_status: str = "PENDING_REVIEW",
        approval_metadata: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates and persists a full experiment dossier to disk and registers it.
        """
        exp_id = experiment_id or self.get_next_experiment_id()
        verdict_clean = verdict.upper() if verdict.upper() in self.VALID_VERDICTS else "INCONCLUSIVE"
        status_clean = (
            human_approval_status.upper()
            if human_approval_status.upper() in self.VALID_APPROVAL_STATUSES
            else "PENDING_REVIEW"
        )

        dossier: Dict[str, Any] = {
            "experiment_id": exp_id,
            "name": name,
            "parent_model_version": parent_model_version,
            "candidate_model_version": candidate_model_version,
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "scenarios_tested": scenarios_tested,
            "perimeter_rules_summary": perimeter_rules_summary or {},
            "threat_rules_summary": threat_rules_summary or {},
            "evaluation_metrics": evaluation_metrics or {},
            "comparison_delta": comparison_delta or {},
            "verdict": verdict_clean,
            "human_approval_status": status_clean,
            "approval_metadata": approval_metadata,
            "lineage": {
                "parent": parent_model_version,
                "dataset": dataset_version,
                "child": candidate_model_version,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Write individual EXP_XXX.json file
        exp_file = self.experiments_dir / f"{exp_id}.json"
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

        # 2. Update registry
        records: List[Dict[str, Any]] = []
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        # Replace existing or append
        updated = False
        for i, r in enumerate(records):
            if r.get("experiment_id") == exp_id:
                records[i] = dossier
                updated = True
                break
        if not updated:
            records.append(dossier)

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        return dossier

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an experiment dossier by ID."""
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Fallback to registry
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    for r in json.load(f):
                        if r.get("experiment_id") == experiment_id:
                            return r
            except Exception:
                pass
        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lists all experiment dossiers sorted chronologically."""
        experiments = []
        for exp_file in sorted(self.experiments_dir.glob("EXP_*.json")):
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    experiments.append(json.load(f))
            except Exception:
                pass

        if not experiments and self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    experiments = json.load(f)
            except Exception:
                pass

        return experiments

    def update_approval(
        self,
        experiment_id: str,
        status: str,
        operator_id: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Updates human commander approval status (APPROVED/REJECTED) atomically.
        """
        st_clean = status.upper()
        if st_clean not in self.VALID_APPROVAL_STATUSES:
            raise ValueError(f"Invalid approval status '{status}'. Must be one of {self.VALID_APPROVAL_STATUSES}")

        dossier = self.get_experiment(experiment_id)
        if not dossier:
            raise FileNotFoundError(f"Experiment dossier {experiment_id} not found.")

        dossier["human_approval_status"] = st_clean
        dossier["approval_metadata"] = {
            "operator_id": operator_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }

        # Write back to individual file
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

        # Update in registry
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for i, r in enumerate(records):
                    if r.get("experiment_id") == experiment_id:
                        records[i] = dossier
                        break
                with open(self.registry_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
            except Exception:
                pass

        return dossier

    def get_genealogy_tree(self) -> Dict[str, Any]:
        """
        Traverses all recorded experiments to construct a model lineage DAG:
          - nodes: list of unique model version strings
          - edges: list of transition links with parent, child, dataset, verdict, and approval
          - deployed_model: latest approved model version
        """
        experiments = self.list_experiments()

        nodes: Set[str] = set()
        edges: List[Dict[str, Any]] = []
        deployed_model = "yolov8l.pt"

        # Track deployed model candidates
        for exp in experiments:
            parent = exp.get("parent_model_version")
            child = exp.get("candidate_model_version")
            dataset = exp.get("dataset_version")
            exp_id = exp.get("experiment_id")
            verdict = exp.get("verdict", "INCONCLUSIVE")
            status = exp.get("human_approval_status", "PENDING_REVIEW")

            if parent:
                nodes.add(parent)
            if child:
                nodes.add(child)

            edges.append({
                "from": parent,
                "to": child,
                "dataset": dataset,
                "experiment_id": exp_id,
                "verdict": verdict,
                "approved": status == "APPROVED",
                "approval_status": status,
            })

            # The latest approved model with improvement becomes deployed
            if status == "APPROVED":
                deployed_model = child

        # If no approved model yet, fallback to latest candidate model or yolov8l.pt
        if deployed_model == "yolov8l.pt" and experiments:
            deployed_model = experiments[-1].get("candidate_model_version", "yolov8l.pt")

        return {
            "nodes": sorted(list(nodes)),
            "edges": edges,
            "deployed_model": deployed_model,
            "total_experiments": len(experiments),
        }

    def export_dossier_markdown(
        self,
        experiment_id: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Produces a standalone tactical markdown audit report for the experiment.
        """
        exp = self.get_experiment(experiment_id)
        if not exp:
            raise FileNotFoundError(f"Experiment {experiment_id} not found.")

        metrics = exp.get("evaluation_metrics", {})
        deltas = exp.get("comparison_delta", {}).get("overall_deltas", {})
        approval = exp.get("approval_metadata") or {}

        md = f"""# BORDER SENTINEL — EXPERIMENT DOSSIER: {exp.get('experiment_id')}
**Experiment Name**: {exp.get('name')}  
**Recorded Timestamp**: {exp.get('created_at')}  
**Approval Status**: **{exp.get('human_approval_status')}**  
**Verdict**: **{exp.get('verdict')}**

---

## 1. Lineage & Training Metadata
| Field | Value |
|---|---|
| **Parent Model** | `{exp.get('parent_model_version')}` |
| **Candidate Model** | `{exp.get('candidate_model_version')}` |
| **Dataset Version** | `{exp.get('dataset_version')}` |
| **Dataset Hash (SHA256)** | `{exp.get('dataset_hash')}` |
| **Scenarios Tested** | {', '.join(exp.get('scenarios_tested', []))} |

---

## 2. Evaluation Metrics (Candidate)
| Metric | Score |
|---|---|
| **Precision** | {metrics.get('precision', 'N/A')}% |
| **Recall** | {metrics.get('recall', 'N/A')}% |
| **F1 Score** | {metrics.get('f1', 'N/A')}% |
| **mAP50** | {metrics.get('mAP50', 'N/A')}% |
| **True Positives (TP)** | {metrics.get('tp', 'N/A')} |
| **False Positives (FP)** | {metrics.get('fp', 'N/A')} |
| **False Negatives (FN)** | {metrics.get('fn', 'N/A')} |

---

## 3. Comparison Deltas (Candidate vs Parent)
| Metric Delta | Change |
|---|---|
| **Δ mAP50** | {deltas.get('delta_mAP50', 'N/A')}% |
| **Δ Precision** | {deltas.get('delta_precision', 'N/A')}% |
| **Δ Recall** | {deltas.get('delta_recall', 'N/A')}% |
| **Δ False Positives (FP)** | {deltas.get('delta_fp', 'N/A')} |
| **Δ False Negatives (FN)** | {deltas.get('delta_fn', 'N/A')} |

---

## 4. Human Commander Approval
- **Status**: {exp.get('human_approval_status')}
- **Approved/Reviewed By**: {approval.get('operator_id', 'None')}
- **Review Timestamp**: {approval.get('timestamp', 'None')}
- **Operator Notes**: {approval.get('notes', 'No notes provided.')}

---
*Report generated by BORDER SENTINEL AI Training Lab — Air-Gapped Operation.*
"""
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(md)

        return md


experiment_manager = ExperimentManager()
