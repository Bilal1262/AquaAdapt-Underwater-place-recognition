"""Evidence-only Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aquaadapt.data.manifest import resolve_camera_topic
from aquaadapt.paths import trajectory_root
from aquaadapt.trajectory.tum import parse_tum


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def generate_report(cfg: dict[str, Any]) -> Path:
    output = Path("results") / str(cfg["project"]["run_name"])
    output.mkdir(parents=True, exist_ok=True)
    topic = resolve_camera_topic(cfg)
    data_root = trajectory_root(cfg, topic)
    extraction = _read_json(data_root / "extraction_summary.json")
    trajectory = parse_tum(cfg["paths"]["tum"]).summary()
    method_metrics: list[dict[str, Any]] = []
    descriptor_root = Path(cfg["paths"]["processed_root"]) / "descriptors" / str(cfg["project"]["run_name"])
    for method in ("raw_dinov2", "enhanced_dinov2", "aquaadapt"):
        metrics = _read_json(descriptor_root / method / "evaluation.json")
        if metrics:
            method_metrics.append({"method": method, **metrics})
    main = pd.DataFrame(method_metrics)
    main.to_csv(output / "main_results.csv", index=False)
    robustness_source = output / "robustness_results.csv"
    if not robustness_source.is_file():
        pd.DataFrame(columns=["method", "corruption", "severity", "metric", "value", "eligible_query_count"]).to_csv(robustness_source, index=False)
    efficiency_source = output / "efficiency_results.csv"
    if not efficiency_source.is_file():
        pd.DataFrame().to_csv(efficiency_source, index=False)
    summary = {
        "project": cfg["project"], "selected_camera_topic": topic,
        "extraction": extraction, "trajectory": trajectory,
        "methods_evaluated": [row["method"] for row in method_metrics],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    result_table = main.to_markdown(index=False) if not main.empty else "No completed descriptor evaluations are available; no metrics were invented."
    regime = str(cfg["project"].get("evaluation_regime", "single_trajectory_development"))
    if regime == "held_out_environment_transfer":
        limitation_statement = (
            f"This is a held-out environment transfer evaluation: the checkpoint was trained only "
            f"on `{cfg['project'].get('training_trajectory', 'other trajectories')}` and evaluated "
            f"without fitting or checkpoint selection on "
            f"`{cfg['project'].get('evaluation_trajectory', 'this environment')}`. "
            "It is a stronger domain-shift test than a held-out trajectory from the same environment, "
            "but one held-out environment alone does not establish universal generalization."
        )
    elif regime == "held_out_trajectory_transfer":
        limitation_statement = (
            f"This is a held-out trajectory transfer evaluation: the checkpoint was trained on "
            f"`{cfg['project'].get('training_trajectory', 'another trajectory')}` and evaluated "
            f"without fitting on `{cfg['project'].get('evaluation_trajectory', 'this trajectory')}`. "
            "It still represents one training/evaluation trajectory pair and does not establish broad "
            "cross-environment generalization."
        )
    else:
        limitation_statement = (
            "These results are a single-trajectory development evaluation and do not establish "
            "cross-trajectory or cross-environment generalization."
        )
    if str(cfg["model"].get("adapter_type", "mlp")) == "residual":
        model_statement = (
            "The raw baseline uses normalized 384-dimensional DINOv2 CLS features. "
            "AquaAdapt V2 applies a zero-initialized 384→512→384 residual adapter, "
            f"scaled by {float(cfg['model'].get('adapter_scale', 0.1)):.2f}, followed by "
            "L2 normalization. Training combines multi-positive InfoNCE, DINO similarity-"
            "geometry preservation, and clean/corrupt consistency. The backbone is frozen."
        )
    else:
        model_statement = (
            "The raw baseline uses normalized 384-dimensional DINOv2 CLS features. "
            f"AquaAdapt applies a 384→512→{int(cfg['model']['descriptor_dim'])} projection "
            "with GELU, dropout, and L2 normalization, trained with multi-positive InfoNCE "
            "over underwater-degraded views. The backbone is frozen by default."
        )
    report = f"""# AquaAdapt technical report

## 1. Objective

AquaAdapt studies self-supervised adaptation of DINOv2 ViT-S/14 descriptors for robust underwater place recognition.

## 2. Dataset and extraction

- ROS bag: `{cfg['paths']['bag']}`
- TUM trajectory: `{cfg['paths']['tum']}`
- Selected camera topic: `{topic}`
- Extracted frames: {extraction.get('frame_count', 'NA')}
- TUM poses: {trajectory.get('pose_count', 'NA')}
- Pose association: {extraction.get('association', {}).get('matched_percent', 'NA')}%

## 3. Model and training

{model_statement}

## 4. Evaluation protocol and coverage

Evaluation uses translation poses, a {cfg['evaluation']['pose_positive_radius_m']} m positive
radius, and a {cfg['evaluation']['temporal_exclusion_sec']} s exclusion window. Queries without
any valid positive after exclusion are reported as ineligible and excluded from retrieval metrics.

{result_table}

## 5. Robustness, ablations, and efficiency

Machine-readable results are in `robustness_results.csv`, `ablation_results.csv` when assembled,
and `efficiency_results.csv`. Missing experiments remain absent or `NA`.

## 6. Qualitative examples and failures

Retrieval panels are generated when enough eligible loop-closure queries exist. Low eligibility
is itself a result and indicates that another repeated trajectory is needed.

## 7. Limitations

{limitation_statement}
The corruptions are controlled robustness probes, not claims of complete underwater image formation.
The classical enhancement is a comparison baseline, not a state-of-the-art enhancement model.

## 8. Reproduction

```bash
bash scripts/run_experiment_matrix.sh --recommended
bash scripts/run_quick_pipeline.sh
```
"""
    destination = output / "report.md"
    destination.write_text(report, encoding="utf-8")
    return destination
