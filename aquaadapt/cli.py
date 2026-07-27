"""Command-line interface for the complete AquaAdapt pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquaadapt-matplotlib")

import pandas as pd
import torch

from aquaadapt.bag.extractor import extract_images
from aquaadapt.bag.inspect import inspect_bag
from aquaadapt.config import load_config
from aquaadapt.data.manifest import (
    build_manifest,
    configured_manifest_path,
    resolve_camera_topic,
)
from aquaadapt.evaluation.ablations import assemble_ablations
from aquaadapt.evaluation.efficiency import benchmark_model
from aquaadapt.evaluation.place_recognition import evaluate_directory
from aquaadapt.evaluation.robustness import run_robustness
from aquaadapt.logging_utils import configure_logging
from aquaadapt.models.aquaadapt import AquaAdaptModel
from aquaadapt.models.dinov2 import DINOv2Backbone
from aquaadapt.paths import trajectory_id, trajectory_root
from aquaadapt.reporting import generate_report
from aquaadapt.retrieval.descriptors import descriptor_directory, encode_raw
from aquaadapt.retrieval.encode import encode_adapted
from aquaadapt.training.checkpointing import load_checkpoint
from aquaadapt.training.trainer import train_model
from aquaadapt.trajectory.tum import parse_tum
from aquaadapt.visualization.augmentations import visualize_augmentations
from aquaadapt.visualization.trajectory import plot_trajectory
from aquaadapt.visualization.qualitative import visualize_retrievals

LOG = logging.getLogger(__name__)


def _add_common(parser: argparse.ArgumentParser, lengthy: bool = False) -> None:
    parser.add_argument("--config", default="configs/mclab1.yaml", help="YAML configuration path")
    if lengthy:
        parser.add_argument("--quick", action="store_true", help="Use smoke-test limits")
        parser.add_argument("--limit", type=int, help="Limit processed samples")
        parser.add_argument("--overwrite", action="store_true", help="Recompute existing output")


def _manifest(cfg: dict[str, Any]) -> Path:
    path = configured_manifest_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}; run `aquaadapt build-manifest`")
    return path


def cmd_doctor(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = cfg["paths"]
    os.environ.setdefault("TORCH_HOME", paths["torch_home"])
    checks: dict[str, Any] = {
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 10),
        "dependencies": {},
    }
    modules = {
        "torch": "torch", "torchvision": "torchvision", "rosbags": "rosbags",
        "opencv": "cv2", "Pillow": "PIL", "PyYAML": "yaml", "numpy": "numpy",
        "pandas": "pandas", "scipy": "scipy", "matplotlib": "matplotlib",
        "scikit-learn": "sklearn", "tensorboard": "tensorboard", "pytest": "pytest",
        "faiss (optional)": "faiss",
    }
    checks["dependencies"] = {name: bool(importlib.util.find_spec(module)) for name, module in modules.items()}
    for key in ("bag", "tum", "calibrations"):
        path = Path(paths[key])
        checks[key] = {
            "path": str(path), "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    calibration_root = Path(paths["calibrations"])
    checks["calibration_files"] = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(calibration_root.rglob("*")) if path.is_file()
    ] if calibration_root.is_dir() else []
    processed = Path(paths["processed_root"])
    processed.mkdir(parents=True, exist_ok=True)
    torch_home = Path(paths["torch_home"])
    torch_home.mkdir(parents=True, exist_ok=True)
    checks["processed_root_writable"] = os.access(processed, os.W_OK)
    checks["torch_home"] = str(torch_home)
    checks["cuda_available"] = torch.cuda.is_available()
    checks["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    checks["gpu_memory_bytes"] = (
        int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None
    )
    checks["free_disk_bytes"] = shutil.disk_usage(processed).free
    try:
        with __import__("rosbags.highlevel", fromlist=["AnyReader"]).AnyReader([Path(paths["bag"])]) as reader:
            checks["bag_readable"] = True
            checks["bag_connections"] = len(reader.connections)
    except (OSError, RuntimeError, ValueError) as exc:
        checks["bag_readable"] = False
        checks["bag_error"] = str(exc)
    checks["faiss_available"] = bool(importlib.util.find_spec("faiss"))
    try:
        backbone = DINOv2Backbone(
            paths["torch_home"], cfg["model"]["pooling"], True, 0, int(cfg["model"]["backbone_dim"])
        )
        checks["dinov2_load"] = "ok"
        checks["dinov2_dimension"] = backbone.output_dim
        del backbone
    except RuntimeError as exc:
        checks["dinov2_load"] = "failed"
        checks["dinov2_error"] = str(exc)
    print(json.dumps(checks, indent=2))
    if not checks["cuda_available"]:
        LOG.warning("CUDA is unavailable; extraction/evaluation work on CPU, but training will be slow")


def cmd_inspect(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    bag = args.bag or cfg["paths"]["bag"]
    output = Path(cfg["paths"]["processed_root"]) / trajectory_id(cfg)
    report = inspect_bag(bag, output, args.camera_topic or cfg["extraction"]["camera_topic"])
    print(json.dumps(report, indent=2))


def cmd_extract(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    print(json.dumps(extract_images(cfg, args.quick, args.limit, args.overwrite), indent=2))


def cmd_parse_trajectory(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    trajectory = parse_tum(cfg["paths"]["tum"])
    output = Path(cfg["paths"]["processed_root"]) / trajectory_id(cfg)
    output.mkdir(parents=True, exist_ok=True)
    summary = trajectory.summary()
    (output / "trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_trajectory(trajectory, output / "trajectory.png")
    print(json.dumps(summary, indent=2))


def cmd_manifest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    _, summary = build_manifest(cfg)
    print(json.dumps(summary, indent=2))


def cmd_visualize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    manifest = _manifest(cfg)
    output = trajectory_root(cfg, resolve_camera_topic(cfg)) / "augmentation_contact_sheet.png"
    print(visualize_augmentations(manifest, output, cfg))


def cmd_visualize_retrievals(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    query_count = args.limit if args.limit is not None else args.queries
    print(visualize_retrievals(
        cfg, args.checkpoint, query_count, args.severity, args.overwrite
    ))


def cmd_check_backbone(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2Backbone(
        cfg["paths"]["torch_home"], cfg["model"]["pooling"], True, 0, int(cfg["model"]["backbone_dim"])
    ).to(device)
    sample = torch.randn(1, 3, int(cfg["images"]["model_size"]), int(cfg["images"]["model_size"]), device=device)
    with torch.inference_mode():
        feature = model(sample)
    print(json.dumps({
        "input_shape": list(sample.shape), "feature_shape": list(feature.shape),
        "embedding_dimension": model.output_dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
    }, indent=2))


def cmd_baseline(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    if args.method not in {"raw_dinov2", "enhanced_dinov2"}:
        raise ValueError("Baseline method must be raw_dinov2 or enhanced_dinov2")
    print(encode_raw(cfg, _manifest(cfg), args.method, args.limit, args.overwrite))


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    limit = args.limit
    if args.quick and limit is None:
        limit = 32
    print(train_model(cfg, _manifest(cfg), args.mode, args.resume, limit))


def cmd_encode(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    print(encode_adapted(cfg, _manifest(cfg), args.checkpoint, args.limit, args.overwrite))


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    source = Path(args.descriptor_dir) if args.descriptor_dir else descriptor_directory(cfg, args.method)
    print(json.dumps(evaluate_directory(cfg, source), indent=2))


def cmd_robustness(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    methods = args.methods.split(",") if args.methods else None
    print(run_robustness(cfg, args.checkpoint, methods, args.limit))


def cmd_ablate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    print(assemble_ablations(cfg))


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    raw = DINOv2Backbone(
        cfg["paths"]["torch_home"], cfg["model"]["pooling"], True, 0, int(cfg["model"]["backbone_dim"])
    )
    rows.append({"method": "raw_dinov2", **benchmark_model(raw, 384, device, int(cfg["images"]["model_size"]))})
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint)
        adapted = AquaAdaptModel(cfg, str(state.get("mode", "projection_head_only")))
        adapted.load_state_dict(state["model_state"])
        rows.append({"method": "aquaadapt", **benchmark_model(
            adapted, int(cfg["model"]["descriptor_dim"]), device,
            int(cfg["images"]["model_size"]), args.checkpoint,
        )})
    output = Path("results") / str(cfg["project"]["run_name"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "efficiency_results.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(destination)


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.quick)
    print(generate_report(cfg))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aquaadapt", description="AquaAdapt underwater place-recognition research pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check environment, paths, bag, GPU, and DINOv2")
    _add_common(doctor); doctor.set_defaults(func=cmd_doctor)
    inspect = sub.add_parser("inspect-bag", help="Inventory bag topics and select an RGB stream")
    _add_common(inspect); inspect.add_argument("--bag"); inspect.add_argument("--camera-topic"); inspect.set_defaults(func=cmd_inspect)
    extract = sub.add_parser("extract", help="Extract and pose-associate sampled images")
    _add_common(extract, True); extract.set_defaults(func=cmd_extract)
    trajectory = sub.add_parser("parse-trajectory", help="Parse and plot the TUM trajectory")
    _add_common(trajectory, True); trajectory.set_defaults(func=cmd_parse_trajectory)
    manifest = sub.add_parser("build-manifest", help="Create chronological leakage-resistant splits")
    _add_common(manifest, True); manifest.set_defaults(func=cmd_manifest)
    visualize = sub.add_parser("visualize-augmentations", help="Render deterministic severity examples")
    _add_common(visualize, True); visualize.set_defaults(func=cmd_visualize)
    retrievals = sub.add_parser(
        "visualize-retrievals",
        help="Generate raw-vs-AquaAdapt retrieval panels and an HTML gallery",
    )
    _add_common(retrievals, True)
    retrievals.add_argument("--checkpoint", required=True)
    retrievals.add_argument("--queries", type=int, default=6, help="Number of test queries to show")
    retrievals.add_argument("--severity", type=int, choices=(1, 2, 3), default=2)
    retrievals.set_defaults(func=cmd_visualize_retrievals)
    backbone = sub.add_parser("check-backbone", help="Load DINOv2 and validate its feature shape")
    _add_common(backbone, True); backbone.set_defaults(func=cmd_check_backbone)
    baseline = sub.add_parser("baseline", help="Encode raw or classically enhanced DINOv2 descriptors")
    _add_common(baseline, True); baseline.add_argument("--method", required=True); baseline.set_defaults(func=cmd_baseline)
    train = sub.add_parser("train", help="Train the self-supervised AquaAdapt model")
    _add_common(train, True); train.add_argument("--mode", default="projection_head_only", choices=sorted(AquaAdaptModel.MODES))
    train.add_argument("--resume"); train.set_defaults(func=cmd_train)
    encode = sub.add_parser("encode", help="Encode adapted 256-D descriptors")
    _add_common(encode, True); encode.add_argument("--checkpoint", required=True); encode.add_argument("--method", default="aquaadapt")
    encode.set_defaults(func=cmd_encode)
    evaluate = sub.add_parser("evaluate", help="Run pose-based retrieval evaluation")
    _add_common(evaluate, True); evaluate.add_argument("--method", default="raw_dinov2")
    evaluate.add_argument("--descriptor-dir"); evaluate.set_defaults(func=cmd_evaluate)
    robustness = sub.add_parser("robustness", help="Benchmark clean-database/corrupted-query robustness")
    _add_common(robustness, True); robustness.add_argument("--checkpoint"); robustness.add_argument("--methods")
    robustness.set_defaults(func=cmd_robustness)
    ablate = sub.add_parser("ablate", help="Assemble available ablation results with NA for missing runs")
    _add_common(ablate, True); ablate.set_defaults(func=cmd_ablate)
    benchmark = sub.add_parser("benchmark", help="Measure model inference efficiency")
    _add_common(benchmark, True); benchmark.add_argument("--checkpoint"); benchmark.set_defaults(func=cmd_benchmark)
    report = sub.add_parser("report", help="Generate an evidence-only Markdown report")
    _add_common(report, True); report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, PermissionError) as exc:
        LOG.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
