"""Qualitative retrieval visualization on real manifest images."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from aquaadapt.augmentations.pipeline import apply_corruption
from aquaadapt.data.transforms import dinov2_transform
from aquaadapt.models.aquaadapt import AquaAdaptModel
from aquaadapt.models.dinov2 import DINOv2Backbone
from aquaadapt.retrieval.descriptors import descriptor_directory
from aquaadapt.retrieval.index import normalize_descriptors
from aquaadapt.training.checkpointing import load_checkpoint


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read retrieval visualization image: {path}")
    return image


def _show_image(axis: Any, image: np.ndarray, title: str, correct: bool | None = None) -> None:
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    if correct is not None:
        color = "tab:green" if correct else "tab:red"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_edgecolor(color)


def _query_database_indices(metadata: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    valid = metadata["pose_valid"].astype(bool).to_numpy()
    test = metadata["split"].astype(str).to_numpy() == "test"
    pool = np.flatnonzero(valid & test)
    query_stride = max(1, int(cfg["evaluation"]["query_stride"]))
    database_stride = max(1, int(cfg["evaluation"]["database_stride"]))
    return pool[1::query_stride], pool[::database_stride]


def _candidates_for_query(
    query: int, database: np.ndarray, metadata: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = metadata["timestamp_sec"].to_numpy(float)
    xyz = metadata[["tx", "ty", "tz"]].to_numpy(float)
    keep = (
        np.abs(timestamps[database] - timestamps[query])
        > float(cfg["evaluation"]["temporal_exclusion_sec"])
    ) & (database != query)
    candidates = database[keep]
    distances = np.linalg.norm(xyz[candidates] - xyz[query], axis=1)
    return candidates, distances


def _top_matches(
    query_descriptor: np.ndarray,
    database_descriptors: np.ndarray,
    candidates: np.ndarray,
    candidate_distances: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = query_descriptor @ database_descriptors[candidates].T
    order = np.argsort(-scores, kind="stable")[:k]
    return candidates[order], scores[order], candidate_distances[order]


def _retrieval_panel(
    query: int,
    query_image: np.ndarray,
    metadata: pd.DataFrame,
    cfg: dict[str, Any],
    database: np.ndarray,
    raw_query: np.ndarray,
    adapted_query: np.ndarray,
    raw_database: np.ndarray,
    adapted_database: np.ndarray,
    destination: Path,
    label: str,
) -> dict[str, Any]:
    candidates, distances = _candidates_for_query(query, database, metadata, cfg)
    if not len(candidates):
        return {"query": query, "panel": None, "reason": "no database candidates after temporal exclusion"}
    raw_indices, raw_scores, raw_distances = _top_matches(
        raw_query, raw_database, candidates, distances
    )
    adapted_indices, adapted_scores, adapted_distances = _top_matches(
        adapted_query, adapted_database, candidates, distances
    )
    radius = float(cfg["evaluation"]["pose_positive_radius_m"])
    timestamps = metadata["timestamp_sec"].to_numpy(float)
    has_positive = bool(np.any(distances <= radius))
    figure, axes = plt.subplots(2, 6, figsize=(18, 6))
    for row, method in enumerate(("Raw DINOv2", "AquaAdapt")):
        _show_image(
            axes[row, 0], query_image,
            f"{method}\nQUERY #{query}\n{label}",
        )
    for row, (indices, scores, pose_distances) in enumerate((
        (raw_indices, raw_scores, raw_distances),
        (adapted_indices, adapted_scores, adapted_distances),
    )):
        for rank, (index, score, pose_distance) in enumerate(
            zip(indices, scores, pose_distances), 1
        ):
            delta_time = abs(float(timestamps[index] - timestamps[query]))
            correct = bool(pose_distance <= radius)
            title = (
                f"top-{rank} | sim {score:.3f}\n"
                f"pose {pose_distance:.2f} m | Δt {delta_time:.1f} s\n"
                f"{'POSE POSITIVE' if correct else 'not a pose positive'}"
            )
            _show_image(
                axes[row, rank],
                _load_image(str(metadata.iloc[index]["image_path"])),
                title,
                correct,
            )
    eligibility = (
        "At least one pose positive exists in the candidate database."
        if has_positive else
        "NO GROUND-TRUTH REVISIT: rankings are qualitative only, not successful place matches."
    )
    figure.suptitle(
        f"{label} retrieval comparison — query manifest row {query}\n{eligibility}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return {
        "query": int(query), "panel": destination.name, "label": label,
        "eligible": has_positive,
        "raw_top1_index": int(raw_indices[0]),
        "raw_top1_similarity": float(raw_scores[0]),
        "raw_top1_pose_distance_m": float(raw_distances[0]),
        "aquaadapt_top1_index": int(adapted_indices[0]),
        "aquaadapt_top1_similarity": float(adapted_scores[0]),
        "aquaadapt_top1_pose_distance_m": float(adapted_distances[0]),
    }


def _encode_corrupted_queries(
    images: list[np.ndarray],
    corruption: str,
    severity: int,
    cfg: dict[str, Any],
    raw_model: DINOv2Backbone,
    adapted_model: AquaAdaptModel,
    device: torch.device,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    degraded = [
        apply_corruption(image, corruption, severity, int(cfg["project"]["seed"]) + index)
        for index, image in enumerate(images)
    ]
    batch = torch.stack([
        dinov2_transform(image, int(cfg["images"]["model_size"])) for image in degraded
    ]).to(device)
    with torch.inference_mode():
        raw = raw_model(batch).cpu().numpy()
        adapted = adapted_model(batch).cpu().numpy()
    return degraded, normalize_descriptors(raw), normalize_descriptors(adapted)


def _similarity_heatmap(
    raw: np.ndarray,
    adapted: np.ndarray,
    metadata: pd.DataFrame,
    queries: np.ndarray,
    database: np.ndarray,
    output: Path,
) -> None:
    raw_similarity = raw[queries] @ raw[database].T
    adapted_similarity = adapted[queries] @ adapted[database].T
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, values, title in zip(
        axes, (raw_similarity, adapted_similarity), ("Raw DINOv2", "AquaAdapt")
    ):
        rendered = axis.imshow(values, aspect="auto", interpolation="nearest")
        axis.set(
            xlabel="database sample index", ylabel="query sample index",
            title=f"{title} query/database cosine similarity",
        )
        figure.colorbar(rendered, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _retrieval_map(
    metadata: pd.DataFrame,
    selected_queries: np.ndarray,
    database: np.ndarray,
    raw: np.ndarray,
    adapted: np.ndarray,
    cfg: dict[str, Any],
    output: Path,
) -> None:
    xyz = metadata[["tx", "ty", "tz"]].to_numpy(float)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for axis, descriptors, title in zip(
        axes, (raw, adapted), ("Raw DINOv2 top-1 links", "AquaAdapt top-1 links")
    ):
        axis.plot(xyz[database, 0], xyz[database, 1], ".", alpha=0.5, label="database")
        axis.scatter(
            xyz[selected_queries, 0], xyz[selected_queries, 1],
            marker="x", s=50, label="queries",
        )
        for query in selected_queries:
            candidates, distances = _candidates_for_query(query, database, metadata, cfg)
            if not len(candidates):
                continue
            scores = descriptors[query] @ descriptors[candidates].T
            match = candidates[int(np.argmax(scores))]
            axis.annotate(
                "", xy=xyz[match, :2], xytext=xyz[query, :2],
                arrowprops={"arrowstyle": "->", "alpha": 0.65},
            )
        axis.set(xlabel="X [m]", ylabel="Y [m]", title=title)
        axis.legend()
    figure.suptitle("Where each selected query retrieves on the test trajectory")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _retrieval_video(
    metadata: pd.DataFrame,
    queries: np.ndarray,
    database: np.ndarray,
    raw: np.ndarray,
    adapted: np.ndarray,
    cfg: dict[str, Any],
    output: Path,
) -> None:
    """Render a compact query/raw-top1/adapted-top1 playback video."""
    tile_width, tile_height, header_height = 480, 360, 74
    frame_size = (tile_width * 3, tile_height + header_height)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create qualitative retrieval video: {output}")
    radius = float(cfg["evaluation"]["pose_positive_radius_m"])
    for query in queries:
        candidates, distances = _candidates_for_query(int(query), database, metadata, cfg)
        if not len(candidates):
            continue
        raw_scores = raw[query] @ raw[candidates].T
        adapted_scores = adapted[query] @ adapted[candidates].T
        raw_position = int(np.argmax(raw_scores))
        adapted_position = int(np.argmax(adapted_scores))
        raw_match, adapted_match = candidates[raw_position], candidates[adapted_position]
        selections = (
            (int(query), "QUERY", None, None),
            (
                int(raw_match), "RAW DINOv2 TOP-1",
                float(raw_scores[raw_position]), float(distances[raw_position]),
            ),
            (
                int(adapted_match), "AQUAADAPT TOP-1",
                float(adapted_scores[adapted_position]), float(distances[adapted_position]),
            ),
        )
        canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        for column, (index, method, score, pose_distance) in enumerate(selections):
            image = cv2.resize(
                _load_image(str(metadata.iloc[index]["image_path"])),
                (tile_width, tile_height), interpolation=cv2.INTER_AREA,
            )
            x0 = column * tile_width
            canvas[header_height:, x0:x0 + tile_width] = image
            if pose_distance is None:
                color = (255, 180, 0)
                detail = f"manifest row {query}"
            else:
                correct = pose_distance <= radius
                color = (30, 190, 30) if correct else (20, 20, 230)
                detail = f"sim {score:.3f} | pose {pose_distance:.2f} m | {'POSITIVE' if correct else 'NOT POSITIVE'}"
            cv2.rectangle(
                canvas, (x0 + 2, header_height + 2),
                (x0 + tile_width - 3, frame_size[1] - 3), color, 5,
            )
            cv2.putText(
                canvas, method, (x0 + 12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, detail, (x0 + 12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA,
            )
        writer.write(canvas)
    writer.release()


def _write_html(output: Path, rows: list[dict[str, Any]], auxiliary: list[str]) -> Path:
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>AquaAdapt qualitative retrievals</title>",
        "<style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:24px}"
        "img{max-width:100%;height:auto;border:1px solid #bbb;margin-bottom:28px}"
        ".warn{background:#fff3cd;padding:14px;border-left:5px solid #d39e00}"
        "code{background:#eee;padding:2px 5px}</style></head><body>",
        "<h1>AquaAdapt qualitative retrieval viewer</h1>",
        "<p class='warn'><strong>Important:</strong> the quick test split has no eligible "
        "pose revisit after temporal exclusion. Red borders mean the retrieved frame is "
        "not within the configured pose radius. These panels show descriptor behavior; "
        "they are not claimed place-recognition successes.</p>",
        "<p>Each panel compares the same query against a clean test database. Titles show "
        "cosine similarity, translation distance, and timestamp separation.</p>",
        "<h2>Overview</h2>",
    ]
    for name in auxiliary:
        sections.append(f"<h3>{html.escape(name.replace('_', ' ').title())}</h3>")
        if name.lower().endswith(".mp4"):
            sections.append(
                f"<video controls preload='metadata' style='max-width:100%'>"
                f"<source src='{html.escape(name)}' type='video/mp4'></video>"
            )
        else:
            sections.append(f"<a href='{html.escape(name)}'><img src='{html.escape(name)}'></a>")
    sections.append("<h2>Retrieval panels</h2>")
    for row in rows:
        if row.get("panel"):
            sections.append(
                f"<h3>Query {row['query']} — {html.escape(str(row['label']))}</h3>"
                f"<a href='{html.escape(str(row['panel']))}'>"
                f"<img src='{html.escape(str(row['panel']))}'></a>"
            )
    sections.append("</body></html>")
    destination = output / "index.html"
    destination.write_text("\n".join(sections), encoding="utf-8")
    return destination


def visualize_retrievals(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    query_count: int = 6,
    severity: int = 2,
    overwrite: bool = False,
) -> Path:
    """Generate clean and corrupted qualitative retrieval panels plus an HTML gallery."""
    raw_dir = descriptor_directory(cfg, "raw_dinov2")
    adapted_dir = descriptor_directory(cfg, "aquaadapt")
    for directory in (raw_dir, adapted_dir):
        if not (directory / "descriptors.npy").is_file():
            raise FileNotFoundError(
                f"Descriptor cache missing in {directory}; run baseline/encode first"
            )
    metadata = pd.read_csv(raw_dir / "descriptor_metadata.csv")
    adapted_metadata = pd.read_csv(adapted_dir / "descriptor_metadata.csv")
    if not metadata["bag_timestamp_ns"].equals(adapted_metadata["bag_timestamp_ns"]):
        raise ValueError("Raw and AquaAdapt descriptor metadata do not describe identical frames")
    raw = normalize_descriptors(np.load(raw_dir / "descriptors.npy"))
    adapted = normalize_descriptors(np.load(adapted_dir / "descriptors.npy"))
    queries, database = _query_database_indices(metadata, cfg)
    if not len(queries) or not len(database):
        raise ValueError("The test split does not contain enough query/database frames")
    query_count = max(1, min(int(query_count), len(queries)))
    selected_queries = queries[
        np.linspace(0, len(queries) - 1, query_count, dtype=int)
    ]
    output = Path("results") / str(cfg["project"]["run_name"]) / "qualitative"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    query_images = [
        _load_image(str(metadata.iloc[index]["image_path"])) for index in selected_queries
    ]
    for query, image in zip(selected_queries, query_images):
        destination = output / f"query_{query:04d}_clean.png"
        if overwrite or not destination.is_file():
            row = _retrieval_panel(
                int(query), image, metadata, cfg, database,
                raw[query], adapted[query], raw, adapted,
                destination, "clean query",
            )
        else:
            row = {"query": int(query), "panel": destination.name, "label": "clean query"}
        rows.append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = DINOv2Backbone(
        cfg["paths"]["torch_home"], cfg["model"]["pooling"], True, 0,
        int(cfg["model"]["backbone_dim"]),
    ).to(device).eval()
    state = load_checkpoint(checkpoint)
    adapted_model = AquaAdaptModel(cfg, str(state.get("mode", "projection_head_only")))
    adapted_model.load_state_dict(state["model_state"])
    adapted_model.to(device).eval()
    for corruption in cfg["robustness"]["corruptions"]:
        degraded, raw_queries, adapted_queries = _encode_corrupted_queries(
            query_images, str(corruption), int(severity), cfg,
            raw_model, adapted_model, device,
        )
        for position, query in enumerate(selected_queries):
            destination = output / f"query_{query:04d}_{corruption}_s{severity}.png"
            if overwrite or not destination.is_file():
                row = _retrieval_panel(
                    int(query), degraded[position], metadata, cfg, database,
                    raw_queries[position], adapted_queries[position], raw, adapted,
                    destination, f"{corruption}, severity {severity}",
                )
            else:
                row = {
                    "query": int(query), "panel": destination.name,
                    "label": f"{corruption}, severity {severity}",
                }
            rows.append(row)

    heatmap = "descriptor_similarity_heatmap.png"
    retrieval_map = "retrieval_map.png"
    playback = "retrieval_comparison.mp4"
    _similarity_heatmap(raw, adapted, metadata, queries, database, output / heatmap)
    _retrieval_map(
        metadata, selected_queries, database, raw, adapted, cfg, output / retrieval_map
    )
    _retrieval_video(
        metadata, queries, database, raw, adapted, cfg, output / playback
    )
    pd.DataFrame(rows).to_csv(output / "qualitative_retrievals.csv", index=False)
    return _write_html(output, rows, [playback, heatmap, retrieval_map])
