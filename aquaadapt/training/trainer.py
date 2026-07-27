"""Self-supervised two-view AquaAdapt training."""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from aquaadapt.data.datasets import ContrastiveDataset
from aquaadapt.losses.multipositive_infonce import multi_positive_infonce
from aquaadapt.models.aquaadapt import AquaAdaptModel
from aquaadapt.reproducibility import seed_everything
from aquaadapt.training.checkpointing import load_checkpoint, save_checkpoint
from aquaadapt.training.scheduler import cosine_scheduler
from aquaadapt.visualization.plots import plot_training

LOG = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _positive_mask(
    batch_size: int,
    device: torch.device,
    sample_indices: torch.Tensor | None = None,
    manifest: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
    indices = torch.arange(batch_size, device=device)
    mask[indices, indices + batch_size] = True
    mask[indices + batch_size, indices] = True
    use_temporal = bool(cfg["training"].get("use_temporal_positive", True)) if cfg else False
    use_spatial = bool(cfg["training"].get("use_spatial_positives", False)) if cfg else False
    if sample_indices is not None and manifest is not None and cfg is not None and (
        use_temporal or use_spatial
    ):
        selected = manifest.iloc[sample_indices.cpu().numpy()]
        timestamps = selected["timestamp_sec"].to_numpy(float)
        dt = np.abs(timestamps[:, None] - timestamps[None, :])
        nearby = np.zeros_like(dt, dtype=bool)
        if "trajectory_id" in selected:
            trajectories = selected["trajectory_id"].astype(str).to_numpy()
            same_trajectory = trajectories[:, None] == trajectories[None, :]
        else:
            same_trajectory = np.ones_like(nearby, dtype=bool)
        if use_temporal:
            nearby |= (
                (dt > 0)
                & (dt <= float(cfg["training"]["temporal_positive_window_sec"]))
                & same_trajectory
            )
        if "pose_valid" in selected and (use_temporal or use_spatial):
            valid = selected["pose_valid"].astype(bool).to_numpy()
            xyz = selected[["tx", "ty", "tz"]].to_numpy(float)
            distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            both_valid = valid[:, None] & valid[None, :]
            if use_temporal:
                temporal_candidates = (
                    (dt > 0)
                    & (dt <= float(cfg["training"]["temporal_positive_window_sec"]))
                    & same_trajectory
                )
                nearby &= ~temporal_candidates | (
                    ~both_valid
                    | (
                        distance
                        <= float(
                            cfg["training"]["temporal_positive_max_pose_distance_m"]
                        )
                    )
                )
            if use_spatial:
                nearby |= (
                    same_trajectory
                    & both_valid
                    & (
                        distance
                        <= float(cfg["training"].get("spatial_positive_radius_m", 1.5))
                    )
                    & (
                        dt
                        >= float(
                            cfg["training"].get(
                                "spatial_positive_temporal_exclusion_sec", 10.0
                            )
                        )
                    )
                )
        np.fill_diagonal(nearby, False)
        temporal = torch.as_tensor(nearby, dtype=torch.bool, device=device)
        # Every view of a valid nearby/revisited frame is an additional positive.
        mask[:batch_size, :batch_size] |= temporal
        mask[:batch_size, batch_size:] |= temporal
        mask[batch_size:, :batch_size] |= temporal
        mask[batch_size:, batch_size:] |= temporal
    return mask


def _adaptation_losses(
    model: AquaAdaptModel,
    images: torch.Tensor,
    positive_mask: torch.Tensor,
    batch_size: int,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute contrastive adaptation while preserving clean DINO geometry."""
    features = model.extract_features(images)
    descriptors = model.project_features(features)
    contrastive = multi_positive_infonce(
        descriptors,
        positive_mask,
        float(cfg["training"]["temperature"]),
    )
    zero = contrastive.new_zeros(())
    preservation = zero
    consistency = zero
    if bool(cfg["training"].get("use_clean_anchor", False)):
        clean_features = F.normalize(features[:batch_size].detach(), dim=-1)
        clean_descriptors = F.normalize(descriptors[:batch_size], dim=-1)
        corrupted_descriptors = F.normalize(descriptors[batch_size:], dim=-1)
        consistency = (1.0 - (clean_descriptors * corrupted_descriptors).sum(dim=-1)).mean()
        if batch_size > 1:
            teacher_similarity = clean_features @ clean_features.T
            student_similarity = clean_descriptors @ clean_descriptors.T
            off_diagonal = ~torch.eye(
                batch_size, dtype=torch.bool, device=descriptors.device
            )
            preservation = F.smooth_l1_loss(
                student_similarity[off_diagonal],
                teacher_similarity[off_diagonal],
            )
    total = (
        float(cfg["training"].get("contrastive_weight", 1.0)) * contrastive
        + float(cfg["training"].get("similarity_preservation_weight", 0.0))
        * preservation
        + float(cfg["training"].get("clean_corrupt_consistency_weight", 0.0))
        * consistency
    )
    return total, {
        "contrastive_loss": contrastive,
        "preservation_loss": preservation,
        "consistency_loss": consistency,
    }


def train_model(
    cfg: dict[str, Any],
    manifest_path: str | Path,
    mode: str = "projection_head_only",
    resume: str | Path | None = None,
    limit: int | None = None,
) -> Path:
    """Train an AquaAdapt projection head and return the best checkpoint."""
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        train_indices = manifest.index[manifest["split"] == "train"][:limit]
        manifest = manifest.loc[train_indices].copy()
        manifest["split"] = "train"
    dataset = ContrastiveDataset(
        manifest,
        int(cfg["images"]["model_size"]),
        seed,
        "train",
        cfg["training"],
    )
    if len(dataset) < 2:
        raise ValueError("Training needs at least two train-split images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = min(int(cfg["training"]["batch_size"]), len(dataset))
    sampler = None
    if bool(cfg["training"].get("balance_trajectories", False)):
        counts = dataset.manifest["trajectory_id"].astype(str).value_counts()
        weights = dataset.manifest["trajectory_id"].astype(str).map(
            lambda value: 1.0 / float(counts[value])
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights.to_numpy(), dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=int(cfg["training"]["num_workers"]),
        drop_last=len(dataset) > batch_size, pin_memory=device.type == "cuda",
    )
    validation_dataset = ContrastiveDataset(
        pd.read_csv(manifest_path),
        int(cfg["images"]["model_size"]),
        seed + 10_000,
        "validation",
        cfg["training"],
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]), drop_last=False,
        pin_memory=device.type == "cuda",
    ) if len(validation_dataset) >= 2 else None
    model = AquaAdaptModel(cfg, mode).to(device)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(cfg), weight_decay=float(cfg["training"]["weight_decay"])
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = cosine_scheduler(optimizer, epochs)
    amp_enabled = bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = max(1, int(cfg["training"].get("gradient_accumulation", 1)))

    run_dir = Path(cfg["paths"]["processed_root"]) / "checkpoints" / str(cfg["project"]["run_name"]) / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "system.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_commit(), "device": str(device),
    }, indent=2), encoding="utf-8")
    writer = SummaryWriter(run_dir / "tensorboard")
    start_epoch, best_metric, stale = 0, float("inf"), 0
    if resume:
        state = load_checkpoint(resume, device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state.get("best_metric", best_metric))

    log_path = run_dir / "training_log.csv"
    log_exists = log_path.exists() and start_epoch > 0
    completed_epoch = start_epoch - 1
    stopped_early = False
    with log_path.open("a" if log_exists else "w", newline="", encoding="utf-8") as log_handle:
        fieldnames = [
            "epoch", "train_loss", "validation_loss",
            "contrastive_loss", "preservation_loss", "consistency_loss",
            "learning_rate", "gradient_norm", "elapsed_sec",
        ]
        csv_writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
        if not log_exists:
            csv_writer.writeheader()
        for epoch in range(start_epoch, epochs):
            completed_epoch = epoch
            dataset.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            component_losses: dict[str, list[float]] = {
                "contrastive_loss": [],
                "preservation_loss": [],
                "consistency_loss": [],
            }
            gradient_norm = 0.0
            started = time.perf_counter()
            for step, (view_a, view_b, sample_indices) in enumerate(tqdm(loader, desc=f"train epoch {epoch + 1}/{epochs}")):
                images = torch.cat((view_a, view_b)).to(device, non_blocking=True)
                positive_mask = _positive_mask(
                    len(view_a), device, sample_indices, dataset.manifest, cfg
                )
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        total_loss, components = _adaptation_losses(
                            model, images, positive_mask, len(view_a), cfg
                        )
                        loss = total_loss / accumulation
                    scaler.scale(loss).backward()
                    if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            float(cfg["training"]["gradient_clip_norm"]),
                        )
                        gradient_norm = float(norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    losses.append(float(loss.detach()) * accumulation)
                    for name, value in components.items():
                        component_losses[name].append(float(value.detach()))
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        f"CUDA ran out of memory at batch_size={batch_size}. "
                        "Reduce training.batch_size or increase training.gradient_accumulation; "
                        "image resolution was not changed."
                    ) from exc
            scheduler.step()
            train_loss = float(sum(losses) / max(len(losses), 1))
            validation_losses: list[float] = []
            if validation_loader is not None:
                model.eval()
                with torch.inference_mode():
                    for view_a, view_b, sample_indices in validation_loader:
                        images = torch.cat((view_a, view_b)).to(device, non_blocking=True)
                        mask = _positive_mask(
                            len(view_a), device, sample_indices,
                            validation_dataset.manifest, cfg,
                        )
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                            value, _ = _adaptation_losses(
                                model, images, mask, len(view_a), cfg
                            )
                        validation_losses.append(float(value))
            validation_loss = (
                float(sum(validation_losses) / len(validation_losses))
                if validation_losses else train_loss
            )
            elapsed = time.perf_counter() - started
            learning_rate = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                **{
                    name: float(sum(values) / max(len(values), 1))
                    for name, values in component_losses.items()
                },
                "learning_rate": learning_rate, "gradient_norm": gradient_norm, "elapsed_sec": elapsed,
            }
            csv_writer.writerow(row); log_handle.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            state = {
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "epoch": epoch,
                "best_metric": min(best_metric, validation_loss), "configuration": cfg,
                "backbone_identifier": cfg["model"]["backbone"],
                "descriptor_dimension": cfg["model"]["descriptor_dim"], "random_seed": seed,
                "mode": mode,
            }
            save_checkpoint(state, run_dir / "latest.pt")
            if validation_loss < best_metric:
                best_metric = validation_loss; stale = 0
                save_checkpoint(state, run_dir / "best.pt")
            else:
                stale += 1
                if stale >= int(cfg["training"]["early_stopping_patience"]):
                    LOG.info("Early stopping after epoch %d", epoch + 1)
                    stopped_early = True
                    break
    writer.close()
    plot_training(log_path, run_dir / "training_curves.png")
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "last_epoch_zero_based": completed_epoch,
                "epochs_completed": completed_epoch + 1,
                "configured_epochs": epochs,
                "stopped_early": stopped_early,
                "best_validation_loss": best_metric,
                "best_checkpoint": str((run_dir / "best.pt").resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir / "best.pt"
