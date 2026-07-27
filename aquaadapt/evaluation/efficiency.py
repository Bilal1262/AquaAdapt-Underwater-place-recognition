"""Inference efficiency benchmark."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def benchmark_model(model: torch.nn.Module, descriptor_dim: int, device: torch.device, image_size: int, checkpoint: str | None = None) -> dict[str, Any]:
    model.to(device).eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(5):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times: list[float] = []
        for _ in range(30):
            started = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total, "trainable_parameters": trainable,
        "checkpoint_size_bytes": Path(checkpoint).stat().st_size if checkpoint and Path(checkpoint).is_file() else None,
        "descriptor_dimension": descriptor_dim, "average_latency_ms": float(np.mean(times)),
        "median_latency_ms": float(np.median(times)), "p95_latency_ms": float(np.percentile(times, 95)),
        "frames_per_second": float(1000 / np.mean(times)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "device": str(device),
    }


def save_efficiency(rows: list[dict[str, Any]], path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
