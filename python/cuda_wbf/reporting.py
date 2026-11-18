"""Summarize and publish the benchmark's validated artifacts."""

from collections import defaultdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .preflight import CUDA_BACKENDS
from .timing import HOST


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["workload"], row["backend"], row["boundary"]].append(row)
    stats = []
    for (workload, backend, boundary), samples in grouped.items():
        p50, p95 = np.percentile([row["latency_ms"] for row in samples], [50, 95])
        batch_size = samples[0]["batch_size"]
        stats.append({
            "workload": workload, "backend": backend, "boundary": boundary, "batch_size": batch_size,
            "p50_ms": float(p50), "p95_ms": float(p95), "images_per_s": float(batch_size * 1000 / p50) if p50 else None,
            "max_upload_bytes": max(row["upload_bytes"] for row in samples),
            "max_download_bytes": max(row["download_bytes"] for row in samples),
        })
    reference = {row["workload"]: row["p50_ms"] for row in stats
                 if row["backend"] == "python" and row["boundary"] == HOST}
    for row in stats:
        row["speedup"] = (reference[row["workload"]] / row["p50_ms"]
                          if row["boundary"] == HOST and row["p50_ms"] else None)
    return stats


def _format(value, decimals):
    return "—" if value is None else f"{value:.{decimals}f}"


def markdown_summary(rows, quality):
    stats = summarize(rows)
    lines = [
        "# CUDA WBF benchmark results", "",
        f"Full-image WBF parity and COCO AP agreement passed on {quality['images']:,} images.", "",
        "## Host-to-host crossovers", "",
        "A workload crosses over when the CUDA median latency is lower than the Python WBF median.", "",
    ]
    for backend in CUDA_BACKENDS:
        won = [f"`{row['workload']}` ({row['speedup']:.3f}×)" for row in stats
               if row["backend"] == backend and (row["speedup"] or 0) > 1]
        lines.append(f"- **{backend}** " + (", ".join(won) if won else "had no crossover workloads."))

    lines.extend([
        "", "## Latency and transfers", "",
        "Images per second uses batch size divided by p50 latency. Transfer columns show the maximum bytes per sample.",
        "", "| Workload | Method | Boundary | p50 ms | p95 ms | Images/s | Python speedup | Upload B | Download B |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in stats:
        lines.append(
            f"| {row['workload']} | {row['backend']} | {row['boundary']} | "
            f"{row['p50_ms']:.6f} | {row['p95_ms']:.6f} | {_format(row['images_per_s'], 1)} | "
            f"{_format(row['speedup'], 4)} | {row['max_upload_bytes']} | {row['max_download_bytes']} |"
        )
    lines.extend([
        "", "## COCO detection quality", "",
        "| Method | AP | AP50 | AP small | AP medium | AP large | AR100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for backend, result in quality["methods"].items():
        metrics = " | ".join(f"{result['metrics'][name]:.6f}"
                             for name in ("AP", "AP50", "AP_small", "AP_medium", "AP_large", "AR100"))
        lines.append(f"| {backend} | {metrics} |")
    return "\n".join(lines) + "\n"


def write_results(output, rows, metadata, quality):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
        staged = Path(directory)
        with (staged / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for filename, document in (("metadata.json", metadata), ("quality.json", quality)):
            (staged / filename).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staged / "summary.md").write_text(markdown_summary(rows, quality), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"choose a new output directory: {output}")
        staged.rename(output)
