"""
Visualization Module for Results Analysis.

Generates publication-quality charts for:
- Ablation comparison bar charts
- Training reward curves
- Hallucination type breakdown
- Per-config metric comparison tables
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
import numpy as np


def plot_ablation_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    output_path: str = "./results/ablation_comparison.png",
    figsize: Tuple = (14, 6),
):
    """
    Bar chart comparing metrics across ablation configurations.
    """
    if metrics is None:
        metrics = [
            "mean_obj_precision",
            "mean_obj_recall",
            "mean_obj_f1",
            "mean_obj_hallucination_rate",
            "mean_composite_score",
        ]

    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.15

    fig, ax = plt.subplots(figsize=figsize)

    for i, metric in enumerate(metrics):
        values = [results[c].get(metric, 0) for c in configs]
        label = metric.replace("mean_", "").replace("_", " ").title()
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xlabel("Pipeline Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Impact of Each Mitigation Component")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(
        [results[c].get("config_description", c)[:25] for c in configs],
        rotation=30,
        ha="right",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation chart saved: {output_path}")


def plot_training_curves(
    training_logs: List[Dict],
    output_path: str = "./results/training_curves.png",
):
    """Plot GRPO training reward and loss curves."""
    steps = [log["step"] for log in training_logs]
    rewards = [log["mean_reward"] for log in training_logs]
    losses = [log["loss"] for log in training_logs]
    halluc_avg = [log.get("num_hallucinated_avg", 0) for log in training_logs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Reward curve
    axes[0].plot(steps, rewards, color="green", alpha=0.7)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("GRPO Training: Mean Reward")
    axes[0].grid(alpha=0.3)

    # Loss curve
    axes[1].plot(steps, losses, color="red", alpha=0.7)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("GRPO Training: Policy Loss")
    axes[1].grid(alpha=0.3)

    # Hallucination count
    axes[2].plot(steps, halluc_avg, color="orange", alpha=0.7)
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Avg Hallucinated Objects")
    axes[2].set_title("GRPO Training: Hallucination Reduction")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {output_path}")


def plot_hallucination_breakdown(
    sample_details: List[Dict],
    output_path: str = "./results/hallucination_breakdown.png",
):
    """Pie chart / bar chart of hallucination types."""
    halluc_objects = []
    for s in sample_details:
        halluc_objects.extend(s.get("hallucinated", []))

    if not halluc_objects:
        print("No hallucinations to plot.")
        return

    # Count most common hallucinated objects
    from collections import Counter
    counts = Counter(halluc_objects).most_common(15)

    objects, freqs = zip(*counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(objects)), freqs, color=sns.color_palette("Reds_r", len(objects)))
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects)
    ax.set_xlabel("Frequency")
    ax.set_title("Most Commonly Hallucinated Objects")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hallucination breakdown saved: {output_path}")


def create_results_table(
    results: Dict[str, Dict],
    output_path: str = "./results/results_table.csv",
) -> pd.DataFrame:
    """Create a formatted results table as CSV and DataFrame."""
    rows = []
    for config_name, metrics in results.items():
        row = {
            "Configuration": metrics.get("config_description", config_name),
            "Obj Precision": f"{metrics.get('mean_obj_precision', 0):.3f}",
            "Obj Recall": f"{metrics.get('mean_obj_recall', 0):.3f}",
            "Obj F1": f"{metrics.get('mean_obj_f1', 0):.3f}",
            "Halluc Rate": f"{metrics.get('mean_obj_hallucination_rate', 0):.3f}",
            "Count Acc": f"{metrics.get('mean_count_accuracy', 0):.3f}",
            "Spatial Acc": f"{metrics.get('mean_spatial_accuracy', 0):.3f}",
            "Composite": f"{metrics.get('mean_composite_score', 0):.3f}",
            "Sample Halluc %": f"{metrics.get('sample_hallucination_rate', 0)*100:.1f}%",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results table saved: {output_path}")
    return df


def plot_pope_results(
    pope_results: Dict[str, Dict],
    output_path: str = "./results/pope_results.png",
):
    """Plot POPE benchmark results across categories."""
    categories = list(pope_results.keys())
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "yes_ratio"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.15

    for i, metric in enumerate(metrics_to_plot):
        values = [pope_results[cat].get(metric, 0) for cat in categories]
        ax.bar(x + i * width, values, width, label=metric.title())

    ax.set_xlabel("POPE Category")
    ax.set_ylabel("Score")
    ax.set_title("POPE Benchmark Results")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"POPE results saved: {output_path}")
