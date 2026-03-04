"""
POPE Benchmark Loader for Hallucination Evaluation.

POPE (Polling-based Object Probing Evaluation) tests object-existence
hallucinations using yes/no questions in three settings:
random, popular, and adversarial.
"""

import json
import os
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class POPEBenchmark(Dataset):
    """
    Loads POPE-format evaluation data.

    Expected format per line (JSONL):
    {
        "question_id": int,
        "image": "COCO_val2014_000000XXXXXX.jpg",
        "text": "Is there a <object> in the image?",
        "label": "yes" or "no"
    }
    """

    def __init__(
        self,
        pope_file: str,
        images_dir: str,
        category: str = "random",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            pope_file: Path to POPE JSONL file.
            images_dir: Directory containing COCO images.
            category: One of 'random', 'popular', 'adversarial'.
            max_samples: Limit number of questions.
        """
        self.images_dir = images_dir
        self.category = category
        self.samples: List[Dict] = []

        with open(pope_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.samples.append(entry)

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image_path = os.path.join(self.images_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")

        return {
            "question_id": sample.get("question_id", idx),
            "image": image,
            "image_path": image_path,
            "question": sample["text"],
            "label": sample["label"].lower().strip(),
            "category": self.category,
        }


def load_pope_benchmarks(
    pope_dir: str,
    images_dir: str,
    max_samples_per_cat: Optional[int] = None,
) -> Dict[str, POPEBenchmark]:
    """
    Load all three POPE evaluation categories.

    Returns:
        dict mapping category name to POPEBenchmark dataset.
    """
    benchmarks = {}
    categories = ["random", "popular", "adversarial"]

    for cat in categories:
        # Common POPE file naming conventions
        possible_names = [
            f"coco_pope_{cat}.json",
            f"pope_{cat}.jsonl",
            f"{cat}.jsonl",
        ]
        for name in possible_names:
            path = os.path.join(pope_dir, name)
            if os.path.exists(path):
                benchmarks[cat] = POPEBenchmark(
                    pope_file=path,
                    images_dir=images_dir,
                    category=cat,
                    max_samples=max_samples_per_cat,
                )
                break

    return benchmarks


def evaluate_pope_predictions(
    predictions: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """
    Compute POPE evaluation metrics.

    Args:
        predictions: List of model outputs (will be parsed to yes/no).
        labels: List of ground-truth labels ('yes' or 'no').

    Returns:
        dict with accuracy, precision, recall, f1, yes_ratio.
    """
    assert len(predictions) == len(labels)

    tp = fp = tn = fn = 0

    for pred, label in zip(predictions, labels):
        # Parse prediction to yes/no
        pred_lower = pred.lower().strip()
        pred_yes = "yes" in pred_lower and "no" not in pred_lower
        label_yes = label.lower().strip() == "yes"

        if pred_yes and label_yes:
            tp += 1
        elif pred_yes and not label_yes:
            fp += 1
        elif not pred_yes and not label_yes:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = (tp + fp) / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
