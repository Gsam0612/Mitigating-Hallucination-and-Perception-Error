"""
Hallucination Metrics Module.

Computes comprehensive metrics for evaluating VLM hallucinations
across all four types: existence, misidentification, attribute, spatial.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.constants import COCO_CATEGORIES as _SHARED_COCO_CATEGORIES


class HallucinationMetrics:
    """
    Compute hallucination metrics by comparing VLM outputs
    against ground-truth annotations.
    """

    COCO_CATEGORIES = _SHARED_COCO_CATEGORIES

    def extract_mentioned_objects(self, response: str) -> Set[str]:
        """Extract object categories mentioned in the response."""
        response_lower = response.lower()
        return {obj for obj in self.COCO_CATEGORIES if re.search(rf'\b{re.escape(obj)}\b', response_lower)}

    def object_existence_metrics(
        self,
        mentioned: Set[str],
        gt_objects: Set[str],
    ) -> Dict[str, float]:
        """
        Compute object-existence hallucination metrics.

        Measures: precision, recall, F1, hallucination rate.
        """
        gt_norm = {obj.lower() for obj in gt_objects} & self.COCO_CATEGORIES

        tp = len(mentioned & gt_norm)
        fp = len(mentioned - gt_norm)    # Hallucinated objects
        fn = len(gt_norm - mentioned)     # Missed objects

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        hallucination_rate = fp / (tp + fp) if (tp + fp) > 0 else 0

        return {
            "obj_precision": precision,
            "obj_recall": recall,
            "obj_f1": f1,
            "obj_hallucination_rate": hallucination_rate,
            "obj_true_positives": tp,
            "obj_false_positives": fp,
            "obj_false_negatives": fn,
            "hallucinated_objects": mentioned - gt_norm,
            "missed_objects": gt_norm - mentioned,
        }

    def count_accuracy(
        self,
        response: str,
        gt_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Evaluate counting accuracy.

        Extracts numeric mentions and compares against GT counts.
        """
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }

        response_lower = response.lower()
        correct = 0
        incorrect = 0
        evaluated = 0

        for obj, gt_count in gt_counts.items():
            obj_lower = obj.lower()
            if obj_lower not in response_lower:
                continue

            claimed = None

            # Check word numbers
            for word, num in number_words.items():
                if f"{word} {obj_lower}" in response_lower:
                    claimed = num
                    break

            # Check digit patterns
            pattern = rf'(\d+)\s+{re.escape(obj_lower)}'
            match = re.search(pattern, response_lower)
            if match:
                claimed = int(match.group(1))

            if claimed is not None:
                evaluated += 1
                if claimed == gt_count:
                    correct += 1
                else:
                    incorrect += 1

        accuracy = correct / evaluated if evaluated > 0 else 0

        return {
            "count_accuracy": accuracy,
            "count_correct": correct,
            "count_incorrect": incorrect,
            "count_evaluated": evaluated,
        }

    def spatial_accuracy(
        self,
        response: str,
        gt_spatial: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate spatial relation accuracy.

        Checks pairwise spatial claims against ground-truth relations.
        """
        relation_keywords = {
            "left-of": ["left of", "to the left of"],
            "right-of": ["right of", "to the right of"],
            "above": ["above", "over", "on top of"],
            "below": ["below", "under", "beneath", "underneath"],
        }

        response_lower = response.lower()
        correct = 0
        incorrect = 0
        evaluated = 0

        for rel in gt_spatial:
            subj = rel["subject"].lower()
            obj = rel["object"].lower()
            gt_rel = rel["relation"]

            if subj not in response_lower or obj not in response_lower:
                continue

            for rel_type, keywords in relation_keywords.items():
                for kw in keywords:
                    if kw in response_lower:
                        evaluated += 1
                        if rel_type == gt_rel:
                            correct += 1
                        else:
                            incorrect += 1
                        break

        accuracy = correct / evaluated if evaluated > 0 else 0

        return {
            "spatial_accuracy": accuracy,
            "spatial_correct": correct,
            "spatial_incorrect": incorrect,
            "spatial_evaluated": evaluated,
        }

    def compute_all_metrics(
        self,
        response: str,
        gt_objects: Set[str],
        gt_counts: Optional[Dict[str, int]] = None,
        gt_spatial: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Compute all hallucination metrics for a single response.
        """
        mentioned = self.extract_mentioned_objects(response)

        # Object existence
        obj_metrics = self.object_existence_metrics(mentioned, gt_objects)

        # Counting
        count_metrics = {"count_accuracy": 0, "count_correct": 0, "count_incorrect": 0, "count_evaluated": 0}
        if gt_counts:
            count_metrics = self.count_accuracy(response, gt_counts)

        # Spatial
        spatial_metrics = {"spatial_accuracy": 0, "spatial_correct": 0, "spatial_incorrect": 0, "spatial_evaluated": 0}
        if gt_spatial:
            spatial_metrics = self.spatial_accuracy(response, gt_spatial)

        # Has any hallucination?
        has_hallucination = (
            obj_metrics["obj_false_positives"] > 0
            or count_metrics["count_incorrect"] > 0
            or spatial_metrics["spatial_incorrect"] > 0
        )

        # Composite score
        composite = (
            obj_metrics["obj_f1"] * 0.4
            + (1 - obj_metrics["obj_hallucination_rate"]) * 0.3
            + count_metrics["count_accuracy"] * 0.15
            + spatial_metrics["spatial_accuracy"] * 0.15
        )

        return {
            **{k: v for k, v in obj_metrics.items()
               if not isinstance(v, set)},
            **count_metrics,
            **spatial_metrics,
            "has_hallucination": has_hallucination,
            "composite_score": composite,
            "mentioned_objects": mentioned,
            "hallucinated_list": list(obj_metrics.get("hallucinated_objects", set())),
            "missed_list": list(obj_metrics.get("missed_objects", set())),
        }


def aggregate_metrics(all_metrics: List[Dict]) -> Dict[str, float]:
    """Aggregate metrics across multiple samples."""
    n = len(all_metrics)
    if n == 0:
        return {}

    # Numeric keys to aggregate
    numeric_keys = [
        "obj_precision", "obj_recall", "obj_f1", "obj_hallucination_rate",
        "count_accuracy", "spatial_accuracy", "composite_score",
    ]

    agg = {}
    for key in numeric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            agg[f"mean_{key}"] = sum(values) / len(values)

    # Hallucination rate (% of samples with any hallucination)
    halluc_count = sum(1 for m in all_metrics if m.get("has_hallucination", False))
    agg["sample_hallucination_rate"] = halluc_count / n

    # Total counts
    agg["total_true_positives"] = sum(m.get("obj_true_positives", 0) for m in all_metrics)
    agg["total_false_positives"] = sum(m.get("obj_false_positives", 0) for m in all_metrics)
    agg["total_false_negatives"] = sum(m.get("obj_false_negatives", 0) for m in all_metrics)
    agg["num_samples"] = n

    return agg
