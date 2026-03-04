"""
Ablation Study Runner.

Runs all ablation configurations from Table 3 of the proposal:
1. Baseline VLA
2. + Detection Grounding
3. + Detection + CoT
4. + Detection + CoT + Self-Verification
5. Full System (no GRPO)
6. Full System + GRPO
"""

import json
import os
import time
from typing import Dict, List, Optional
from tqdm import tqdm

from src.evaluation.hallucination_metrics import HallucinationMetrics, aggregate_metrics


class AblationRunner:
    """
    Runs ablation experiments across different pipeline configurations.
    """

    CONFIGS = {
        "baseline": {
            "detection": False,
            "cot": False,
            "self_verify": False,
            "grpo": False,
            "description": "Baseline VLA — raw LLaVA with simple prompt",
        },
        "detection_only": {
            "detection": True,
            "cot": False,
            "self_verify": False,
            "grpo": False,
            "description": "+ Object Detection Grounding (YOLO)",
        },
        "detection_cot": {
            "detection": True,
            "cot": True,
            "self_verify": False,
            "grpo": False,
            "description": "+ Detection + Chain-of-Thought",
        },
        "detection_cot_selfverify": {
            "detection": True,
            "cot": True,
            "self_verify": True,
            "grpo": False,
            "description": "+ Detection + CoT + Self-Verification",
        },
        "full_no_grpo": {
            "detection": True,
            "cot": True,
            "self_verify": True,
            "grpo": False,
            "description": "Full System (no GRPO training)",
        },
        "full_grpo": {
            "detection": True,
            "cot": True,
            "self_verify": True,
            "grpo": True,
            "description": "Full System + GRPO Fine-Tuning",
        },
    }

    def __init__(
        self,
        vlm_baseline,
        grounded_vlm,
        cot_builder,
        self_verifier,
        detector,
        metrics: HallucinationMetrics,
        output_dir: str = "./results/ablation",
    ):
        self.vlm_baseline = vlm_baseline
        self.grounded_vlm = grounded_vlm
        self.cot_builder = cot_builder
        self.self_verifier = self_verifier
        self.detector = detector
        self.metrics = metrics
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single_config(
        self,
        config_name: str,
        dataset,
        max_samples: Optional[int] = None,
        grpo_model=None,
    ) -> Dict:
        """
        Run evaluation for a single configuration.

        Returns:
            dict with aggregated metrics and per-sample details.
        """
        config = self.CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Running: {config_name} — {config['description']}")
        print(f"{'='*60}")

        all_sample_metrics = []
        sample_details = []
        n = min(len(dataset), max_samples) if max_samples else len(dataset)

        for idx in tqdm(range(n), desc=config_name):
            sample = dataset[idx]
            image = sample["image"]
            question = sample["question"]
            gt = sample["ground_truth"]

            start_time = time.time()

            # Generate response based on configuration
            response = self._generate_for_config(
                config, image, question, gt, grpo_model
            )

            elapsed = time.time() - start_time

            # Compute metrics
            sample_metric = self.metrics.compute_all_metrics(
                response=response,
                gt_objects=gt["unique_objects"],
                gt_counts=gt.get("object_counts"),
                gt_spatial=sample.get("spatial_relations"),
            )

            all_sample_metrics.append(sample_metric)
            sample_details.append({
                "image_id": sample["image_id"],
                "question": question,
                "response": response,
                "gt_objects": list(gt["unique_objects"]),
                "hallucinated": sample_metric.get("hallucinated_list", []),
                "missed": sample_metric.get("missed_list", []),
                "has_hallucination": sample_metric["has_hallucination"],
                "composite_score": sample_metric["composite_score"],
                "time_seconds": elapsed,
            })

        # Aggregate
        agg = aggregate_metrics(all_sample_metrics)
        agg["config_name"] = config_name
        agg["config_description"] = config["description"]

        # Save detailed results
        detail_path = os.path.join(
            self.output_dir, f"{config_name}_details.json"
        )
        with open(detail_path, "w") as f:
            json.dump(sample_details, f, indent=2, default=str)

        print(f"\nResults for {config_name}:")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        return agg

    def _generate_for_config(
        self,
        config: Dict,
        image,
        question: str,
        gt: Dict,
        grpo_model=None,
    ) -> str:
        """Generate response based on pipeline configuration."""

        if not config["detection"]:
            # Pure baseline — no grounding
            return self.vlm_baseline.generate(image, question)

        # Get detections
        detections = self.detector.detect(image)
        scene_summary = self.detector.format_as_scene_summary(detections)
        detected_objects = self.detector.get_detected_categories(detections)

        if config["cot"]:
            # Build CoT prompt
            prompt = self.cot_builder.build_grounded_cot_prompt(
                question=question,
                scene_summary=scene_summary,
                detections=detections,
            )
        else:
            # Simple grounded prompt
            prompt = self.grounded_vlm.build_grounded_prompt(
                question, detections
            )

        # Generate response
        response = self.vlm_baseline.generate(image, prompt)

        # Self-verification
        if config["self_verify"]:
            verify_result = self.self_verifier.verify_and_correct(
                vlm=self.vlm_baseline,
                image=image,
                original_response=response,
                detected_objects=detected_objects,
                scene_summary=scene_summary,
            )
            response = verify_result["corrected_response"]

        return response

    def run_all_configs(
        self,
        dataset,
        max_samples: Optional[int] = None,
        grpo_model=None,
    ) -> Dict[str, Dict]:
        """Run all ablation configurations."""
        results = {}

        for config_name in self.CONFIGS:
            agg = self.run_single_config(
                config_name=config_name,
                dataset=dataset,
                max_samples=max_samples,
                grpo_model=grpo_model,
            )
            results[config_name] = agg

        # Save summary
        summary_path = os.path.join(self.output_dir, "ablation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nAblation summary saved: {summary_path}")
        return results
