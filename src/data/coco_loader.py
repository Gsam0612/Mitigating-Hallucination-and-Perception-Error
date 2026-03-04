"""
COCO Dataset Loader for VLA Hallucination Mitigation.

Loads COCO 2017 images and annotations to provide ground-truth
object lists, attributes, and bounding boxes for reward computation
and evaluation.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class COCOGroundTruth:
    """Parses COCO annotations into structured ground-truth for reward scoring."""

    def __init__(self, ann_file: str, images_dir: str):
        self.images_dir = images_dir

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        # Build category id -> name mapping
        self.cat_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        # Build image id -> annotations mapping
        self.img_to_anns: Dict[int, List[dict]] = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Build image id -> image info mapping
        self.img_id_to_info = {
            img["id"]: img for img in self.coco_data["images"]
        }

        self.image_ids = list(self.img_id_to_info.keys())

    def get_image_path(self, image_id: int) -> str:
        """Get the file path for an image."""
        info = self.img_id_to_info[image_id]
        return os.path.join(self.images_dir, info["file_name"])

    def get_ground_truth(self, image_id: int) -> Dict:
        """
        Extract structured ground-truth for a single image.

        Returns:
            dict with keys:
                - image_id: int
                - image_path: str
                - objects: list of {category, bbox, area, iscrowd}
                - object_counts: dict {category_name: count}
                - unique_objects: set of category names present
        """
        anns = self.img_to_anns.get(image_id, [])
        objects = []
        object_counts: Dict[str, int] = {}

        for ann in anns:
            cat_name = self.cat_id_to_name[ann["category_id"]]
            objects.append({
                "category": cat_name,
                "bbox": ann["bbox"],  # [x, y, width, height]
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            object_counts[cat_name] = object_counts.get(cat_name, 0) + 1

        return {
            "image_id": image_id,
            "image_path": self.get_image_path(image_id),
            "objects": objects,
            "object_counts": object_counts,
            "unique_objects": set(object_counts.keys()),
        }

    def get_spatial_relations(self, image_id: int) -> List[Dict]:
        """
        Compute pairwise spatial relations from bounding boxes.

        Relations: left-of, right-of, above, below, overlaps
        """
        anns = self.img_to_anns.get(image_id, [])
        relations = []

        for i, ann_a in enumerate(anns):
            for j, ann_b in enumerate(anns):
                if i >= j:
                    continue
                cat_a = self.cat_id_to_name[ann_a["category_id"]]
                cat_b = self.cat_id_to_name[ann_b["category_id"]]
                rel = self._compute_relation(ann_a["bbox"], ann_b["bbox"])
                relations.append({
                    "subject": cat_a,
                    "object": cat_b,
                    "relation": rel,
                    "subject_bbox": ann_a["bbox"],
                    "object_bbox": ann_b["bbox"],
                })

        return relations

    @staticmethod
    def _compute_relation(bbox_a: List[float], bbox_b: List[float]) -> str:
        """Compute spatial relation between two bboxes [x, y, w, h]."""
        cx_a = bbox_a[0] + bbox_a[2] / 2
        cy_a = bbox_a[1] + bbox_a[3] / 2
        cx_b = bbox_b[0] + bbox_b[2] / 2
        cy_b = bbox_b[1] + bbox_b[3] / 2

        dx = cx_b - cx_a
        dy = cy_b - cy_a

        if abs(dx) > abs(dy):
            return "left-of" if dx > 0 else "right-of"
        else:
            return "above" if dy > 0 else "below"


class COCOHallucinationDataset(Dataset):
    """
    PyTorch Dataset for hallucination evaluation.

    Each item provides an image + ground-truth + a question prompt.
    """

    QUESTION_TEMPLATES = [
        "Describe all objects you can see in this image.",
        "List every object visible in this image with their positions.",
        "What objects are present in this scene? Be precise and only mention what you can actually see.",
        "Carefully examine this image. What objects do you observe and where are they located?",
    ]

    def __init__(
        self,
        coco_gt: COCOGroundTruth,
        image_ids: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.coco_gt = coco_gt
        random.seed(seed)

        # Filter to images that have annotations
        available_ids = [
            img_id for img_id in (image_ids or coco_gt.image_ids)
            if img_id in coco_gt.img_to_anns and len(coco_gt.img_to_anns[img_id]) > 0
        ]

        if max_samples and max_samples < len(available_ids):
            available_ids = random.sample(available_ids, max_samples)

        self.image_ids = available_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        gt = self.coco_gt.get_ground_truth(image_id)
        spatial = self.coco_gt.get_spatial_relations(image_id)

        # Load image
        image = Image.open(gt["image_path"]).convert("RGB")

        # Select a question prompt
        question = self.QUESTION_TEMPLATES[idx % len(self.QUESTION_TEMPLATES)]

        return {
            "image_id": image_id,
            "image": image,
            "question": question,
            "ground_truth": gt,
            "spatial_relations": spatial,
        }


def create_train_eval_split(
    coco_gt: COCOGroundTruth,
    train_ratio: float = 0.8,
    max_train: int = 2000,
    max_eval: int = 500,
    seed: int = 42,
) -> Tuple[COCOHallucinationDataset, COCOHallucinationDataset]:
    """Create train/eval splits from COCO ground truth."""
    random.seed(seed)
    all_ids = list(coco_gt.img_to_anns.keys())
    random.shuffle(all_ids)

    split_idx = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:split_idx]
    eval_ids = all_ids[split_idx:]

    train_dataset = COCOHallucinationDataset(
        coco_gt, image_ids=train_ids, max_samples=max_train, seed=seed
    )
    eval_dataset = COCOHallucinationDataset(
        coco_gt, image_ids=eval_ids, max_samples=max_eval, seed=seed
    )

    return train_dataset, eval_dataset
