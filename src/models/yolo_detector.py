"""
YOLOv8 Object Detection Module.

Provides structured object detections that are injected into
VLM prompts for grounded perception.
"""

import torch
from typing import Dict, List, Optional, Tuple

from PIL import Image
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8-based object detector for scene grounding.

    Processes images and returns structured detection results
    that can be formatted as text prompts for the VLM.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        device: str = "cuda",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device

        print(f"Loading YOLOv8 model: {model_name}...")
        self.model = YOLO(model_name)
        print("YOLOv8 loaded successfully.")

    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run object detection on a single image.

        Returns:
            List of detection dicts with keys:
            - category: str (object class name)
            - confidence: float
            - bbox: [x1, y1, x2, y2]
            - center: (cx, cy)
            - area: float
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                category = result.names[cls_id]

                detections.append({
                    "category": category,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": (round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)),
                    "area": round((x2 - x1) * (y2 - y1), 1),
                })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def format_as_scene_summary(self, detections: List[Dict]) -> str:
        """
        Format detections as a human-readable scene summary
        for injection into VLM prompts.

        Example output:
        "Detected objects: cup (0.92), plate (0.87), microwave (0.81).
         Object counts: cup: 1, plate: 2, microwave: 1."
        """
        if not detections:
            return "Detected objects: None detected with sufficient confidence."

        # Group by category and count
        counts: Dict[str, int] = {}
        for det in detections:
            cat = det["category"]
            counts[cat] = counts.get(cat, 0) + 1

        # Object list with confidence
        obj_list = ", ".join(
            f"{d['category']} ({d['confidence']:.2f})" for d in detections
        )

        # Count summary
        count_str = ", ".join(f"{cat}: {cnt}" for cat, cnt in counts.items())

        return (
            f"Detected objects: {obj_list}.\n"
            f"Object counts: {count_str}."
        )

    def format_with_spatial(self, detections: List[Dict]) -> str:
        """
        Format detections with spatial information for
        enhanced grounding.
        """
        if not detections:
            return "No objects detected."

        lines = ["Detected objects with positions:"]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            lines.append(
                f"  - {det['category']} (conf: {det['confidence']:.2f}) "
                f"at position ({cx:.0f}, {cy:.0f}), "
                f"bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
            )

        # Add pairwise spatial relations for top objects
        top_dets = detections[:8]  # Limit to avoid explosion
        relations = []
        for i, det_a in enumerate(top_dets):
            for j, det_b in enumerate(top_dets):
                if i >= j:
                    continue
                rel = self._spatial_relation(det_a, det_b)
                relations.append(
                    f"  - {det_a['category']} is {rel} {det_b['category']}"
                )

        if relations:
            lines.append("\nSpatial relations:")
            lines.extend(relations[:10])  # Limit to top 10

        return "\n".join(lines)

    @staticmethod
    def _spatial_relation(det_a: Dict, det_b: Dict) -> str:
        """Compute spatial relation between two detections."""
        cx_a, cy_a = det_a["center"]
        cx_b, cy_b = det_b["center"]

        dx = cx_b - cx_a
        dy = cy_b - cy_a

        if abs(dx) > abs(dy):
            return "to the left of" if dx > 0 else "to the right of"
        else:
            return "above" if dy > 0 else "below"

    def get_detected_categories(self, detections: List[Dict]) -> set:
        """Return set of unique category names from detections."""
        return {d["category"] for d in detections}

    def get_category_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """Return category count dict from detections."""
        counts: Dict[str, int] = {}
        for det in detections:
            cat = det["category"]
            counts[cat] = counts.get(cat, 0) + 1
        return counts
