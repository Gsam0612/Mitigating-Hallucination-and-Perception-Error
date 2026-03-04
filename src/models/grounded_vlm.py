"""
Grounded VLM — VLM with Detection-Enhanced Prompting.

Combines YOLOv8 detections with LLaVA to produce
perception-grounded outputs that reduce hallucinations.
"""

from typing import Dict, List, Optional

from PIL import Image

from src.models.vlm_baseline import BaselineVLM
from src.models.yolo_detector import YOLODetector


class GroundedVLM:
    """
    Vision-Language Model with object-detection grounding.

    Injects YOLO detection results into the VLM prompt
    to anchor generation in verifiable visual evidence.
    """

    def __init__(
        self,
        vlm: BaselineVLM,
        detector: YOLODetector,
        use_spatial: bool = True,
    ):
        self.vlm = vlm
        self.detector = detector
        self.use_spatial = use_spatial

    def build_grounded_prompt(
        self,
        question: str,
        detections: List[Dict],
        include_spatial: bool = True,
    ) -> str:
        """
        Build a prompt that includes detection grounding.

        The prompt instructs the VLM to only mention objects
        supported by the detector output.
        """
        if include_spatial and self.use_spatial:
            scene_summary = self.detector.format_with_spatial(detections)
        else:
            scene_summary = self.detector.format_as_scene_summary(detections)

        prompt = (
            f"You are a precise visual assistant. An object detector has analyzed "
            f"this image and found the following:\n\n"
            f"{scene_summary}\n\n"
            f"IMPORTANT: Only mention objects that are supported by the detector output "
            f"or that you can clearly see in the image. Do NOT guess or assume objects "
            f"that are not visible. If you are unsure about an object, say so.\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the image and the detected objects:"
        )
        return prompt

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Generate a grounded response.

        Returns:
            dict with:
            - response: str (generated text)
            - detections: list of detection dicts
            - scene_summary: str
            - prompt: str (full prompt used)
        """
        # Step 1: Run object detection
        detections = self.detector.detect(image)

        # Step 2: Build grounded prompt
        prompt = self.build_grounded_prompt(question, detections)

        # Step 3: Generate response
        response = self.vlm.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        return {
            "response": response,
            "detections": detections,
            "scene_summary": self.detector.format_as_scene_summary(detections),
            "prompt": prompt,
        }

    def generate_candidates(
        self,
        image: Image.Image,
        question: str,
        k: int = 4,
        temperature: float = 0.8,
    ) -> Dict:
        """
        Generate K candidate responses for GRPO training.

        All candidates use the same detections (consistent grounding).
        """
        detections = self.detector.detect(image)
        prompt = self.build_grounded_prompt(question, detections)

        candidates = self.vlm.generate_multiple_candidates(
            image=image,
            prompt=prompt,
            k=k,
            temperature=temperature,
        )

        return {
            "candidates": candidates,
            "detections": detections,
            "prompt": prompt,
        }
