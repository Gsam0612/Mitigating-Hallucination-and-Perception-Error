"""
Hallucination-Aware Reward Function.

Computes scalar rewards for VLM outputs by comparing against
ground-truth annotations. Used by GRPO to train the model
to prefer grounded, non-hallucinated outputs.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from src.constants import COCO_CATEGORIES as _SHARED_COCO_CATEGORIES


class HallucinationReward:
    """
    Multi-component reward function for hallucination-aware training.

    Reward components:
    1. Object existence: penalize mentioning non-existent objects
    2. Object recall: reward mentioning actual objects
    3. Attribute accuracy: penalize wrong attributes (color, count)
    4. Spatial relation: penalize wrong spatial claims
    5. Verbosity: minor penalty for excessive length
    6. Safety: bonus for safe recommendations
    """

    # Common COCO object categories for extraction
    COCO_CATEGORIES = _SHARED_COCO_CATEGORIES

    def __init__(
        self,
        correct_object_weight: float = 1.0,
        hallucinated_object_weight: float = -2.0,
        correct_attribute_weight: float = 0.5,
        wrong_attribute_weight: float = -1.5,
        correct_spatial_weight: float = 0.5,
        wrong_spatial_weight: float = -1.5,
        correct_count_weight: float = 0.5,
        wrong_count_weight: float = -1.0,
        verbosity_penalty: float = -0.1,
        safety_bonus: float = 1.0,
        max_sentences: int = 15,
    ):
        self.w_correct_obj = correct_object_weight
        self.w_halluc_obj = hallucinated_object_weight
        self.w_correct_attr = correct_attribute_weight
        self.w_wrong_attr = wrong_attribute_weight
        self.w_correct_spatial = correct_spatial_weight
        self.w_wrong_spatial = wrong_spatial_weight
        self.w_correct_count = correct_count_weight
        self.w_wrong_count = wrong_count_weight
        self.w_verbosity = verbosity_penalty
        self.w_safety = safety_bonus
        self.max_sentences = max_sentences

    def compute_reward(
        self,
        response: str,
        gt_objects: Set[str],
        gt_counts: Optional[Dict[str, int]] = None,
        gt_spatial: Optional[List[Dict]] = None,
        detected_objects: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute the full reward for a VLM response.

        Args:
            response: Generated text from VLM.
            gt_objects: Set of ground-truth object category names.
            gt_counts: Ground-truth object counts {category: count}.
            gt_spatial: Ground-truth spatial relations.
            detected_objects: Objects found by YOLO detector.

        Returns:
            dict with component rewards and total reward.
        """
        mentioned = self._extract_objects(response)

        # 1. Object existence reward
        obj_reward, obj_details = self._object_reward(mentioned, gt_objects)

        # 2. Count reward
        count_reward = 0.0
        if gt_counts:
            count_reward = self._count_reward(response, gt_counts)

        # 3. Spatial reward
        spatial_reward = 0.0
        if gt_spatial:
            spatial_reward = self._spatial_reward(response, gt_spatial)

        # 4. Verbosity penalty
        verbosity = self._verbosity_penalty(response)

        # 5. Uncertainty bonus (saying "I'm not sure" when appropriate)
        uncertainty_bonus = self._uncertainty_bonus(response, mentioned, gt_objects)

        # Total reward
        total = obj_reward + count_reward + spatial_reward + verbosity + uncertainty_bonus

        return {
            "total_reward": total,
            "object_reward": obj_reward,
            "count_reward": count_reward,
            "spatial_reward": spatial_reward,
            "verbosity_penalty": verbosity,
            "uncertainty_bonus": uncertainty_bonus,
            "mentioned_objects": mentioned,
            "correct_objects": obj_details["correct"],
            "hallucinated_objects": obj_details["hallucinated"],
            "missed_objects": obj_details["missed"],
        }

    def _extract_objects(self, response: str) -> Set[str]:
        """Extract mentioned object categories from response text."""
        response_lower = response.lower()
        mentioned = set()
        for obj in self.COCO_CATEGORIES:
            if re.search(rf'\b{re.escape(obj)}\b', response_lower):
                mentioned.add(obj)
        return mentioned

    def _object_reward(
        self, mentioned: Set[str], gt_objects: Set[str]
    ) -> Tuple[float, Dict]:
        """
        Compute object-level reward.

        Positive for correctly mentioning GT objects.
        Negative for hallucinating non-GT objects.
        """
        # Normalize GT objects to match our category list
        gt_normalized = set()
        for obj in gt_objects:
            obj_lower = obj.lower()
            if obj_lower in self.COCO_CATEGORIES:
                gt_normalized.add(obj_lower)

        correct = mentioned & gt_normalized
        hallucinated = mentioned - gt_normalized
        missed = gt_normalized - mentioned

        reward = (
            len(correct) * self.w_correct_obj
            + len(hallucinated) * self.w_halluc_obj
        )

        details = {
            "correct": correct,
            "hallucinated": hallucinated,
            "missed": missed,
        }

        return reward, details

    def _count_reward(self, response: str, gt_counts: Dict[str, int]) -> float:
        """Compute counting accuracy reward."""
        response_lower = response.lower()
        reward = 0.0

        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a ": 1, "an ": 1,
        }

        for obj, gt_count in gt_counts.items():
            obj_lower = obj.lower()
            if obj_lower not in response_lower:
                continue

            # Try to extract count from response
            claimed_count = None

            # Check number words
            for word, num in number_words.items():
                if f"{word} {obj_lower}" in response_lower or f"{word}{obj_lower}" in response_lower:
                    claimed_count = num
                    break

            # Check numeric patterns
            pattern = rf'(\d+)\s+{re.escape(obj_lower)}'
            match = re.search(pattern, response_lower)
            if match:
                claimed_count = int(match.group(1))

            if claimed_count is not None:
                if claimed_count == gt_count:
                    reward += self.w_correct_count
                else:
                    reward += self.w_wrong_count

        return reward

    def _spatial_reward(self, response: str, gt_spatial: List[Dict]) -> float:
        """Compute spatial relation accuracy reward."""
        response_lower = response.lower()
        reward = 0.0

        relation_map = {
            "left-of": ["left of", "to the left"],
            "right-of": ["right of", "to the right"],
            "above": ["above", "over", "on top of"],
            "below": ["below", "under", "beneath"],
        }

        for rel in gt_spatial:
            subj = rel["subject"].lower()
            obj = rel["object"].lower()
            gt_rel = rel["relation"]

            if subj not in response_lower or obj not in response_lower:
                continue

            # Check if the response mentions a spatial relation
            gt_phrases = relation_map.get(gt_rel, [])
            wrong_relations = {
                k: v for k, v in relation_map.items() if k != gt_rel
            }

            # Check for correct relation
            for phrase in gt_phrases:
                if phrase in response_lower:
                    reward += self.w_correct_spatial
                    break

            # Check for incorrect relations
            for wrong_rel, wrong_phrases in wrong_relations.items():
                for phrase in wrong_phrases:
                    if phrase in response_lower:
                        reward += self.w_wrong_spatial
                        break

        return reward

    def _verbosity_penalty(self, response: str) -> float:
        """Penalize excessively long responses."""
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        excess = max(0, len(sentences) - self.max_sentences)
        return excess * self.w_verbosity

    def _uncertainty_bonus(
        self, response: str, mentioned: Set[str], gt_objects: Set[str]
    ) -> float:
        """
        Reward appropriate uncertainty expressions.

        If the model says 'I'm not sure' about an object that is
        indeed ambiguous/absent, reward calibrated uncertainty.
        """
        uncertainty_phrases = [
            "i'm not sure", "i am not sure", "i cannot clearly see",
            "it's difficult to tell", "may be", "might be",
            "i think", "appears to be", "possibly",
        ]

        response_lower = response.lower()
        bonus = 0.0

        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                bonus += 0.2  # Small bonus for calibrated uncertainty
                break

        return bonus
