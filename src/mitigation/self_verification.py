"""
Self-Verification Module.

After the VLM generates an initial response, this module
generates follow-up verification questions to catch and
correct hallucinations before committing to a final answer.
"""

from typing import Dict, List, Optional, Tuple
import re

from src.constants import COCO_CATEGORIES as _SHARED_COCO_CATEGORIES


class SelfVerifier:
    """
    Implements self-verification by generating probing questions
    about the VLM's initial response and checking them against
    detection grounding.
    """

    # Templates for verification questions
    EXISTENCE_CHECK = "You mentioned '{object}'. Can you confirm you clearly see a {object} in the image? Look carefully."
    ATTRIBUTE_CHECK = "You described '{object}' as '{attribute}'. Re-examine the image — is this attribute correct?"
    COUNT_CHECK = "You said there are {count} {object}(s). Count again carefully by scanning the image from left to right."
    SPATIAL_CHECK = "You said '{obj_a}' is {relation} '{obj_b}'. Look at the image again — is this spatial relationship correct?"

    def __init__(self, max_verification_rounds: int = 1):
        self.max_rounds = max_verification_rounds

    def extract_mentioned_objects(self, response: str) -> List[str]:
        """
        Extract object names mentioned in the VLM response.

        Uses word-boundary regex matching against shared COCO categories.
        """
        coco_objects = _SHARED_COCO_CATEGORIES

        response_lower = response.lower()
        mentioned = []
        for obj in coco_objects:
            if re.search(rf'\b{re.escape(obj)}\b', response_lower):
                mentioned.append(obj)

        return mentioned

    def extract_counts(self, response: str) -> Dict[str, int]:
        """Extract object counts from response text."""
        counts = {}
        # Pattern: "two cups", "3 chairs", "a cup" etc.
        number_words = {
            "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
            "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10,
        }

        response_lower = response.lower()

        # Match patterns like "two cups" or "3 bottles"
        for word, num in number_words.items():
            pattern = rf'\b{word}\s+(\w+)'
            matches = re.findall(pattern, response_lower)
            for match in matches:
                counts[match] = num

        # Match numeric patterns like "3 cups"
        numeric_pattern = r'(\d+)\s+(\w+)'
        matches = re.findall(numeric_pattern, response_lower)
        for num_str, obj in matches:
            counts[obj] = int(num_str)

        return counts

    def generate_verification_questions(
        self,
        response: str,
        detected_objects: set,
    ) -> List[Dict]:
        """
        Generate verification questions based on discrepancies
        between the VLM response and detector output.

        Args:
            response: The VLM's initial response.
            detected_objects: Set of object names from YOLO detector.

        Returns:
            List of verification question dicts.
        """
        questions = []
        mentioned = self.extract_mentioned_objects(response)

        # 1. Check for hallucinated objects (mentioned but not detected)
        for obj in mentioned:
            if obj not in detected_objects:
                questions.append({
                    "type": "existence",
                    "question": self.EXISTENCE_CHECK.format(object=obj),
                    "concern": f"'{obj}' mentioned but not detected by YOLO",
                    "object": obj,
                })

        # 2. Check counts
        counts = self.extract_counts(response)
        for obj, count in counts.items():
            if count > 1:
                questions.append({
                    "type": "count",
                    "question": self.COUNT_CHECK.format(count=count, object=obj),
                    "concern": f"Claimed {count} {obj}(s) — needs verification",
                    "object": obj,
                    "claimed_count": count,
                })

        return questions

    def build_verification_prompt(
        self,
        original_response: str,
        questions: List[Dict],
        scene_summary: str,
    ) -> str:
        """
        Build a self-verification prompt that asks the VLM to
        re-examine its initial response.
        """
        q_text = "\n".join(
            f"  {i+1}. {q['question']}" for i, q in enumerate(questions)
        )

        prompt = (
            f"You previously said:\n"
            f'"{original_response}"\n\n'
            f"The object detector found: {scene_summary}\n\n"
            f"Please re-examine the image and answer these verification questions:\n"
            f"{q_text}\n\n"
            f"Based on your re-examination, provide a CORRECTED final answer. "
            f"Remove any objects you cannot confirm. Fix any incorrect counts "
            f"or spatial relationships. If your original answer was correct, "
            f"repeat it as-is."
        )
        return prompt

    def verify_and_correct(
        self,
        vlm,
        image,
        original_response: str,
        detected_objects: set,
        scene_summary: str,
    ) -> Dict:
        """
        Full self-verification pipeline.

        1. Generate verification questions from initial response
        2. If discrepancies found, ask VLM to re-examine
        3. Return corrected response

        Returns:
            dict with corrected_response, questions, was_corrected.
        """
        questions = self.generate_verification_questions(
            original_response, detected_objects
        )

        if not questions:
            return {
                "corrected_response": original_response,
                "questions": [],
                "was_corrected": False,
            }

        # Build verification prompt and re-query VLM
        verify_prompt = self.build_verification_prompt(
            original_response, questions, scene_summary
        )

        corrected = vlm.generate(
            image=image,
            prompt=verify_prompt,
        )

        return {
            "corrected_response": corrected,
            "questions": questions,
            "was_corrected": True,
            "num_concerns": len(questions),
        }
