"""
Chain-of-Thought (CoT) Prompting Module.

Designs structured prompts that force the VLM to reason
step-by-step before producing a final answer, reducing
impulsive hallucinations.
"""

from typing import Dict, List, Optional


class CoTPromptBuilder:
    """
    Builds Chain-of-Thought prompts for structured visual reasoning.

    The CoT approach forces the model to:
    1. Enumerate visible objects and attributes
    2. Reason about spatial relationships
    3. Synthesize into a final answer
    """

    # ---- Standard CoT Prompt ----
    STANDARD_COT = (
        "Think step by step before answering.\n\n"
        "Step 1 - Object Enumeration: List every object you can clearly see "
        "in the image. For each object, note its approximate position.\n\n"
        "Step 2 - Attribute Check: For each object listed, describe its "
        "visible attributes (color, size, state) — only what you can confirm.\n\n"
        "Step 3 - Spatial Reasoning: Describe the spatial relationships "
        "between the objects (e.g., 'the cup is to the left of the plate').\n\n"
        "Step 4 - Final Answer: Based ONLY on what you confirmed in the steps "
        "above, provide your final answer. Do not add any objects or details "
        "that were not mentioned in the previous steps.\n\n"
    )

    # ---- Grounded CoT (uses detector output) ----
    GROUNDED_COT_TEMPLATE = (
        "An object detector found the following in this image:\n"
        "{scene_summary}\n\n"
        "Now think step by step:\n\n"
        "Step 1 - Verify Detections: For each detected object, confirm whether "
        "you can see it in the image. Mark any you cannot clearly see.\n\n"
        "Step 2 - Check for Missing Objects: Are there any clearly visible "
        "objects the detector may have missed? Only add objects you are very "
        "confident about.\n\n"
        "Step 3 - Attribute & Spatial Analysis: Describe attributes and spatial "
        "relationships ONLY for confirmed objects.\n\n"
        "Step 4 - Final Answer: Provide your answer based strictly on the "
        "confirmed observations from the steps above.\n\n"
    )

    # ---- Counting-Focused CoT ----
    COUNTING_COT = (
        "To answer this question, count carefully:\n\n"
        "Step 1: Scan the image systematically from left to right.\n"
        "Step 2: For each instance of the target object, note its position.\n"
        "Step 3: Double-check your count by scanning again.\n"
        "Step 4: Report your final count with confidence.\n\n"
    )

    # ---- Safety-Aware CoT ----
    SAFETY_COT = (
        "Before answering, assess the scene for safety:\n\n"
        "Step 1 - Object Identification: List all objects in the scene.\n"
        "Step 2 - Hazard Assessment: Identify any potentially hazardous "
        "objects or situations (sharp objects, hot surfaces, fragile items, "
        "spills, obstacles).\n"
        "Step 3 - Safe Action Planning: Consider what actions would be safe "
        "and what should be avoided.\n"
        "Step 4 - Final Response: Provide your answer, incorporating safety "
        "considerations.\n\n"
    )

    def __init__(self, default_mode: str = "grounded"):
        """
        Args:
            default_mode: One of 'standard', 'grounded', 'counting', 'safety'.
        """
        self.default_mode = default_mode

    def build_prompt(
        self,
        question: str,
        scene_summary: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """
        Build a CoT prompt for the given question.

        Args:
            question: The user's question about the image.
            scene_summary: Detection results (required for 'grounded' mode).
            mode: Override default mode.

        Returns:
            Full prompt string with CoT structure.
        """
        mode = mode or self.default_mode

        if mode == "grounded" and scene_summary:
            cot = self.GROUNDED_COT_TEMPLATE.format(scene_summary=scene_summary)
        elif mode == "counting":
            cot = self.COUNTING_COT
        elif mode == "safety":
            cot = self.SAFETY_COT
        else:
            cot = self.STANDARD_COT

        prompt = (
            f"You are a precise and careful visual assistant.\n\n"
            f"{cot}"
            f"Question: {question}\n\n"
            f"Begin your step-by-step analysis:"
        )
        return prompt

    def build_grounded_cot_prompt(
        self,
        question: str,
        scene_summary: str,
        detections: List[Dict],
    ) -> str:
        """
        Build a grounded CoT prompt with detailed detection info.

        This is the full pipeline prompt combining detection + CoT.
        """
        det_list = ", ".join(
            f"{d['category']} ({d['confidence']:.2f})" for d in detections
        )

        prompt = (
            f"You are a meticulous visual assistant that only states facts "
            f"supported by visual evidence.\n\n"
            f"=== DETECTOR OUTPUT ===\n"
            f"{scene_summary}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Think step by step:\n\n"
            f"Step 1 - VERIFY: Check each detected object ({det_list}) against "
            f"what you see in the image. Confirm or reject each.\n\n"
            f"Step 2 - DESCRIBE: For confirmed objects only, list their visible "
            f"attributes (color, size, condition).\n\n"
            f"Step 3 - RELATE: Describe spatial relationships between confirmed "
            f"objects.\n\n"
            f"Step 4 - ANSWER: Based ONLY on Steps 1-3, answer the question below. "
            f"If uncertain about anything, explicitly say 'I am not sure about...'.\n\n"
            f"Question: {question}\n\n"
            f"Step-by-step analysis:"
        )
        return prompt


def extract_cot_steps(response: str) -> Dict[str, str]:
    """
    Parse a CoT response into individual steps.

    Returns:
        dict mapping step names to their content.
    """
    steps = {}
    current_step = None
    current_content = []

    for line in response.split("\n"):
        line_lower = line.lower().strip()

        # Check for step markers
        if any(marker in line_lower for marker in
               ["step 1", "step 2", "step 3", "step 4",
                "verify", "describe", "relate", "answer",
                "enumeration", "attribute", "spatial", "final"]):
            if current_step:
                steps[current_step] = "\n".join(current_content).strip()
            current_step = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_step:
        steps[current_step] = "\n".join(current_content).strip()

    return steps
