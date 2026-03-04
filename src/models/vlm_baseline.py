"""
Baseline VLM (LLaVA) Inference Module.

Provides the baseline Vision-Language Agent without any
hallucination mitigation — used as the control condition.
"""

import torch
from typing import Dict, List, Optional, Union

from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)


class BaselineVLM:
    """
    Baseline LLaVA inference wrapper.

    Loads LLaVA-1.5-7B with optional 4-bit quantization
    and provides simple image-to-text generation.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        load_in_4bit: bool = True,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name

        # Quantization config for efficient inference
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("Model loaded successfully.")

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1,
    ) -> Union[str, List[str]]:
        """
        Generate text response for an image + prompt.

        Args:
            image: PIL Image.
            prompt: Text prompt/question.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            num_return_sequences: Number of outputs (for GRPO sampling).

        Returns:
            Generated text string (or list if num_return_sequences > 1).
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        # Format conversation for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        # Generate
        do_sample = temp > 0
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp if do_sample else None,
            top_p=self.top_p if do_sample else None,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )

        # Decode — remove the input prompt tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        if num_return_sequences == 1:
            return texts[0].strip()
        return [t.strip() for t in texts]

    @torch.no_grad()
    def generate_multiple_candidates(
        self,
        image: Image.Image,
        prompt: str,
        k: int = 4,
        temperature: float = 0.8,
    ) -> List[str]:
        """
        Generate K candidate responses for GRPO training.

        Uses higher temperature to get diverse outputs.
        """
        candidates = []
        for _ in range(k):
            text = self.generate(
                image=image,
                prompt=prompt,
                temperature=temperature,
                num_return_sequences=1,
            )
            candidates.append(text)
        return candidates

    def get_model(self):
        """Return the underlying model for fine-tuning."""
        return self.model

    def get_processor(self):
        """Return the processor/tokenizer."""
        return self.processor
