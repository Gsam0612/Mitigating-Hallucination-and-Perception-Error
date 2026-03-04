"""
GRPO (Group Relative Policy Optimization) Trainer.

Implements the GRPO training loop for fine-tuning the VLM
with hallucination-aware rewards. GRPO does NOT require a
separate critic/value network — it uses group-relative advantages.

Key Steps:
1. Sample K candidate responses per input
2. Score each with hallucination-aware reward
3. Compute group-relative advantages
4. Update policy using clipped objective (like PPO)
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    num_candidates: int = 4          # K candidates per input
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    kl_coeff: float = 0.05          # KL penalty coefficient
    clip_range: float = 0.2         # Clipping range for policy ratio
    temperature: float = 0.8        # Sampling temperature for candidates
    max_new_tokens: int = 512
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "./results/grpo"
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Trains VLM to prefer outputs with fewer hallucinations
    using group-relative advantage estimation.
    """

    def __init__(
        self,
        model,
        processor,
        reward_fn,
        config: GRPOConfig,
        device: str = "cuda",
    ):
        """
        Args:
            model: The VLM model (e.g., LLaVA).
            processor: The processor/tokenizer.
            reward_fn: HallucinationReward instance.
            config: GRPOConfig.
            device: Device string.
        """
        self.config = config
        self.reward_fn = reward_fn
        self.processor = processor
        self.device = device

        # Prepare model for LoRA fine-tuning
        self.model = self._setup_lora(model)

        # Store reference model for KL penalty (frozen copy)
        self.ref_model = model  # The original frozen model

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Training state
        self.global_step = 0
        self.train_logs: List[Dict] = []

    def _setup_lora(self, model):
        """Apply LoRA adapters for parameter-efficient fine-tuning."""
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

        # Fix: explicitly set use_reentrant=False to suppress PyTorch warning
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Disable use_cache (incompatible with gradient checkpointing)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def _compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding tokens
        mask = (shift_labels != -100).float()
        sequence_log_probs = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return sequence_log_probs

    def _sample_candidates(
        self,
        image,
        prompt: str,
        k: int,
    ) -> List[Dict]:
        """
        Sample K candidate responses from current policy.

        Returns list of dicts with 'text', 'input_ids', 'labels'.
        """
        candidates = []

        for _ in range(k):
            # Process input
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

            # Generate with sampling
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                )

            # Decode
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            candidates.append({
                "text": text,
                "full_ids": output_ids,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            })

        return candidates

    def _compute_group_advantages(
        self,
        rewards: List[float],
    ) -> List[float]:
        """
        Compute group-relative advantages.

        GRPO key insight: advantages are computed WITHIN the group
        of K candidates, not against an external baseline.

        advantage_i = (reward_i - mean(rewards)) / std(rewards)
        """
        if len(rewards) <= 1:
            return [0.0]

        mean_r = sum(rewards) / len(rewards)
        std_r = (
            sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        ) ** 0.5

        if std_r < 1e-8:
            return [0.0] * len(rewards)

        advantages = [(r - mean_r) / std_r for r in rewards]
        return advantages

    def train_step(
        self,
        image,
        prompt: str,
        gt_objects: set,
        gt_counts: Dict = None,
        gt_spatial: List = None,
    ) -> Dict:
        """
        Single GRPO training step.

        1. Sample K candidates
        2. Score each with reward function
        3. Compute group-relative advantages
        4. Update policy

        Returns:
            dict with training metrics.
        """
        self.model.train()
        k = self.config.num_candidates

        # Step 1: Sample candidates
        candidates = self._sample_candidates(image, prompt, k)

        # Step 2: Score with reward function
        rewards = []
        reward_details = []
        for cand in candidates:
            r = self.reward_fn.compute_reward(
                response=cand["text"],
                gt_objects=gt_objects,
                gt_counts=gt_counts,
                gt_spatial=gt_spatial,
            )
            rewards.append(r["total_reward"])
            reward_details.append(r)

        # Step 3: Compute group-relative advantages
        advantages = self._compute_group_advantages(rewards)

        # Step 4: Compute policy loss
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for i, (cand, adv) in enumerate(zip(candidates, advantages)):
            if abs(adv) < 1e-8:
                continue

            # Compute log prob under current policy
            full_ids = cand["full_ids"].to(self.device)
            input_ids = cand["input_ids"]
            attn_mask = torch.ones_like(full_ids)

            # Create labels (mask input tokens)
            labels = full_ids.clone()
            labels[:, :input_ids.shape[1]] = -100

            log_prob = self._compute_log_probs(
                self.model, full_ids, attn_mask, labels
            )

            # Compute log prob under reference policy (frozen)
            with torch.no_grad():
                ref_log_prob = self._compute_log_probs(
                    self.ref_model, full_ids, attn_mask, labels
                )

            # Policy ratio
            ratio = torch.exp(log_prob - ref_log_prob)

            # Clipped objective (PPO-style)
            adv_tensor = torch.tensor(adv, device=self.device)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_range,
                1 + self.config.clip_range,
            ) * adv_tensor

            # GRPO loss = negative of min(surr1, surr2) — we maximize
            policy_loss = -torch.min(surr1, surr2)

            # KL penalty
            kl = (ref_log_prob - log_prob).mean()
            kl_penalty = self.config.kl_coeff * kl

            total_loss = total_loss + policy_loss + kl_penalty

        # Normalize by number of candidates
        total_loss = total_loss / k

        # Backprop
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # Optimizer step (with gradient accumulation)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        # Log metrics
        metrics = {
            "step": self.global_step,
            "loss": total_loss.item(),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "reward_std": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards)/len(rewards))**0.5,
            "best_candidate": candidates[rewards.index(max(rewards))]["text"][:200],
            "num_hallucinated_avg": sum(
                len(rd["hallucinated_objects"]) for rd in reward_details
            ) / len(reward_details),
        }

        self.train_logs.append(metrics)
        return metrics

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        grounded_vlm=None,
    ):
        """
        Full GRPO training loop.

        Args:
            train_dataset: COCOHallucinationDataset.
            eval_dataset: Optional evaluation dataset.
            grounded_vlm: GroundedVLM for building prompts.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            epoch_rewards = []

            for idx in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}"):
                sample = train_dataset[idx]
                image = sample["image"]
                question = sample["question"]
                gt = sample["ground_truth"]

                # Build grounded prompt if available
                if grounded_vlm:
                    result = grounded_vlm.generate(image, question)
                    prompt = result["prompt"]
                else:
                    prompt = question

                # Train step
                metrics = self.train_step(
                    image=image,
                    prompt=prompt,
                    gt_objects=gt["unique_objects"],
                    gt_counts=gt["object_counts"],
                    gt_spatial=sample.get("spatial_relations"),
                )

                epoch_rewards.append(metrics["mean_reward"])

                # Log periodically
                if (self.global_step % self.config.eval_steps) == 0:
                    avg_reward = sum(epoch_rewards[-50:]) / len(epoch_rewards[-50:])
                    print(f"\n  Step {self.global_step}: "
                          f"avg_reward={avg_reward:.3f}, "
                          f"halluc_avg={metrics['num_hallucinated_avg']:.1f}")

                # Save checkpoint
                if (self.global_step % self.config.save_steps) == 0:
                    self._save_checkpoint(epoch, self.global_step)

            # End-of-epoch evaluation
            if eval_dataset:
                eval_metrics = self.evaluate(eval_dataset, grounded_vlm)
                print(f"\nEpoch {epoch+1} Eval: {eval_metrics}")

        # Save final model
        self._save_checkpoint(self.config.num_epochs, self.global_step, final=True)
        self._save_training_logs()

        print("\nTraining complete!")
        return self.train_logs

    def evaluate(self, eval_dataset, grounded_vlm=None) -> Dict:
        """Run evaluation on held-out data."""
        self.model.eval()
        all_rewards = []
        total_hallucinated = 0
        total_correct = 0
        total_mentioned = 0

        with torch.no_grad():
            for idx in tqdm(range(min(len(eval_dataset), 100)), desc="Evaluating"):
                sample = eval_dataset[idx]
                image = sample["image"]
                question = sample["question"]
                gt = sample["ground_truth"]

                if grounded_vlm:
                    result = grounded_vlm.generate(image, question)
                    response = result["response"]
                else:
                    response = "No response"

                reward = self.reward_fn.compute_reward(
                    response=response,
                    gt_objects=gt["unique_objects"],
                    gt_counts=gt["object_counts"],
                )

                all_rewards.append(reward["total_reward"])
                total_hallucinated += len(reward["hallucinated_objects"])
                total_correct += len(reward["correct_objects"])
                total_mentioned += len(reward["mentioned_objects"])

        n = len(all_rewards)
        return {
            "mean_reward": sum(all_rewards) / n if n > 0 else 0,
            "avg_hallucinated": total_hallucinated / n if n > 0 else 0,
            "avg_correct": total_correct / n if n > 0 else 0,
            "precision": total_correct / total_mentioned if total_mentioned > 0 else 0,
            "num_samples": n,
        }

    def _save_checkpoint(self, epoch: int, step: int, final: bool = False):
        """Save model checkpoint."""
        tag = "final" if final else f"epoch{epoch}_step{step}"
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{tag}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"  Checkpoint saved: {save_path}")

    def _save_training_logs(self):
        """Save training logs to JSON."""
        log_path = os.path.join(self.config.output_dir, "training_logs.json")
        with open(log_path, "w") as f:
            json.dump(self.train_logs, f, indent=2, default=str)
        print(f"Training logs saved: {log_path}")
