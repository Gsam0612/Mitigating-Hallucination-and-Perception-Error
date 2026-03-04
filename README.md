# Mitigating Hallucination and Perception Errors in Vision Language Agents

**MSc Artificial Intelligence Dissertation — Heriot-Watt University**

**Author:** Vedhagiri Alagesan | **Supervisor:** Dr. Oliver Lemon

## Overview

This project implements a training-time reinforcement learning approach to reduce hallucinations in Vision-Language Agents (VLAs). It combines:

1. **Object Detection Grounding** — YOLOv8 detections injected into VLM prompts
2. **Chain-of-Thought Reasoning** — Step-by-step visual reasoning prompts
3. **Self-Verification** — Automated hallucination checking and correction
4. **GRPO Fine-Tuning** — Group Relative Policy Optimization with hallucination-aware rewards

## Project Structure

```
├── config/
│   └── config.yaml                  # All hyperparameters and paths
├── src/
│   ├── data/
│   │   ├── coco_loader.py           # COCO dataset loading + ground truth
│   │   └── pope_loader.py           # POPE benchmark evaluation
│   ├── models/
│   │   ├── vlm_baseline.py          # LLaVA-1.5-7B baseline inference
│   │   ├── yolo_detector.py         # YOLOv8 object detection
│   │   └── grounded_vlm.py          # Detection-grounded VLM pipeline
│   ├── mitigation/
│   │   ├── cot_prompting.py         # Chain-of-thought prompt templates
│   │   └── self_verification.py     # Self-verification module
│   ├── training/
│   │   ├── reward_function.py       # Hallucination-aware reward computation
│   │   └── grpo_trainer.py          # GRPO training loop
│   └── evaluation/
│       ├── hallucination_metrics.py  # All hallucination metrics
│       ├── ablation.py              # Ablation study runner
│       └── visualization.py         # Results charts and tables
├── notebooks/
│   └── VLA_Hallucination_Mitigation.ipynb  # Main Colab notebook (run this!)
├── results/                         # Generated during execution
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Upload this repo to GitHub
2. Open `notebooks/VLA_Hallucination_Mitigation.ipynb` in Colab
3. Select **A100 GPU** runtime
4. Update the `REPO_URL` in the first cell
5. Run all cells sequentially

### Option 2: Local Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/VLA_Hallucination_Mitigation.ipynb
```

## Datasets

| Dataset | Purpose | Download |
|---------|---------|----------|
| COCO 2017 Val | Ground-truth objects, attributes, spatial relations | Auto-downloaded in notebook |
| POPE | Object-existence hallucination benchmark (random/popular/adversarial) | Auto-downloaded in notebook |

## Ablation Configurations

| Config | Detection | CoT | Self-Verify | GRPO |
|--------|-----------|-----|-------------|------|
| Baseline | ✗ | ✗ | ✗ | ✗ |
| +Detection | ✓ | ✗ | ✗ | ✗ |
| +Detection+CoT | ✓ | ✓ | ✗ | ✗ |
| +Det+CoT+Verify | ✓ | ✓ | ✓ | ✗ |
| Full (no GRPO) | ✓ | ✓ | ✓ | ✗ |
| Full+GRPO | ✓ | ✓ | ✓ | ✓ |

## Metrics

- **Object Precision/Recall/F1** — Correctness of mentioned objects
- **Hallucination Rate** — % of outputs containing hallucinated objects
- **Count Accuracy** — Correctness of object counting
- **Spatial Accuracy** — Correctness of spatial relation claims
- **POPE Accuracy/F1** — Standard hallucination benchmark scores
- **Composite Score** — Weighted combination of all metrics

## Hardware Requirements

- **GPU:** A100 (40GB+) recommended for GRPO training
- **RAM:** 16GB+ system memory
- **Storage:** ~20GB for COCO dataset + model weights

## Key References

- [LLaVA-1.5](https://arxiv.org/abs/2310.03744) — Vision-Language Model
- [YOLOv8](https://docs.ultralytics.com/) — Object Detection
- [POPE](https://arxiv.org/abs/2305.10355) — Hallucination Benchmark
- [GRPO](https://yugeten.github.io/posts/2025/01/ppogrpo/) — Policy Optimization

## License

This project is for academic purposes as part of an MSc dissertation at Heriot-Watt University.
