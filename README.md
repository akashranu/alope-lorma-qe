# Domain-Specific Quality Estimation with LoRMA-Extended ALOPE

Implementation of training and inference code for the **ALOPE** framework extended with **LoRMA** (Low-Rank Multiplicative Adaptation), used for domain-specific Quality Estimation (QE) in low-resource English→Indic machine translation scenarios.

This repository accompanies the paper:

> **Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios**
> Namrata Patil Gurav, Akashdeep Ranu, Archchana Sindhujan, Diptesh Kanojia
> *EACL 2026 LoResLM Workshop*
> [arXiv:2603.07372](https://arxiv.org/abs/2603.07372) · [Official ALOPE Repo](https://github.com/surrey-nlp/ALOPE)

---

## Overview

Quality Estimation (QE) predicts the quality of machine translation output without requiring reference translations, making it especially valuable in low-resource and domain-sensitive settings. This codebase extends the ALOPE framework — which attaches lightweight regression heads to intermediate Transformer layers of an open-weight LLM — by introducing LoRMA adapters as an alternative to the original LoRA-based fine-tuning.

Unlike standard LoRA, which additively injects low-rank weight updates, **LoRMA modulates existing weights multiplicatively**, providing a different regularisation dynamic during fine-tuning. A key finding of the paper is that intermediate Transformer layers (particularly layers −9 and −11) consistently produce stronger cross-lingual QE signals than the final layer, even across typologically distant language pairs such as English→Tamil and English→Gujarati.

---

## Repository Structure

```
├── train.py                  # LoRMA-based ALOPE training script
├── inference.py              # Inference and correlation evaluation script
├── data/                     # TSV datasets (train/test splits per domain and language pair)
│   ├── train.en-hi.tsv
│   ├── train.en-ma.tsv
│   ├── train.en-te.tsv
│   ├── test.en-hi.tsv
│   ├── test.en-ma.tsv
│   └── test.en-te.tsv
└── README.md
```

---

## Key Components

### LoRMA Implementation
A custom `LormaLinear_plus` module replaces target linear layers with a multiplicative low-rank adaptation. Unlike additive LoRA updates, LoRMA modulates the original weight matrix directly:

- **Pre-mode:** `W' = (B @ (A @ W)) * scaling + W`
- **Post-mode:** `W' = ((W @ B) @ A) * scaling + W`

Supports RRI (Random Right Inverse) initialisation for stable training, configurable rank, scaling factor, dropout, and target module selection.

### Custom Regression Trainer
Extends HuggingFace's `Trainer` class to handle:
- **Weighted mean pooling** over attention masks for sequence-level regression
- **Separate parameter groups** for LoRMA adapters, regression heads, and base model parameters
- **MSE loss** against human Direct Assessment (DA) scores

### Layer-wise Evaluation
Regression heads are attached to selected intermediate Transformer layers (`−1, −7, −9, −11`) to evaluate which layer representations produce the strongest QE correlation signals. Results are reported using both Spearman's rank correlation (ρ) and Pearson's correlation (r).

---

## Installation

```bash
pip install torch transformers datasets pandas scipy safetensors
pip install transformer-heads
```

A GPU with at least 16GB VRAM is recommended. The training script uses `bfloat16` precision automatically when CUDA is available.

---

## Data Format

Data files should be TSV format with the following columns:

| Column | Description |
|--------|-------------|
| `original` | Source sentence (English) |
| `translation` | Machine-translated output (Indic language) |
| `mean` | Human Direct Assessment score (0–100) |
| `source_lang` | Source language name (e.g., `English`) |
| `target_lang` | Target language name (e.g., `Hindi`) |

---

## Usage

### Training

Configure paths and hyperparameters at the top of `train.py`:

```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./output_dir"
TRAIN_FILES = {
    "en-hindi":   "data/train.en-hi.tsv",
    "en-marathi": "data/train.en-ma.tsv",
    "en-telugu":  "data/train.en-te.tsv",
}
```

Then run:

```bash
python train.py
```

Key LoRMA configuration:

```python
lorma_args = LoRMAArguments(
    lorma_r=128,
    lorma_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lorma_mode="post",
    do_rri=True,
    lorma_dropout=0.05
)
```

### Inference

Update `TRAINED_MODEL_PATH` and `TEST_SETS` in `inference.py`, then run:

```bash
python inference.py
```

This will output per-language-pair Spearman and Pearson correlation scores, save per-sample predictions as CSV files, and write a summary correlation table.

---

## Results

ALOPE with LoRMA achieves competitive QE performance against LoRA across most domains and language pairs, with intermediate Transformer layers consistently outperforming final-layer representations. LoRMA introduces a stabilising effect across layers, reducing variance between adjacent layers and mitigating low correlations at shallow layers — at a small cost to peak performance in some domains compared to LoRA.

| Domain | Best Avg. Spearman (LoRMA) | Best Avg. Spearman (LoRA) |
|--------|---------------------------|--------------------------|
| General | 0.329 | 0.388 |
| Healthcare | 0.329 | 0.415 |
| Legal | 0.280 | 0.431 |
| Tourism | 0.404 | 0.404 |

Full results and layer-wise analysis are reported in the paper.

---

## System Patches

The codebase includes six compatibility patches applied at runtime to resolve issues between `transformer-heads`, `LLaMA 3.2`, and the HuggingFace `transformers` library:

1. Missing `EosTokenCriteria` in older `transformers` versions
2. `fsspec` compression extension attributes
3. Hardware availability checks (`MLU`, `XLA`)
4. LLaMA 3 `rope_scaling` validation
5. SDPA attention forced to `eager` mode for `HeadedModel`
6. Gradient checkpointing support for `HeadedModel`

---

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{gurav2026domain,
  title     = {Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios},
  author    = {Gurav, Namrata Patil and Ranu, Akashdeep and Sindhujan, Archchana and Kanojia, Diptesh},
  booktitle = {Proceedings of the EACL 2026 LoResLM Workshop},
  year      = {2026}
}
```

---

## Acknowledgements

This work was conducted at the [Surrey Institute for People-Centred AI](https://www.surrey.ac.uk/artificial-intelligence), University of Surrey. LoRMA implementation adapted from [LoRMA](https://github.com/AshutoshM10/LoRMA) (Bihany et al., ACL 2025). The ALOPE framework is described in [Sindhujan et al., 2025](https://arxiv.org/abs/2508.07484).
