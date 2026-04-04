# Domain-Specific Quality Estimation with LoRMA-Extended ALOPE

Implementation of training and inference code for the **ALOPE** framework extended with **LoRMA** (Low-Rank Multiplicative Adaptation), used for domain-specific Quality Estimation (QE) in low-resource English→Indic machine translation scenarios.

This repository accompanies the paper:

> **Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios**
> Namrata Patil Gurav, Akashdeep Ranu, Archchana Sindhujan, Diptesh Kanojia
> *EACL 2026 LoResLM Workshop*
> [arXiv:2603.07372](https://arxiv.org/abs/2603.07372) · [Official ALOPE Repo](https://github.com/surrey-nlp/ALOPE)

---

## Overview

Quality Estimation (QE) predicts the quality of machine translation output without requiring reference translations, making it especially valuable in low-resource and domain-sensitive settings. This codebase extends the ALOPE framework, which attaches lightweight regression heads to intermediate Transformer layers of an open-weight LLM, by introducing LoRMA adapters as an alternative to the original LoRA-based fine-tuning.

Unlike standard LoRA, which additively injects low-rank weight updates, **LoRMA modulates existing weights multiplicatively**, providing a different regularisation dynamic during fine-tuning. A key finding of the paper is that intermediate Transformer layers (particularly layers −9 and −11) consistently produce stronger cross-lingual QE signals than the final layer, even across typologically distant language pairs such as English→Tamil and English→Gujarati.

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

