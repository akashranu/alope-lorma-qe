import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader

import transformers
import transformers.utils as t_utils
import transformers.pytorch_utils as pt_utils
from transformers.generation import stopping_criteria as sc_mod
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_utils import PreTrainedModel as HFPreTrainedModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    LlamaForCausalLM
)

from datasets import Dataset, concatenate_datasets
from transformer_heads.model.model import HeadedModel
from transformer_heads.model.head import MLPHead
from transformer_heads.config import HeadConfig
from transformer_heads.util.helpers import get_model_params
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.constants import model_type_map


if not hasattr(sc_mod, "EosTokenCriteria"):
    from transformers.generation.stopping_criteria import StoppingCriteria
    class EosTokenCriteria(StoppingCriteria):
        def __init__(self, eos_token_id):
            self.eos_token_id = eos_token_id
        def __call__(self, input_ids, scores, **kwargs):
            last_tokens = input_ids[..., -1]
            return bool((last_tokens == self.eos_token_id).any())
    sc_mod.EosTokenCriteria = EosTokenCriteria


try:
    import fsspec.compression as comp
    filesystems = getattr(comp, "filesystems", {})
    for name, fs_cls in filesystems.items():
        if not hasattr(fs_cls, "extensions"):
            lower = name.lower()
            if "bz" in lower: fs_cls.extensions = [".bz2"]
            elif "gz" in lower: fs_cls.extensions = [".gz"]
            elif "xz" in lower: fs_cls.extensions = [".xz"]
            else: fs_cls.extensions = []
except Exception:
    pass

if not hasattr(t_utils, "is_torch_mlu_available"):
    t_utils.is_torch_mlu_available = lambda: False
if not hasattr(pt_utils, "is_torch_greater_or_equal_than_2_3"):
    pt_utils.is_torch_greater_or_equal_than_2_3 = lambda: True
if not hasattr(t_utils, "XLA_FSDPV2_MIN_VERSION"):
    t_utils.XLA_FSDPV2_MIN_VERSION = "0.0.0"


if not getattr(LlamaConfig, "_qe_rope_patched", False):
    LlamaConfig._qe_orig_rope_scaling_validation = getattr(LlamaConfig, "_rope_scaling_validation", None)
    def _patched_rope_validation(self):
        rs = getattr(self, "rope_scaling", None)
        if isinstance(rs, dict):
            rope_type = rs.get("type", rs.get("rope_type", None))
            if rope_type not in ("linear", "dynamic") or len(rs.keys()) != 2:
                factor = rs.get("factor", 1.0)
                self.rope_scaling = {"type": "dynamic", "factor": factor}
                return
        orig = getattr(LlamaConfig, "_qe_orig_rope_scaling_validation", None)
        if orig is not None and orig is not _patched_rope_validation:
            return orig(self)
    LlamaConfig._rope_scaling_validation = _patched_rope_validation
    LlamaConfig._qe_rope_patched = True


if not getattr(HFPreTrainedModel, "_qe_attn_patched", False):
    _orig_check_and_enable_sdpa = HFPreTrainedModel._check_and_enable_sdpa.__func__
    def _patched_check_and_enable_sdpa(cls, config, hard_check_only: bool = False):
        if cls.__name__ == "HeadedModel":
            config._attn_implementation = "eager"
            return config
        return _orig_check_and_enable_sdpa(cls, config, hard_check_only=hard_check_only)
    HFPreTrainedModel._check_and_enable_sdpa = classmethod(_patched_check_and_enable_sdpa)
    HFPreTrainedModel._qe_attn_patched = True

if not getattr(HeadedModel, "_qe_gc_patched", False):
    def _headedmodel_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self, "model") and hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        return None
    def _headedmodel_gradient_checkpointing_disable(self):
        if hasattr(self, "model") and hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
        return None
    HeadedModel.gradient_checkpointing_enable = _headedmodel_gradient_checkpointing_enable
    HeadedModel.gradient_checkpointing_disable = _headedmodel_gradient_checkpointing_disable
    HeadedModel._qe_gc_patched = True

@dataclass
class LoRMAArguments:
    lorma_r: int = field(default=128)
    lorma_alpha: int = field(default=64)
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lorma_mode: str = field(default="pre")
    do_rri: bool = field(default=True)
    lorma_dropout: float = field(default=0.05)

class LormaLinear_plus(nn.Module):
    def __init__(self, original_linear: nn.Linear, lorma_r: int, lorma_alpha: int, 
                 mode: str, do_rri: bool, lorma_dropout: float = 0.0):
        super().__init__()
        assert mode in ["pre", "post"]
        self.mode = mode
        self.original_layer = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.lorma_r = lorma_r
        self.scaling = lorma_alpha / lorma_r
        self.do_rri = do_rri 
        
        self.lora_dropout = nn.Dropout(p=lorma_dropout) if lorma_dropout > 0. else lambda x: x
        
        if self.mode == "pre":
            self.lora_B = nn.Parameter(original_linear.weight.new_zeros(self.out_features, self.lorma_r))
            self.lora_A = nn.Parameter(original_linear.weight.new_zeros(self.lorma_r, self.out_features))
        else:
            self.lora_B = nn.Parameter(original_linear.weight.new_zeros(self.in_features, self.lorma_r))
            self.lora_A = nn.Parameter(original_linear.weight.new_zeros(self.lorma_r, self.in_features))
                    
        if self.do_rri:
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            with torch.no_grad():
                self.lora_A.copy_(torch.linalg.pinv(self.lora_B))
                self.lora_A.data /= math.sqrt(self.scaling)
                self.lora_B.data /= math.sqrt(self.scaling)

    def forward(self, x):
        if self.mode == "pre":
            if self.do_rri:
                adapted_weight = (self.lora_B @ (self.lora_A @ self.original_layer.weight)) * self.scaling + self.original_layer.weight
            else:
                adapted_weight = (self.lora_B @ (self.lora_A @ self.original_layer.weight)) * self.scaling
            output = F.linear(self.lora_dropout(x), adapted_weight, self.original_layer.bias)
        else:
            if self.do_rri:
                adapted_weight = ((self.original_layer.weight @ self.lora_B) @ self.lora_A) * self.scaling + self.original_layer.weight
            else:
                adapted_weight = (self.original_layer.weight @ self.lora_B) @ self.lora_A
            output = F.linear(self.lora_dropout(x), adapted_weight, self.original_layer.bias)
        return output

def get_target_modules_list(model, target_modules):
    target_names = []
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules:
            target_names.append(name)
    return target_names

def apply_lorma(model, lorma_args):
    print(f"Applying LoRMA with rank {lorma_args.lorma_r}, mode {lorma_args.lorma_mode}")
    target_modules_list = get_target_modules_list(model, lorma_args.target_modules)
    
    for target_path in target_modules_list:
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        
        lorma_target = LormaLinear_plus(
            original_linear=target,
            lorma_r=lorma_args.lorma_r,
            lorma_alpha=lorma_args.lorma_alpha,
            mode=lorma_args.lorma_mode,
            do_rri=lorma_args.do_rri,
            lorma_dropout=lorma_args.lorma_dropout
        )
        parent.__setattr__(target_name, lorma_target)
    return model

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PER_DEVICE_TRAIN_BATCH = 32
GRADIENT_CHECKPOINTING = True
GRADIENT_ACCUM_STEPS = 1
TRAIN_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LOGGING_STEPS = 100
MAX_LENGTH = 256
NUM_PROC = max(1, (os.cpu_count() or 2) // 2)
OUTPUT_DIR = "./tourism_domain_lorma_l1_fnl12832_second"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILES = {
    "en-hindi":   "data/train.en-hi.tsv",
    "en-marathi": "data/train.en-ma.tsv",
    "en-telugu":  "data/train.en-te.tsv",
}


model_type_map["meta-llama"] = ("model", LlamaForCausalLM)
model_params = get_model_params(MODEL_NAME)
hidden_size = model_params["hidden_size"]

head_configs = [
    HeadConfig(
        name="mean_regression",
        layer_hook=-1,
        in_size=hidden_size,
        output_activation="linear",
        is_causal_lm=False,
        pred_for_sequence=True,
        loss_fct="mse",
        num_outputs=1,
        is_regression=True,
        loss_weight=1.0,
    ),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LENGTH


def load_and_prepare_datasets(files):
    datasets = []
    for lang_pair, file_path in files.items():
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='warn', quoting=csv.QUOTE_NONE)
        df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
        df = df.dropna(subset=['mean'])
        df = df[["original", "translation", "mean", "source_lang", "target_lang"]]
        datasets.append(Dataset.from_pandas(df, preserve_index=False))
    return datasets

train_datasets = load_and_prepare_datasets(TRAIN_FILES)
train_ds = concatenate_datasets(train_datasets)
print(f"Merged train size: {len(train_ds)}")

PROMPT_TMPL = (
    'Score the following translation from {src} to {tgt} on a scale 0–100. '
    '{src} source: "{source}" {tgt} translation: "{target}" Score: '
)

def processing_fn(examples):
    prompts = [
        PROMPT_TMPL.format(src=sl, tgt=tl, source=s, target=t)
        for s, t, sl, tl in zip(
            examples["original"], examples["translation"], examples["source_lang"], examples["target_lang"]
        )
    ]
    out = tokenizer(prompts, padding=False, truncation=True, max_length=MAX_LENGTH)
    out["mean_regression"] = examples["mean"]
    return out

train_ds = train_ds.map(processing_fn, batched=True, num_proc=NUM_PROC)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "mean_regression"])


base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

lorma_args = LoRMAArguments(
    lorma_r=128,
    lorma_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lorma_mode="post",
    do_rri=True,
    lorma_dropout=0.05
)

base_model = apply_lorma(base_model, lorma_args)

model = HeadedModel(base_model.config, head_configs)
model.model = base_model
model.head_configs = {hc.name: hc for hc in head_configs}
model.adaptive_loss = False

heads = {}
for name, config in model.head_configs.items():
    head = MLPHead.from_head_config(config)
    head.to(base_model.device)
    head.to(base_model.dtype)
    heads[name] = head

model.heads = nn.ModuleDict(heads)
model.lm_head = None # Disable LM head for regression

if GRADIENT_CHECKPOINTING:
    base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

mark_only_lora_as_trainable(model, bias="none")
for name, param in model.named_parameters():
    if "heads" in name:
        param.requires_grad = True

print_trainable_parameters(model)


base_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

def collate(features):
    batch = base_collator(features)
    batch["mean_regression"] = torch.tensor([f["mean_regression"] for f in features], dtype=torch.float)
    return batch

class RegressionTrainer(Trainer):
    """Custom Trainer for correct regression loss with pooling and optimizer grouping"""
    def create_optimizer(self):
        if self.optimizer is None:
            lorma_params, head_params = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad: continue
                if "lora_" in name: lorma_params.append(param)
                elif name.startswith("heads."): head_params.append(param)
                
            
            optimizer_grouped_parameters = []
            if lorma_params: optimizer_grouped_parameters.append({"params": lorma_params, "lr": self.args.learning_rate})
            if head_params: optimizer_grouped_parameters.append({"params": head_params, "lr": self.args.learning_rate})
            
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, 
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        head = outputs.preds_by_head["mean_regression"]
        
        if head.dim() == 3 and head.size(-1) == 1: head = head.squeeze(-1)
        
        if head.dim() == 2:
            mask = inputs["attention_mask"].float()
            yhat = (head * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            yhat = head
            
        device = next(model.parameters()).device
        y = inputs["mean_regression"].float().to(device)
        loss = ((yhat.float() - y) ** 2).mean()
        
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    do_eval=False,
    remove_unused_columns=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    save_total_limit=1,
    bf16=torch.cuda.is_available(),
)

trainer = RegressionTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collate,
)

print("TRAINING HERE")
trainer.train()

save_dir = os.path.join(OUTPUT_DIR, "lorma_model")
trainer.save_model(save_dir)
print(f"✓ Saved LoRMA model to {save_dir}")