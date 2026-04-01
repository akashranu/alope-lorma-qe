import csv
import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, LlamaForCausalLM

import transformers.utils as t_utils
import transformers.pytorch_utils as pt_utils
from transformers.generation import stopping_criteria as sc_mod
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_utils import PreTrainedModel as HFPreTrainedModel

from transformer_heads.model.model import HeadedModel
from transformer_heads.model.head import MLPHead
from transformer_heads.config import HeadConfig
from transformer_heads.util.helpers import get_model_params
from transformer_heads.constants import model_type_map
from safetensors.torch import load_file


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


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TRAINED_MODEL_PATH = "./tourism_domain_lorma_l1_fnl12832_second/lorma_model"
OUT_DIR = "./tourism_infer_results_lorma_l1_fnl12832_second"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SETS = {
    "en-hindi":   "data/test.en-hi.tsv",
    "en-marathi": "data/test.en-ma.tsv",
    "en-telugu":  "data/test.en-te.tsv",
}

BATCH_SIZE = 8
MAX_LENGTH = 256


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

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=None,
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

print("Injecting LoRMA adapters...")
base_model = apply_lorma(base_model, lorma_args)

print("Wrapping with HeadedModel...")
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
model.lm_head = None


print(f"Loading weights from {TRAINED_MODEL_PATH}...")
state_dict = {}
index_file = os.path.join(TRAINED_MODEL_PATH, "model.safetensors.index.json")
single_file = os.path.join(TRAINED_MODEL_PATH, "model.safetensors")
bin_file = os.path.join(TRAINED_MODEL_PATH, "pytorch_model.bin")

if os.path.exists(index_file):
    with open(index_file, "r") as f:
        index = json.load(f)
    for shard in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(os.path.join(TRAINED_MODEL_PATH, shard)))
elif os.path.exists(single_file):
    state_dict = load_file(single_file)
elif os.path.exists(bin_file):
    state_dict = torch.load(bin_file, map_location="cpu")
else:
    raise FileNotFoundError(f"No model weights found in {TRAINED_MODEL_PATH}")

print("Applying state dict to model...")
model.load_state_dict(state_dict, strict=False)
model.eval()


print(f"Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LENGTH

def lp_to_names(lp):
    m = {"en":"English","hindi":"Hindi","marathi":"Marathi","gujarati":"Gujarati","tamil":"Tamil","telugu":"Telugu"}
    src, tgt = lp.split("-")
    return (m.get(src, src), m.get(tgt, tgt))

PROMPT_TMPL = (
    'Score the following translation from {src} to {tgt} on a scale 0–100. '
    '{src} source: "{source}" {tgt} translation: "{target}" Score: '
)

def load_parquet_as_hf_dataset(url, fallback_src, fallback_tgt):
    print(f"Loading Test File: {url}")
    df = pd.read_csv(url, sep='\t', on_bad_lines='warn', quoting=csv.QUOTE_NONE)
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
    df = df.dropna(subset=['mean'])
    if "source_lang" not in df.columns: df["source_lang"] = fallback_src
    if "target_lang" not in df.columns: df["target_lang"] = fallback_tgt
    return Dataset.from_pandas(df, preserve_index=False)

def build_processing_fn(tok):
    def processing_fn(examples):
        prompts = [
            PROMPT_TMPL.format(src=sl, tgt=tl, source=s, target=t)
            for s, t, sl, tl in zip(
                examples["original"], examples["translation"], examples["source_lang"], examples["target_lang"]
            )
        ]
        out = tok(prompts, padding=False, truncation=True, max_length=MAX_LENGTH)
        out["mean_regression"] = examples["mean"]
        return out
    return processing_fn

base_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

def collate(features):
    batch = base_collator(features)
    labels = torch.tensor([f["mean_regression"] for f in features], dtype=torch.float)
    return batch, labels

def pearsonr_np(x, y):
    if len(x) < 2: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def try_spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y)[0])
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return pearsonr_np(rx, ry)

all_rows = []

for lp, url in TEST_SETS.items():
    src_name, tgt_name = lp_to_names(lp)
    raw = load_parquet_as_hf_dataset(url, src_name, tgt_name)
    processing_fn = build_processing_fn(tokenizer)
    ds = raw.map(processing_fn, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","mean_regression"])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    preds, gts = [], []
    with torch.no_grad():
        for (inputs, labels) in loader:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            
            head = out.preds_by_head["mean_regression"] if hasattr(out, "preds_by_head") else out["mean_regression"]
            if head.ndim == 3 and head.size(-1) == 1: head = head.squeeze(-1)
            
            if head.ndim == 2:
                mask = inputs["attention_mask"].float()
                yhat = (head * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                yhat = head
            
            preds.extend(yhat.detach().cpu().float().numpy().reshape(-1).tolist())
            gts.extend(labels.cpu().float().numpy().reshape(-1).tolist())

    preds = np.array(preds, dtype=float)
    gts = np.array(gts, dtype=float)
    pear = pearsonr_np(preds, gts)
    spea = try_spearman(preds, gts)
    
    all_rows.append({"language": lp, "pearson": pear, "spearman": spea})
    
    per_lp_csv = os.path.join(OUT_DIR, f"{lp}_preds.csv")
    pd.DataFrame({"ground_truth": gts, "prediction": preds}).to_csv(per_lp_csv, index=False)
    print(f"[{lp}] N={len(gts)}  pearson={pear:.6f}  spearman={spea:.6f}  -> {per_lp_csv}")


summary_df = pd.DataFrame(all_rows, columns=["language","pearson","spearman"])
print("\nCorrelation Summary:")
print(summary_df.to_string(index=False))

summary_csv = os.path.join(OUT_DIR, "correlation_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary to: {summary_csv}")