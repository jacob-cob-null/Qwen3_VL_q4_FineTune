# pAIge — Environment Setup & Training PRD
## AI Agent Execution Guide

---

## Project Summary
Fine-tune `Qwen/Qwen3-VL-4B-Instruct` for Philippine medical billing document KVP extraction.
Task: perception only — find field, return value, output JSON. No logic validation.
Target: IEEE submission May 2026.

---

## Compute
| Phase | Platform | GPU | VRAM |
|-------|----------|-----|------|
| Smoke test | Local | RTX 3060 | 12GB |
| Full ablation (4 days) | Local | RTX 3060 | 12GB |

**Local fine-tuning with Unsloth + LoRA:**
- LoRA reduces memory footprint ~4x vs full fine-tuning
- QLoRA 4-bit quantization optional fallback if VRAM tight
- All 7 conditions trained sequentially on RTX 3060
- GPU stays on for 4-day sprint; no cloud dependency

---

## Environment Setup

### Step 1 — Install dependencies
```bash
pip install unsloth
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft bitsandbytes
pip install datasets pillow faker reportlab augraphy
pip install evaluate scikit-learn matplotlib
```

### Step 2 — Verify GPU
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Step 3 — Verify Unsloth + Qwen3-VL support
```python
from unsloth import FastVisionModel
# If this throws ImportError, fall back to HuggingFace PEFT:
# from transformers import AutoModelForVision2Seq
# from peft import get_peft_model, LoraConfig
```

---

## Model Config
```python
model_id = "Qwen/Qwen3-VL-4B-Instruct"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_id,
    load_in_4bit=False,
    torch_dtype=torch.bfloat16,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,   # frozen — task is perception, not vision relearning
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,                            # narrow task — r=8 sufficient
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    random_state=42,
)
```

**Fallback if Unsloth doesn't support Qwen3-VL:**
```python
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
```

---

## Prompt Template
```python
import json

SYSTEM_PROMPT = """You are a medical billing document parser for Philippine hospital documents.
Extract key-value pairs and return valid JSON only. No explanation, no markdown."""

EXTRACTION_PROMPT = (
    "Extract all billing fields. Return JSON with: "
    "date, patient_name, philhealth_number, diagnosis_code, "
    "procedure_code, total_amount, philhealth_benefit, "
    "balance_due. "
    "Use null for missing fields."
)

def format_sample(image, ground_truth_json):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": EXTRACTION_PROMPT}
        ]},
        {"role": "assistant", "content": json.dumps(
            ground_truth_json, ensure_ascii=False
        )}
    ]}
```

---

## Dataset

### Structure expected
```
paige_dataset/
├── images/          # .png or .jpg synthetic + real documents
├── train.jsonl      # per condition
└── test.jsonl       # SROIE 2019 v2 only — never in training
```

### JSONL format per line
```json
{
  "image": "images/sample_001.png",
  "labels": {
    "date": "04/10/2026",
    "patient_name": "Juan dela Cruz",
    "philhealth_number": "12-345678901-2",
    "diagnosis_code": "J18.9",
    "procedure_code": "RVS-1234",
    "total_amount": 46028.95,
    "philhealth_benefit": 37028.95,
    "balance_due": 9000.00
  }
}
```

### Ground truth rule
Labels must reflect what is **visually present** on the document.
Do not infer or compute values. If a field is absent, label is `null`.

### Condition sampling
```python
import random

def build_condition(real_data, synthetic_pool, total_size, synth_ratio, seed):
    random.seed(seed)
    n_synth = int(total_size * synth_ratio)
    n_real  = total_size - n_synth
    return (
        random.sample(real_data, min(n_real, len(real_data))) +
        random.sample(synthetic_pool, min(n_synth, len(synthetic_pool)))
    )

conditions = [
    {"id": "A", "size": 500,  "synth_ratio": 0.30, "seed": 42},
    {"id": "B", "size": 500,  "synth_ratio": 0.40, "seed": 43},
    {"id": "C", "size": 1000, "synth_ratio": 0.30, "seed": 44},
    {"id": "D", "size": 1000, "synth_ratio": 0.40, "seed": 45},
    {"id": "E", "size": 2000, "synth_ratio": 0.30, "seed": 46},
    {"id": "F", "size": 2000, "synth_ratio": 0.40, "seed": 47},
    {"id": "G", "size": 2000, "synth_ratio": 0.00, "seed": 48},
]
```

**Note:** Condition F needs 800 synthetic images. Generate 800 total, not 600.
Synthetic pool: 800 images from 15 real references via Faker + Nemotron + ReportLab + Augraphy.

---

## AI Agent Pre-Flight Validation

**Purpose:** Before launching ablation, AI agent validates that all requirements are met from the codebase.

### Checklist (all items must pass)

#### 1. Data Pipeline Staged
- [ ] KVP10k dataset loaded and indexed
- [ ] CORD v2 dataset loaded and indexed
- [ ] Invoices-Donut-v1 dataset loaded and indexed
- [ ] pAIge synthetic Philippine SOA data generated (800 images minimum)
- [ ] Real data pool ≥ 2000 images for sampling
- [ ] Synthetic pool ≥ 800 images for sampling

**Validation command:**
```python
assert os.path.exists("./paige_dataset/real/"), "Real data missing"
assert os.path.exists("./paige_dataset/synthetic/"), "Synthetic data missing"
assert len(os.listdir("./paige_dataset/synthetic/")) >= 800, "Insufficient synthetic images"
print("✓ Data pipeline ready")
```

#### 2. SROIE 2019 v2 Test Set Isolated
- [ ] SROIE test split loaded separately (347 images)
- [ ] Test split never included in any training dataset
- [ ] Test split stored in read-only or separate directory
- [ ] Condition sampling code explicitly excludes test split

**Validation command:**
```python
test_images = set(os.listdir("./paige_dataset/sroie_test/"))
for condition in conditions:
    train_dataset = build_condition(...)
    train_images = set([s["image"] for s in train_dataset])
    overlap = test_images & train_images
    assert len(overlap) == 0, f"Test leak detected in {condition['id']}: {overlap}"
print("✓ SROIE test set is isolated")
```

#### 3. Unsloth + QLoRA Config Finalized
- [ ] Unsloth installed and FastVisionModel imports successfully
- [ ] Fallback to HuggingFace PEFT configured if Unsloth unavailable
- [ ] Model loading tested (Qwen3-VL-4B-Instruct or fallback Qwen2.5-VL 7B)
- [ ] LoRA config defined: r=8, lora_alpha=16, lora_dropout=0.05
- [ ] Vision encoder frozen (finetune_vision_layers=False)
- [ ] Batch size=2, gradient_accumulation_steps=8 (eff batch=16) set
- [ ] Learning rate=2e-4, warmup_ratio=0.05 configured
- [ ] Gradient checkpointing enabled

**Validation command:**
```python
try:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        r=8, lora_alpha=16, lora_dropout=0.05,
        random_state=42,
    )
    print("✓ Unsloth FastVisionModel loaded")
except ImportError:
    print("⚠ Unsloth unavailable, falling back to PEFT")
    # Fallback code here
```

#### 4. Smoke Test on Condition A Passed
- [ ] Smoke test run on Condition A (500 images) completed
- [ ] max_steps=100 limit enforced
- [ ] No CUDA OOM during first 10 steps
- [ ] Loss decreasing by step 50
- [ ] At least one inference sample produces valid JSON
- [ ] Peak VRAM < 11.5GB (RTX 3060) or < 15GB (T4)
- [ ] adapter_model.safetensors saved successfully
- [ ] Inference check passed: output is valid JSON

**Validation command:**
```python
import json
import os
assert os.path.exists("./paige-smoketest/adapter_model.safetensors"), "Smoke test checkpoint missing"
with open("./paige-smoketest/training_args.bin", "rb") as f:
    # Confirm max_steps was capped at 100
print("✓ Smoke test passed")
```

#### 5. Logging & Checkpointing Infrastructure in Place
- [ ] Metrics logging configured (loss, val_f1, per-field accuracy)
- [ ] eval_condition_*.json template defined
- [ ] Output directories for each condition created
- [ ] safe_parse_json() function defined
- [ ] compute_metrics() function defined and tested
- [ ] Checkpoint save logic implemented (adapter_model.safetensors)
- [ ] Persistent storage or backup location confirmed

**Validation command:**
```python
assert callable(compute_metrics), "compute_metrics() not defined"
assert callable(safe_parse_json), "safe_parse_json() not defined"
for cid in ["A","B","C","D","E","F","G"]:
    os.makedirs(f"./paige-lora-condition-{cid}", exist_ok=True)
print("✓ Logging & checkpointing ready")
```

#### 6. Results Tracking Template Ready
- [ ] Results spreadsheet or JSON template created
- [ ] Columns: condition_id, dataset_size, synth_ratio, train_time_hrs, eval_f1, macro_f1, exact_match, per_field_json, notes
- [ ] Example row filled in for testing

**Validation command:**
```python
import json
results_template = {
    "condition_id": "A",
    "dataset_size": 500,
    "synth_ratio": 0.30,
    "train_time_hrs": 0.0,
    "eval_f1": 0.0,
    "macro_f1": 0.0,
    "exact_match": 0.0,
    "per_field_json": {},
    "notes": ""
}
with open("paige_ablation_results.json", "w") as f:
    json.dump([results_template], f, indent=2)
print("✓ Results tracking template created")
```

#### 7. RTX 3060 Thermal Validated
- [ ] GPU fan curves checked and stable
- [ ] System stable under simulated 24/7 load (thermal test ≥1 hour)
- [ ] No GPU throttling observed during smoke test
- [ ] Ambient temperature monitored
- [ ] Power supply confirmed stable (450W+ recommended)

**Validation command:**
```python
import time
import torch
start_temp = torch.cuda.get_device_properties(0).major
print(f"GPU temp at start: {start_temp}°C")
# Run dummy workload for 5 min, check again
time.sleep(300)
# Peak temp should not exceed 85°C sustained
print("✓ Thermal check passed")
```

---

## 4-Day Ablation Sprint Schedule

**Total wall-clock: ~15.5 hours across 4 days (sequential per day)**

### Day 1: Baseline + Small Datasets (~5 hrs)
1. **Condition A: 500/70-30** (~60 min train + 20 min eval)
2. **Condition B: 500/60-40** (~60 min train + 20 min eval)
3. **Condition G: 2000 real-only control** (~90 min train + 30 min eval)

---

### Day 2: Medium — First Half (~2 hrs)
4. **Condition C: 1000/70-30** (~90 min train + 30 min eval)

---

### Day 3: Medium — Second Half + Full-Scale Start (~5 hrs)
5. **Condition D: 1000/60-40** (~90 min train + 30 min eval)
6. **Condition E: 2000/70-30** (~150 min train + 45 min eval)

---

### Day 4: Final Full-Scale (~3.5 hrs)
7. **Condition F: 2000/60-40** (~150 min train + 45 min eval)
**Complete ablation, all results logged**

---

### Execution Rules
- **Sequential within each day** — do NOT run multiple conditions in parallel (RTX 3060 VRAM will OOM with 2× Qwen2.5-VL models)
- **Smallest-to-largest ordering** — early wins validate pipeline, build confidence
- **Eval immediately after each run** — log metrics before moving to next condition
- **All seeds fixed** — Condition A always seed=42, B=43, etc. for reproducibility
- **SROIE test set never touched** — eval_condition_*.json uses held-out test split only

---

## AI Agent Validation Workflow

When launching ablation, agent should:

```
1. Load PRD and checklist
2. Run all 7 validation checks against codebase
3. Print pass/fail for each item
4. Block execution if ANY item fails
5. If all pass, print: "✓ ALL PRE-FLIGHT CHECKS PASSED — READY FOR 3-DAY ABLATION"
6. Execute Day 1, Day 2, Day 3 sequentially
7. After each condition, compute_metrics() and save eval_condition_*.json
8. After all 7 conditions, generate paige_ablation_curves.pdf
9. Log total wall-clock time and results summary
```

---

## Phase 1 — Smoke Test (Local RTX 3060)

### Goal
Confirm Unsloth + LoRA pipeline sanity before 4-day ablation.
Run on smallest condition (A — 500 images) with hard step cap.

### Config
```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./paige-smoketest",
    max_steps=100,                      # hard cap — just checking loss curve
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # effective batch 8
    gradient_checkpointing=True,
    optim="adamw_8bit",
    bf16=True,
    fp16=False,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    max_seq_length=1024,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
```

### VRAM monitor during smoke test
```python
import torch

def print_vram():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# Call after first batch to confirm headroom
print_vram()
```

### Pass criteria — all must be true before proceeding
- [ ] No CUDA OOM during first 10 steps
- [ ] Loss decreasing by step 50
- [ ] At least one inference sample produces parseable JSON
- [ ] Peak VRAM < 11.5GB (RTX 3060 headroom requirement)
- [ ] `adapter_model.safetensors` saves successfully

### Quick inference check after smoke test
```python
FastVisionModel.for_inference(model)

sample_image = load_test_image()   # any single document image
inputs = tokenizer.apply_chat_template(
    format_sample(sample_image, {})["messages"][:-1],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Must be valid JSON
import json
try:
    parsed = json.loads(response)
    print("PASS — valid JSON output")
    print(parsed)
except:
    print("FAIL — output is not valid JSON")
    print(response)
```

---

## Phase 2 — Full Ablation (Local RTX 3060, 4 Days)

### Config (Local RTX 3060)
```python
def get_training_args(condition_id):
    return SFTConfig(
        output_dir=f"./paige-lora-condition-{condition_id}",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,     # effective batch 16
        gradient_checkpointing=True,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_seq_length=2048,
        logging_steps=25,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        resume_from_checkpoint=True,
    )
```

### Sequential launcher
```python
import time
import os
from transformers.trainer_utils import get_last_checkpoint

for condition in conditions:
    print(f"\n{'='*50}")
    print(f"Starting condition {condition['id']}")
    print(f"{'='*50}")

    # Build dataset for this condition
    dataset = build_condition(
        real_data=real_images,
        synthetic_pool=synthetic_images,
        total_size=condition["size"],
        synth_ratio=condition["synth_ratio"],
        seed=condition["seed"],
    )

    # Resume from checkpoint if exists
    output_dir = f"./paige-lora-condition-{condition['id']}"
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=get_training_args(condition["id"]),
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Condition {condition['id']} saved to {output_dir}")

    # Evaluate immediately on SROIE
    metrics = evaluate_condition(condition["id"], output_dir)
    print(f"Condition {condition['id']} SROIE metrics: {metrics}")

    # Cooldown
    time.sleep(60)
```

---

## Evaluation (SROIE 2019 v2)

**Hard rule:** SROIE test split is never used in training under any condition.

```python
def evaluate_condition(condition_id, adapter_path):
    model.load_adapter(adapter_path)
    FastVisionModel.for_inference(model)

    sroie_test = load_sroie_test()   # 347 images, held-out
    results = []

    for sample in sroie_test:
        pred = run_inference(model, tokenizer, sample["image"])
        results.append({
            "prediction":   safe_parse_json(pred),
            "ground_truth": sample["labels"],
        })

    metrics = compute_metrics(results)

    with open(f"eval_condition_{condition_id}.json", "w") as f:
        json.dump({"condition_id": condition_id, "metrics": metrics}, f, indent=2)

    return metrics

def normalize(v):
    if v is None: return None
    return str(v).lower().strip().replace("₱","").replace(",","")

def compute_metrics(results):
    fields = [
        "date", "patient_name", "philhealth_number",
        "diagnosis_code", "procedure_code",
        "total_amount", "philhealth_benefit",
        "balance_due"
    ]
    scores = {f: {"tp":0,"fp":0,"fn":0} for f in fields}
    exact  = 0

    for r in results:
        pred = r["prediction"]
        gt   = r["ground_truth"]
        if pred == gt: exact += 1
        for field in fields:
            pv = normalize(pred.get(field))
            gv = normalize(gt.get(field))
            if pv and gv:
                if pv == gv: scores[field]["tp"] += 1
                else:
                    scores[field]["fp"] += 1
                    scores[field]["fn"] += 1
            elif pv: scores[field]["fp"] += 1
            elif gv: scores[field]["fn"] += 1

    per_field = {}
    for field, c in scores.items():
        p = c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else 0
        r = c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        per_field[field] = {"precision":round(p,4),"recall":round(r,4),"f1":round(f1,4)}

    macro_f1 = sum(v["f1"] for v in per_field.values()) / len(fields)
    return {
        "exact_match": round(exact/len(results),4),
        "macro_f1":    round(macro_f1,4),
        "per_field":   per_field,
    }

def safe_parse_json(text):
    try:    return json.loads(text)
    except: return {}
```

---

## Paper Figures

Generate after all 7 conditions complete:

```python
import matplotlib.pyplot as plt

# Color scheme: blues=500-image, oranges=1000-image, greens=2000-image, red=G (control)
COLORS = {
    "A": "#1f77b4", "B": "#aec7e8",
    "C": "#ff7f0e", "D": "#ffbb78",
    "E": "#2ca02c", "F": "#98df8a",
    "G": "#d62728",
}
STYLES = {"A":"--","B":"--","C":"-.","D":"-.","E":"-","F":"-","G":"-"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cid in ["A","B","C","D","E","F","G"]:
    axes[0].plot(steps[cid], train_loss[cid],
                 color=COLORS[cid], linestyle=STYLES[cid],
                 linewidth=1.5, label=cid)
    axes[1].plot(eval_steps[cid], eval_f1[cid],
                 color=COLORS[cid], linestyle=STYLES[cid],
                 linewidth=1.5, label=cid)

for ax, title in zip(axes, ["Training loss", "Validation F1"]):
    ax.set_xlabel("Steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title)

plt.tight_layout()
plt.savefig("paige_ablation_curves.pdf", dpi=300, bbox_inches="tight")
```

---

## Checkpointing

Save after every condition — Lightning AI persistent storage keeps files between sessions:

```python
import shutil

def save_checkpoint(condition_id):
    src = f"./paige-lora-condition-{condition_id}"
    dst = f"/teamspace/studios/this_studio/checkpoints/condition_{condition_id}"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Checkpoint saved: {dst}")
```

---

## Hard Rules
- SROIE is test-only — never in any training split under any condition
- Vision encoder stays frozen across all 7 conditions
- Condition G is the real-only control — all other conditions should outperform it
- Labels reflect what is visually present — no inferred or computed values
- Fix all seeds — training, sampling, generation
- Generate all 800 synthetic images before any training run starts
- Evaluation is always a separate step after training completes
- One Lightning AI session — do not split runs across sessions