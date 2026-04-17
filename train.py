import json
import os
# Disable Dynamo globally since Triton is unstable on Windows
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import random
import time
import gc
from PIL import Image


try:
    import torch
    # Provide fallback for float8 dtype names expected by PEFT/unsloth
    try:
        float8_names = [
            "float8_e8m0fnu",
            "float8_e4m3fn",
            "float8_e5m2",
        ]
        for _name in float8_names:
            if not hasattr(torch, _name):
                try:
                    setattr(torch, _name, torch.bfloat16)
                except Exception:
                    pass

        # Add a fallback __getattr__ to map unknown float8 names to bfloat16
        if not hasattr(torch, "__orig_getattr__"):
            orig_getattr = getattr(torch, "__getattr__", None)
            def _torch_getattr(name):
                if "float8" in name:
                    return torch.bfloat16
                if orig_getattr:
                    return orig_getattr(name)
                raise AttributeError(f"module 'torch' has no attribute '{name}'")
            setattr(torch, "__orig_getattr__", orig_getattr)
            setattr(torch, "__getattr__", _torch_getattr)
    except Exception:
        pass
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer
    # Ensure multimodal messages contain image placeholders when images are present
    try:
        from scripts.patch_prepare_multimodal import patch as _patch_prepare_multimodal
        _patch_prepare_multimodal()
    except Exception:
        pass
    UNSLOTH_AVAILABLE = True
except (ImportError, OSError):
    UNSLOTH_AVAILABLE = False

try:
    from evaluate import evaluate_condition
except ImportError:
    def evaluate_condition(cond_id, path, model, tokenizer):
        return {"status": "Evaluation module missing", "score": 0.0}

# ---------------------------------------------------------------------------
# Image path resolution
# The JSONL files were generated with relative/stale paths. This function
# resolves them to the actual on-disk locations without modifying the data.
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))  # FineTune/ root

DATASET_SEARCH_ROOTS = [
    # Where real training source images live (cord_v2, invoices_donut_v1, etc.)
    os.path.join(_BASE, "Datasets", "Training_Data", "golden", "sources"),
    # Synthetic images: sources/synthetic/images/train/
    os.path.join(_BASE, "Datasets", "Training_Data", "golden", "sources", "synthetic"),
    # paige_synthetic canon variant
    os.path.join(_BASE, "Datasets", "Training_Data", "golden", "sources", "paige_synthetic"),
    # Test images (for eval)
    os.path.join(_BASE, "Datasets", "Testing_Data", "sroie_2019_v2"),
    # Fallback: workspace root itself
    _BASE,
]

def resolve_image_path(image_path):
    """
    Resolve an image_path from the JSONL to an absolute path that actually
    exists on disk. Tries multiple strategies:
      1. Absolute path as-is (already resolved).
      2. Relative to the FineTune workspace root.
      3. Strip any leading directory components and search known image roots.
    Returns the resolved path, or the original string if nothing is found.
    """
    if not image_path:
        return ""
    # Already absolute and exists
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    # Relative to workspace root
    candidate = os.path.join(_BASE, image_path)
    if os.path.exists(candidate):
        return candidate
    # Search in known roots using just the filename
    filename = os.path.basename(image_path)
    for root in DATASET_SEARCH_ROOTS:
        for dirpath, _dirs, files in os.walk(root):
            if filename in files:
                return os.path.join(dirpath, filename)
    # Give up — return original so callers can decide what to do
    return image_path


# --- UNCHANGED CONFIGURATION ---
CONDITIONS = [
    {"id": "A", "size": 500,  "synth_ratio": 0.30, "seed": 42},
    {"id": "B", "size": 500,  "synth_ratio": 0.40, "seed": 43},
    {"id": "C", "size": 1000, "synth_ratio": 0.30, "seed": 44},
    {"id": "D", "size": 1000, "synth_ratio": 0.40, "seed": 45},
    {"id": "E", "size": 2000, "synth_ratio": 0.30, "seed": 46},
    {"id": "F", "size": 2000, "synth_ratio": 0.40, "seed": 47},
    {"id": "G", "size": 2000, "synth_ratio": 0.00, "seed": 48},
]

def build_condition(real_data, synthetic_pool, total_size, synth_ratio, seed):
    random.seed(seed)
    n_synth = int(total_size * synth_ratio)
    n_real  = total_size - n_synth
    selected_real = random.sample(real_data, min(n_real, len(real_data)))
    selected_synth = random.sample(synthetic_pool, min(n_synth, len(synthetic_pool)))
    combined = selected_real + selected_synth
    random.shuffle(combined)
    return combined

def setup_model():
    if not UNSLOTH_AVAILABLE:
        return None, None
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_id,
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    
    # Fix mismatched tokens issue
    if actual_tokenizer.pad_token is None:
        # Default fallback for Qwen models
        actual_tokenizer.pad_token = "<|endoftext|>"
        
    if len(actual_tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(actual_tokenizer))

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, 
        lora_alpha=32,
        lora_dropout=0,
        random_state=42,
    )
    return model, tokenizer

def get_training_args(condition_id, is_smoke_test=False):
    output_dir = f"./paige-lora-condition-{condition_id}"
    params = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 1 if is_smoke_test else 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        # Use a transformers/trl-recognized optimizer name
        "optim": "adamw_torch",
        # Use bfloat16 for training on this model; disable fp16
        "bf16": True,
        "fp16": False,
        "learning_rate": 2e-4,
        "max_seq_length": 1024,
        "logging_steps": 10,
        "save_total_limit": 1,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    if is_smoke_test:
        params.update({"max_steps": 10, "output_dir": "./paige-smoketest"})
    else:
        params.update({"num_train_epochs": 3, "save_steps": 100})
    return SFTConfig(**params)


# Dataset wrapper that yields raw conversation dicts consumed by UnslothVisionDataCollator.
# DO NOT pre-tokenize here – pixel_values are variable-length for vision models like Qwen3-VL
# (patch count depends on image resolution) and cannot be stacked by the default collator.
# UnslothVisionDataCollator handles tokenization + collation in one step.
class RawSampleDataset:
    """Returns one dict per sample with the keys the UnslothVisionDataCollator expects:
      - 'messages': list of chat-format dicts (role/content)
      - 'images':   list containing one PIL.Image (or None / empty list)
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_messages = sample.get("messages", [])

        # --- Normalise to the chat format Qwen3-VL / apply_chat_template expects ---
        # Each item should be {"role": "user"|"assistant", "content": [{"type": "text", "text": ...}]}
        def _normalise_message(m):
            """Best-effort conversion of an arbitrary message object to a chat dict."""
            if isinstance(m, dict) and "role" in m and "content" in m:
                # Already in chat format – pass through
                return m
            if isinstance(m, str):
                return {"role": "user", "content": [{"type": "text", "text": m}]}
            if isinstance(m, dict):
                text = " ".join(str(v) for v in m.values())
                return {"role": "user", "content": [{"type": "text", "text": text}]}
            return {"role": "user", "content": [{"type": "text", "text": str(m)}]}

        if isinstance(raw_messages, list) and raw_messages:
            messages = [_normalise_message(m) for m in raw_messages]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": str(raw_messages)}]}]

        # Ensure at least a user + assistant turn so the trainer has a loss target
        if not any(m.get("role") == "assistant" for m in messages):
            gt = sample.get("ground_truth", {})
            answer = gt if isinstance(gt, str) else str(gt)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

        # --- Load image ---
        image_path = sample.get("image") or ""
        images = []
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                images = [img]
                # Add image placeholder to the first user turn if missing
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            has_image_item = any(
                                isinstance(c, dict) and c.get("type") == "image"
                                for c in content
                            )
                            if not has_image_item:
                                msg["content"] = [{"type": "image"}] + content
                        break
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")

        return {"messages": messages, "images": images}
        
# Fields required by the fine-tuning target schema
_TARGET_FIELDS = [
    "date", "patient_name", "philhealth_number", "diagnosis_code",
    "procedure_code", "total_amount", "philhealth_benefit", "balance_due",
]

# Datasets that have been assessed as actively harmful (all-null labels).
# These are excluded at load time to prevent the model learning to output nulls.
_EXCLUDED_SOURCES = {"invoices_donut_v1"}

_EXTRACTION_PROMPT = (
    "You are a medical billing document extraction assistant. "
    "Given the image of a Philippine hospital Statement of Account (SOA) or billing form, "
    "extract the following fields and return them as a single JSON object with exactly these keys: "
    "date, patient_name, philhealth_number, diagnosis_code, procedure_code, "
    "total_amount, philhealth_benefit, balance_due. "
    "Use null for any field not present in the document."
)

def _build_messages_from_ground_truth(ground_truth):
    """Build a valid user+assistant chat turn pair for samples that have no messages.
    Used for paige_synthetic entries where messages=[] but ground_truth is fully populated.
    """
    user_turn = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": _EXTRACTION_PROMPT},
        ],
    }
    # Ensure all 8 target fields are present in the output (null if missing)
    answer = {f: ground_truth.get(f) for f in _TARGET_FIELDS}
    assistant_turn = {
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(answer, ensure_ascii=False)}],
    }
    return [user_turn, assistant_turn]


def run_train(is_smoke_test, selected_ids=None):
    # Load training pools from extracted JSONL files
    merged_train_path = "Datasets/Training_Data/golden/merged/all_sources_train.jsonl"
    real_images = []
    synthetic_images = []
    data_loaded = False
    if os.path.exists(merged_train_path):
        data_loaded = True
        with open(merged_train_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                # Normalize to the fields the trainer expects.
                # Resolve image_path to an absolute on-disk path at load time
                # so the dataset/collator never sees stale relative paths.
                raw_img_path = item.get("image_path", "")
                source = item.get("source_dataset", "")

                # Skip datasets assessed as actively harmful (all-null labels)
                if source in _EXCLUDED_SOURCES:
                    continue

                # Skip any sample where every target field is null/missing
                gt = item.get("ground_truth", {})
                if not any(gt.get(f) is not None for f in _TARGET_FIELDS):
                    continue

                msgs = item.get("messages", [])
                # For paige_synthetic (and others) with no messages,
                # synthesise a proper chat turn from the ground_truth.
                if not msgs:
                    msgs = _build_messages_from_ground_truth(gt)

                sample = {
                    "image": resolve_image_path(raw_img_path),
                    "ground_truth": gt,
                    "messages": msgs,
                    "is_synthetic": bool(item.get("is_synthetic", False)),
                }
                if sample["is_synthetic"]:
                    synthetic_images.append(sample)
                else:
                    real_images.append(sample)
    else:
        # fall back to previous mocks if merged file missing
        real_images = [{"image": "real.png", "instruction": "Extract data", "output": "2026-04-15"}] * 3000
        synthetic_images = [{"image": "synth.png", "instruction": "Extract data", "output": "2026-04-15"}] * 2000

    # Filtering Logic: If IDs are provided, only run those. Otherwise run all.
    if is_smoke_test:
        run_conditions = [CONDITIONS[0]]
    elif selected_ids:
        # Normalize to uppercase and filter
        targets = [id.upper() for id in selected_ids]
        run_conditions = [c for c in CONDITIONS if c["id"] in targets]

        if not run_conditions:
            print(f"Error: No valid conditions found in {targets}")
            return
    else:
        run_conditions = CONDITIONS

    for condition in run_conditions:
        print(f"\n{'='*60}\nACTIVE SESSION: Condition {condition['id']}\n{'='*60}")

        model, tokenizer = setup_model()
        
        train_data = build_condition(
            real_data=real_images,
            synthetic_pool=synthetic_images,
            total_size=condition["size"],
            synth_ratio=condition["synth_ratio"],
            seed=condition["seed"],
        )

        # For smoke tests with real data, cap at 32 samples to keep runtime short.
        # When real data is NOT available, fall back to minimal dummy samples so
        # the pipeline can still be verified end-to-end.
        if is_smoke_test:
            if data_loaded:
                # Use a small slice of real data so the smoke test is meaningful
                import random as _rnd
                _rnd.seed(0)
                train_data = _rnd.sample(train_data, min(32, len(train_data)))
            else:
                simple = []
                for i in range(32):
                    simple.append({
                        "image": "",
                        "ground_truth": {"date": "01/01/2024"},
                        "messages": [
                            {"role": "user",    "content": [{"type": "text", "text": f"Example prompt {i}: extract fields."}]},
                            {"role": "assistant","content": [{"type": "text", "text": '{"date": "01/01/2024"}'}]},
                        ],
                        "is_synthetic": False,
                    })
                train_data = simple

        args = get_training_args(condition["id"], is_smoke_test)
        # The dataset is raw samples; UnslothVisionDataCollator handles tokenization,
        # so we must skip SFTTrainer's internal dataset preparation.
        try:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
            args.dataset_text_field = ""
        except Exception:
            pass
        
        if UNSLOTH_AVAILABLE:
            # Build the raw-sample dataset – tokenization is deferred to the collator
            raw_train = RawSampleDataset(train_data)
            data_collator = UnslothVisionDataCollator(model, tokenizer)

            if is_smoke_test and data_loaded:
                print("Smoke test with real data: running a short training run")
                args.max_steps = 20
                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    args=args,
                    train_dataset=raw_train,
                )
                trainer.train()
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"Smoke-train complete: saved to {args.output_dir}")
            elif is_smoke_test and not data_loaded:
                print("Smoke test mode: skipping training to avoid dataset/tokenization setup.")
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"Smoke test: saved model snapshot to {args.output_dir}")
            else:
                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    args=args,
                    train_dataset=raw_train,
                )
                trainer.train()
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"Complete: Saved to {args.output_dir}")
        else:
            print("Mode: Dry Run (Unsloth Unavailable)")

        # Evaluation – cap at 20 samples during smoke tests for speed
        print(f"Starting Evaluation for {condition['id']}...")
        eval_cap = 20 if is_smoke_test else None
        metrics = evaluate_condition(condition["id"], args.output_dir, model, tokenizer, max_eval_samples=eval_cap)
        print(f"Final Metrics for {condition['id']}: {metrics}")

        # Resource Release
        if 'trainer' in locals():
            try:
                del trainer
            except Exception:
                pass
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"VRAM Cleared. Cooling down...")
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pAIge Fine-tuning Ablation Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification run")
    
    # Updated Argument: Accepts multiple letters (e.g., --id A B G)
    parser.add_argument("--id", nargs="+", help="Specific condition letters to run (e.g., A B G)")
    
    cmd_args = parser.parse_args()

    run_train(cmd_args.smoke_test, cmd_args.id)