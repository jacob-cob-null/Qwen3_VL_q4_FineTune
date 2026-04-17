"""
Dataset suitability assessment for Qwen3-VL fine-tuning.
Evaluates each source dataset on:
  - Sample count
  - Image availability (%)
  - Label completeness (% of fields non-null per sample)
  - Message format validity (has user+assistant turns)
  - Image size distribution (proxy for ocr-readability)
"""
import json, os, sys
from collections import defaultdict
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import resolve_image_path

MERGED = "Datasets/Training_Data/golden/merged/all_sources_train.jsonl"
FIELDS = ["date","patient_name","philhealth_number","diagnosis_code",
          "procedure_code","total_amount","philhealth_benefit","balance_due"]

stats = defaultdict(lambda: {
    "count": 0,
    "images_ok": 0,
    "label_fields_filled": 0,
    "label_fields_total": 0,
    "valid_chat_format": 0,
    "img_widths": [],
    "img_heights": [],
    "is_synthetic": None,
})

print("Scanning merged training JSONL...")
with open(MERGED, encoding="utf-8") as f:
    for raw in f:
        try:
            item = json.loads(raw)
        except Exception:
            continue

        src = item.get("source_dataset", "unknown")
        s = stats[src]
        s["count"] += 1
        s["is_synthetic"] = item.get("is_synthetic", False)

        # --- Image availability ---
        img_path = resolve_image_path(item.get("image_path", ""))
        img_ok = img_path and os.path.exists(img_path)
        if img_ok:
            s["images_ok"] += 1
            # Sample image dimensions (first 50 per source to keep it fast)
            if len(s["img_widths"]) < 50:
                try:
                    with Image.open(img_path) as im:
                        s["img_widths"].append(im.width)
                        s["img_heights"].append(im.height)
                except Exception:
                    pass

        # --- Label completeness ---
        gt = item.get("ground_truth", {})
        filled = sum(1 for f in FIELDS if gt.get(f) is not None)
        s["label_fields_filled"] += filled
        s["label_fields_total"] += len(FIELDS)

        # --- Chat format validity ---
        msgs = item.get("messages", [])
        has_user = any(m.get("role") == "user" for m in msgs if isinstance(m, dict))
        has_asst = any(m.get("role") == "assistant" for m in msgs if isinstance(m, dict))
        if has_user and has_asst:
            s["valid_chat_format"] += 1

print(f"\n{'='*70}")
print(f"{'Dataset':<25} {'N':>5} {'Img%':>6} {'Label%':>7} {'Chat%':>6} {'AvgSz':>10} {'Synth':>6}")
print(f"{'='*70}")

scored = []
for src, s in sorted(stats.items()):
    n = s["count"]
    img_pct   = 100 * s["images_ok"] / n if n else 0
    label_pct = 100 * s["label_fields_filled"] / s["label_fields_total"] if s["label_fields_total"] else 0
    chat_pct  = 100 * s["valid_chat_format"] / n if n else 0
    avg_w = int(sum(s["img_widths"]) / len(s["img_widths"])) if s["img_widths"] else 0
    avg_h = int(sum(s["img_heights"]) / len(s["img_heights"])) if s["img_heights"] else 0
    synth = "yes" if s["is_synthetic"] else "no"
    avg_sz = f"{avg_w}x{avg_h}" if avg_w else "n/a"

    # Composite score (higher = more suitable):
    # Weight: image availability 40%, label completeness 40%, chat format 20%
    score = 0.40 * img_pct + 0.40 * label_pct + 0.20 * chat_pct

    scored.append((score, src, n, img_pct, label_pct, chat_pct, avg_sz, synth))
    print(f"{src:<25} {n:>5} {img_pct:>5.1f}% {label_pct:>6.1f}% {chat_pct:>5.1f}% {avg_sz:>10} {synth:>6}")

print(f"{'='*70}")
scored.sort(reverse=True)
print(f"\n{'TOP 3 DATASETS (by suitability score)'}")
print(f"{'='*70}")
for rank, (score, src, n, img_pct, label_pct, chat_pct, avg_sz, synth) in enumerate(scored[:3], 1):
    print(f"\n#{rank} {src}")
    print(f"    Score:            {score:.1f}/100")
    print(f"    Samples:          {n}")
    print(f"    Image avail:      {img_pct:.1f}%")
    print(f"    Label fill:       {label_pct:.1f}%")
    print(f"    Chat format ok:   {chat_pct:.1f}%")
    print(f"    Avg image size:   {avg_sz} px")
    print(f"    Is synthetic:     {synth}")
