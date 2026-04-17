import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import resolve_image_path

tests = [
    # Synthetic training image (missing prefix)
    "synthetic/images/train/paige_synth_000001.png",
    # SROIE test image (wrong Training_Data path in JSONL)
    "Datasets/Training_Data/golden/sources/sroie_2019_v2/images/test/sroie_2019_v2_test_000000.png",
    # Real training image (should already work relative to root)
    "Datasets/Training_Data/golden/sources/cord_v2/images/train/cord_v2_train_000001.png",
    # Test image using actual Testing_Data location
    "Datasets/Testing_Data/sroie_2019_v2/images/test/sroie_2019_v2_test_000000.png",
]

for p in tests:
    resolved = resolve_image_path(p)
    exists = os.path.exists(resolved)
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {p}")
    if resolved != p:
        print(f"       -> {resolved}")
