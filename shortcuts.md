## Train

.\.venv311\Scripts\python.exe train.py --id clean_data
.\.venv311\Scripts\python.exe train.py --id A
.\.venv311\Scripts\python.exe train.py --id C
.\.venv311\Scripts\python.exe train.py --id E

## Evaluate — 1_epoch group (data-scale ablation, fixed 1 epoch)

.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id A C E

## Evaluate — scale_epoch group (epoch-scaling ablation)

.\.venv311\Scripts\python.exe evaluate.py --results-dir results/scale_epoch --id A C E

## Evaluate — individual conditions

.\.venv311\Scripts\python.exe evaluate.py --results-dir . --id clean_data # Evaluates all checkpoints for clean_data
.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id A
.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id C
.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id E

## Evaluate — shortcuts for new flow

- Evaluate using workspace `eval_tiered_v1.jsonl` (preferred manifest):
  .\.venv311\Scripts\python.exe evaluate.py --test-manifest eval_tiered_v1.jsonl --results-dir . --id clean_data

- Evaluate a specific adapter directory (override autodetect):
  .\.venv311\Scripts\python.exe evaluate.py --adapter-path c:\\Users\\Jacob\\Documents\\FineTune\\paige-lora-condition-clean_data-epoch-2.5 --id clean_data-epoch-2.5

- Evaluate all adapters under a results dir (detects paige-lora-condition-\*):
  .\.venv311\Scripts\python.exe evaluate.py --results-dir c:\\Users\\Jacob\\Documents\\FineTune\\clean_data --id ALL

.\.venv311\Scripts\python.exe evaluate.py --results-dir results/scale_epoch --id A
.\.venv311\Scripts\python.exe evaluate.py --results-dir results/scale_epoch --id E

## Evaluate — one-by-one (clean_data adapters)

.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data" --id clean_data
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-1.0" --id clean_data-epoch-1.0
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-1.5" --id clean_data-epoch-1.5
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-2.0" --id clean_data-epoch-2.0
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-2.5" --id clean_data-epoch-2.5
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-3.0" --id clean_data-epoch-3.0

## Evaluate — final_res group (specific epochs)

.\.venv311\Scripts\python.exe evaluate.py --adapter-path Eval_result/final_res/paige-lora-condition-E-epoch-2 --id E-epoch-2
.\.venv311\Scripts\python.exe evaluate.py --adapter-path Eval_result/final_res/paige-lora-condition-E-epoch-3 --id E-epoch-3
.\.venv311\Scripts\python.exe evaluate.py --results-dir Eval_result/final_res --id E # Evaluates all epochs for condition E

## Evaluate — smoke test (50 samples, lower VRAM)

.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id A --max-samples 50 --batch-size 2

## Evaluate — VRAM-constrained (lower resolution)

.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id A C E --max-image-size 672 --batch-size 2

## Rebuild mixed test set (100 CORD + 26 Donut + 100 SROIE = 226, seed 42)

.\.venv311\Scripts\python.exe Datasets\Training_Data\golden\build_mixed_test.py
