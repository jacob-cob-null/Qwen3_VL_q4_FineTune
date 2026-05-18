## Evaluate — one-by-one (clean_data adapters)

.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data" --id clean_data
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-1.0" --id clean_data-epoch-1.0
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-1.5" --id clean_data-epoch-1.5
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-2.0" --id clean_data-epoch-2.0
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-2.5" --id clean_data-epoch-2.5
.\.venv311\Scripts\python.exe evaluate.py --adapter-path "C:\\Users\\Jacob\\Documents\\FineTune\\clean_data\\paige-lora-condition-clean_data-epoch-3.0" --id clean_data-epoch-3.0

## Evaluate — smoke test (50 samples, lower VRAM)

.\.venv311\Scripts\python.exe evaluate.py --results-dir results/1_epoch --id A --max-samples 50 --batch-size 2
