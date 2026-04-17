from unsloth import FastVisionModel
from PIL import Image
import torch

model_id = "Qwen/Qwen3-VL-4B-Instruct"
print('Loading model/tokenizer (this may take a moment)...')
model, tokenizer = FastVisionModel.from_pretrained(model_name=model_id, load_in_4bit=True, torch_dtype=torch.float16)
print('Loaded.')

print('\nTokenizer info:')
try:
    print('model_input_names:', getattr(tokenizer, 'model_input_names', None))
except Exception as e:
    print('model_input_names error:', e)
print('type(tokenizer):', type(tokenizer))
print('has image_processor:', hasattr(tokenizer, 'image_processor'))
if hasattr(tokenizer, 'image_processor'):
    print('image_processor class:', type(tokenizer.image_processor))

print('\nModel config keys (subset):')
for k in ['vision_config', 'image_size', 'mm_vision', 'mm_token_type_ids']:
    print(k, getattr(model.config, k, None))

# Create a tiny image
img = Image.new('RGB', (64, 64), color=(128,128,128))
text = 'Example prompt: extract fields.'
print('\nTokenizing a single sample...')
enc = tokenizer(text=[text], images=[img], return_tensors='pt', padding=True, truncation=True, max_length=1024)
print('encoding keys:', list(enc.keys()))
for k,v in enc.items():
    try:
        if hasattr(v, 'shape'):
            print(f"{k}: tensor shape {v.shape}")
        else:
            print(f"{k}: type {type(v)}")
    except Exception as e:
        print(f"{k}: error printing - {e}")

print('\nAttempt tokenizer.pad on two samples...')
enc2 = tokenizer.pad(enc, padding=True, return_tensors='pt')
for k,v in enc2.items():
    try:
        print(f"{k}: {type(v)} {getattr(v,'shape', '')}")
    except Exception:
        print(k, type(v))

print('\nDone')
