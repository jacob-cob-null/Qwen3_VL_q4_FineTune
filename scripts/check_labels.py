import json, os
path = "Datasets/Training_Data/golden/merged/all_sources_train.jsonl"
if not os.path.exists(path):
    print("missing")
    raise SystemExit
real=0; synth=0
mismatch_source_synth=0; mismatch_source_real=0
examples_synth=[]; examples_real=[]
for line in open(path, encoding='utf-8'):
    it=json.loads(line)
    is_synth=bool(it.get('is_synthetic', False))
    src = (it.get('source_dataset') or '').lower()
    img = (it.get('image_path') or '').lower()
    if is_synth:
        synth+=1
    else:
        real+=1
    hint = 'synthetic' in src or 'synthetic' in img or 'paige_synthetic' in src or 'paige_synthetic' in img
    if hint and not is_synth:
        mismatch_source_synth+=1
        if len(examples_synth)<5:
            examples_synth.append({'source':it.get('source_dataset'), 'image_path':it.get('image_path'), 'is_synthetic':is_synth})
    if is_synth and not hint:
        mismatch_source_real+=1
        if len(examples_real)<5:
            examples_real.append({'source':it.get('source_dataset'), 'image_path':it.get('image_path'), 'is_synthetic':is_synth})
print(f"real:{real} synth:{synth}")
print(f"mismatch_source_synth:{mismatch_source_synth}  mismatch_source_real:{mismatch_source_real}")
if examples_synth:
    print('\nExamples where path/source suggests synthetic but flagged real:')
    for e in examples_synth:
        print(e)
if examples_real:
    print('\nExamples where flagged synthetic but path/source not synthetic:')
    for e in examples_real:
        print(e)
