#!/usr/bin/env python3
"""Create eval_tiered_v1.jsonl per user rules:
- Tier1 (Specialized): 50 synthetic, stratified by prompt_id (~5 per prompt), exclude Prompt 4
- Tier2 (Structural): 100 invoices_donut_v1, random seed 42
- Tier3 (Baseline): 100 sroie samples, prefer 'Vertical'/'Dense' tags else random
Normalizes image paths to workspace-relative and validates image existence.
"""
import os
import json
import random
from collections import defaultdict

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def normalize_image_path(p):
    if not p:
        return None
    p_norm = p.replace('\\','/').replace('\\\\','/')
    # If path contains 'Datasets/', return from there
    idx = p_norm.find('Datasets/')
    if idx != -1:
        rel = p_norm[idx:]
        return rel
    # If absolute under workspace root, make relative
    wp = WORKSPACE_ROOT.replace('\\','/')
    if p_norm.startswith(wp):
        rel = p_norm[len(wp)+1:]
        return rel
    # Otherwise return as-is but normalized slashes
    return p_norm

def path_exists(rel_path):
    if not rel_path:
        return False
    abs_p = os.path.join(WORKSPACE_ROOT, rel_path)
    return os.path.exists(abs_p)

def select_synthetic(synth_path, k=50, per_group_n=5, seed=42):
    random.seed(seed)
    rows = list(load_jsonl(synth_path))
    groups = defaultdict(list)
    for r in rows:
        prompt = None
        md = r.get('metadata') or {}
        prompt = md.get('prompt_id') or md.get('template_id') or md.get('generator') or 'default'
        groups[prompt].append(r)
    # Exclude Prompt 4 variants
    exclude_keys = set(['Prompt 4','prompt_4','4','Prompt4'])
    for ex in list(groups.keys()):
        if ex in exclude_keys:
            del groups[ex]
    selected = []
    # sample per group ~per_group_n
    for kgrp, items in groups.items():
        n = min(per_group_n, len(items))
        selected.extend(random.sample(items, n))
    # If not enough, fill from remaining pool
    if len(selected) < k:
        remaining = [r for r in rows if r not in selected]
        need = k - len(selected)
        if remaining:
            selected.extend(random.sample(remaining, min(need, len(remaining))))
    return selected[:k]

def select_invoices(mixed_path, k=100, seed=42):
    # Search the Testing_Data tree for invoice/donut jsonl rows in BOTH validation and test splits.
    # Deterministically pick the first `k` unique sample_ids (sorted by sample_id) to form a holdout set.
    candidates = []
    seen = set()
    # Search both Testing_Data and Training_Data (validation splits often live in Training_Data)
    search_roots = [os.path.join('Datasets', 'Testing_Data'), os.path.join('Datasets', 'Training_Data')]
    accept_splits = set(['test', 'validation', 'val'])
    for search_root in search_roots:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root):
                for fname in files:
                    if not fname.lower().endswith('.jsonl'):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        for r in load_jsonl(fpath):
                            src = (r.get('source_dataset') or '').lower()
                            if src == 'cord_v2':
                                continue
                            text_blob = src + ' ' + (r.get('sample_id') or '') + ' ' + json.dumps(r.get('metadata', {})).lower()
                            if 'invoice' in text_blob or 'donut' in text_blob:
                                split = (r.get('split') or '').lower()
                                if split in accept_splits:
                                    sid = r.get('sample_id')
                                    if sid and sid not in seen:
                                        candidates.append(r)
                                        seen.add(sid)
                    except Exception:
                        # ignore malformed files
                        continue
    # Also include any entries from the mixed_path manifest as a fallback
    if mixed_path and os.path.exists(mixed_path):
        try:
            for r in load_jsonl(mixed_path):
                src = (r.get('source_dataset') or '').lower()
                if src == 'cord_v2':
                    continue
                text_blob = src + ' ' + (r.get('sample_id') or '') + ' ' + json.dumps(r.get('metadata', {})).lower()
                if 'invoice' in text_blob or 'donut' in text_blob:
                    split = (r.get('split') or '').lower()
                    if split in accept_splits:
                        sid = r.get('sample_id')
                        if sid and sid not in seen:
                            candidates.append(r)
                            seen.add(sid)
        except Exception:
            pass

    # Deterministic ordering and selection
    candidates_sorted = sorted(candidates, key=lambda x: x.get('sample_id') or '')
    return candidates_sorted[:k]

def select_sroie(sroie_path, k=100, seed=42):
    random.seed(seed)
    rows = list(load_jsonl(sroie_path))
    # Try to filter by tags/metadata containing Vertical or Dense
    high_density = []
    for r in rows:
        md = r.get('metadata') or {}
        notes = ''
        for v in md.values():
            if isinstance(v, str):
                notes += ' ' + v
        # also check messages text
        for m in r.get('messages', []) or []:
            c = m.get('content')
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        notes += ' ' + str(part.get('text',''))
                    else:
                        notes += ' ' + str(part)
            else:
                notes += ' ' + str(c)
        if 'Vertical'.lower() in notes.lower() or 'Dense'.lower() in notes.lower() or 'vertical' in notes or 'dense' in notes:
            high_density.append(r)
    if len(high_density) >= k:
        return random.sample(high_density, k)
    # otherwise fallback to random sample
    if len(rows) <= k:
        return rows
    return random.sample(rows, k)

def uniquify_ids(entries):
    seen = {}
    for e in entries:
        sid = e.get('sample_id')
        if sid in seen:
            seen[sid] += 1
            new = f"{sid}-dup{seen[sid]}"
            e['sample_id'] = new
        else:
            seen[sid] = 0
    return entries

def annotate_and_normalize(entries, tier, stress_level=None):
    out = []
    for e in entries:
        e['tier'] = tier
        if stress_level:
            e.setdefault('metadata', {})['stress_level'] = stress_level
        ip = e.get('image_path')
        rel = normalize_image_path(ip) if ip else None
        e['image_path'] = rel
        e.setdefault('metadata', {})['image_exists'] = path_exists(rel)
        out.append(e)
    return out

def write_jsonl(path, entries):
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

def write_report(path, report):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

def main():
    synth_path = os.path.join('Datasets', 'Testing_Data', 'synthetic', 'canonical', 'test.raw.jsonl')
    mixed_path = os.path.join('Datasets', 'Testing_Data', 'mixed_test_226.jsonl')
    sroie_path = os.path.join('Datasets', 'Testing_Data', 'sroie_2019_v2', 'canonical', 'test.jsonl')

    tier1 = select_synthetic(synth_path, k=50, per_group_n=5, seed=42)
    tier2 = select_invoices(mixed_path, k=100, seed=42)
    tier3 = select_sroie(sroie_path, k=100, seed=42)

    # Annotate
    tier1 = annotate_and_normalize(tier1, 'Specialized', stress_level='high')
    tier2 = annotate_and_normalize(tier2, 'Structural')
    tier3 = annotate_and_normalize(tier3, 'Baseline')

    merged = tier1 + tier2 + tier3
    merged = uniquify_ids(merged)

    out_file = os.path.join(WORKSPACE_ROOT, 'eval_tiered_v1.jsonl')
    write_jsonl(out_file, merged)

    # Validation summary
    report = {
        'counts': {
            'Specialized': len(tier1),
            'Structural': len(tier2),
            'Baseline': len(tier3),
            'total': len(merged)
        },
        'missing_images': [],
        'duplicate_ids': []
    }
    ids = {}
    for e in merged:
        sid = e.get('sample_id')
        if sid in ids:
            report['duplicate_ids'].append(sid)
        ids[sid] = ids.get(sid, 0) + 1
        if not e.get('metadata', {}).get('image_exists'):
            report['missing_images'].append({'sample_id': sid, 'image_path': e.get('image_path')})

    report_path = os.path.join(WORKSPACE_ROOT, 'eval_tiered_v1.report.json')
    write_report(report_path, report)

    print('Wrote', out_file)
    print('Report written to', report_path)
    print(json.dumps(report['counts'], indent=2))

if __name__ == '__main__':
    main()
