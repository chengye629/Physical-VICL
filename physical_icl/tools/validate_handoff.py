#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path, PureWindowsPath


def read_jsonl(p: Path):
    out=[]
    with p.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            x=json.loads(line)
            if not isinstance(x,dict): raise AssertionError(f'{p}:{i} not object')
            out.append(x)
    return out

def is_rel(s):
    value=str(s)
    return not (value.startswith('/') or Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or value.startswith('\\\\'))

ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args(); root=args.root
data_root=root/'data' if (root/'data/manifest').is_dir() else root
manifest=read_jsonl(data_root/'manifest/scale_5000_alpha3.jsonl')
index_rows=read_jsonl(data_root/'raw_retrieval/retrieval_index_eligible.jsonl')
retr=read_jsonl(data_root/'raw_retrieval/retrievals_clean_v21.jsonl')
pairs=read_jsonl(data_root/'raw_retrieval/pairs_clean_v21.jsonl')
flat=read_jsonl(data_root/'training_mapping/training_pairs_1demo.jsonl')
qmap=read_jsonl(data_root/'training_mapping/query_demo_map.jsonl')

assert len(manifest)==5000, len(manifest)
manifest_ids=[str(x['sample_id']) for x in manifest]; assert len(set(manifest_ids))==5000
index={str(x['sample_id']):x for x in index_rows}; assert len(index)==len(index_rows)==4151
assert {str(x['query_id']) for x in retr}==set(index)
raw_edges=[(str(x['query_id']),str(x['sample_id'])) for x in pairs]
assert len(raw_edges)==39260 and len(set(raw_edges))==39260
assert all(a!=b for a,b in raw_edges)
flat_edges={(str(x['query_id']),str(x['demo_id'])) for x in flat}
q_edges={(str(q['query_id']),str(d['demo_id'])) for q in qmap for d in q.get('demos',[])}
assert flat_edges==set(raw_edges)==q_edges

# Threshold sanity on raw selected pairs.
for p in pairs:
    assert float(p['language']) >= 0.84 - 1e-9
    assert float(p['dims']['process']) >= 0.55 - 1e-9
    assert float(p['physical']) >= 0.55 - 1e-9

# Endpoint mode and portable path sanity in derived mappings.
for x in flat:
    q=index[str(x['query_id'])]; d=index[str(x['demo_id'])]
    assert x['query_physics']['mode']==q.get('process_resolution_level')
    assert x['demo_physics']['mode']==d.get('process_resolution_level')
    assert is_rel(x['query_video_relpath']) and is_rel(x['demo_video_relpath'])
for q in qmap:
    qi=index[str(q['query_id'])]
    assert q['query_mode']==qi.get('process_resolution_level')
    assert is_rel(q['query_video_relpath'])
    ranks=[d.get('rank') for d in q.get('demos',[])]
    if ranks: assert ranks==list(range(1,len(ranks)+1))
    for d in q.get('demos',[]):
        di=index[str(d['demo_id'])]
        assert d['demo_mode']==di.get('process_resolution_level')
        assert is_rel(d['demo_video_relpath'])

# Exported raw files must not contain absolute video_path fields.
def walk(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in {'video_path','query_video_path','demo_video_path'} and isinstance(v,str):
                assert is_rel(v), f'absolute exported path: {v}'
            walk(v)
    elif isinstance(x,list):
        for v in x: walk(v)
for r in manifest+index_rows+retr+pairs: walk(r)

cards_dir=data_root/'physics_cards'
card_files=list(cards_dir.glob('*.json'))
if card_files:
    assert len(card_files)==5000, len(card_files)
    succ=0; fail=0
    for p in card_files:
        x=json.loads(p.read_text(encoding='utf-8'))
        assert str(x.get('sample_id'))==p.stem
        walk(x)
        if x.get('status')=='success': succ+=1
        else: fail+=1
    assert succ==4861 and fail==139, (succ,fail)

# Scan auxiliary failure and audit exports too. These used to bypass the
# canonical JSON/JSONL checks when copied as CSV or sidecar JSON.
path_keys={'video_path','query_video_path','demo_video_path','video_relpath','query_video_relpath','demo_video_relpath'}
for p in data_root.rglob('*'):
    if not p.is_file(): continue
    if cards_dir in p.parents: continue  # already parsed and checked above
    if p.suffix=='.json':
        walk(json.loads(p.read_text(encoding='utf-8')))
    elif p.suffix=='.jsonl':
        for row in read_jsonl(p): walk(row)
    elif p.suffix=='.csv':
        with p.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                for key in path_keys:
                    value=row.get(key)
                    if isinstance(value,str) and value.strip():
                        assert is_rel(value), f'absolute exported path: {p}:{key}={value}'

print(json.dumps({
    'manifest':len(manifest),
    'eligible':len(index),
    'queries':len(retr),
    'directed_pairs':len(raw_edges),
    'training_pairs':len(flat),
    'query_maps':len(qmap),
    'cards':len(card_files),
    'status':'PASS'
},indent=2))
