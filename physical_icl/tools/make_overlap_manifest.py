#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from collections import defaultdict, deque
from pathlib import Path

def read_jsonl(p):
    out=[]
    for line in Path(p).open(encoding='utf-8'):
        if line.strip(): out.append(json.loads(line))
    return out

ap=argparse.ArgumentParser(description='Build process-family / resolution-level balanced overlap subset for 8B-vs-32B comparison.')
ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--eligible-index',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--count',type=int,default=500); ap.add_argument('--seed',type=int,default=20260808); args=ap.parse_args()
manifest={str(r['sample_id']):r for r in read_jsonl(args.manifest)}; idx=read_jsonl(args.eligible_index); rng=random.Random(args.seed)
groups=defaultdict(list)
for r in idx:
    sid=str(r['sample_id'])
    if sid not in manifest: continue
    family=str(r.get('process_family') or 'unknown')
    mode=str(r.get('process_resolution_level') or 'unknown')
    groups[(family,mode)].append(sid)
keys=sorted(groups); rng.shuffle(keys); qs={}
for k in keys:
    vals=groups[k]; rng.shuffle(vals); qs[k]=deque(vals)
sel=[]; seen=set(); target=min(args.count,sum(map(len,groups.values())))
while len(sel)<target:
    progress=False
    for k in keys:
        q=qs[k]
        while q and q[0] in seen: q.popleft()
        if not q: continue
        sid=q.popleft(); seen.add(sid); sel.append(manifest[sid]); progress=True
        if len(sel)>=target: break
    if not progress: break
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('w',encoding='utf-8') as f:
    for r in sel: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({'selected':len(sel),'groups':len(groups),'seed':args.seed},indent=2))
