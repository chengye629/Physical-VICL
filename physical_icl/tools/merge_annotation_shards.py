#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--shards-root',type=Path,required=True)
ap.add_argument('--output-root',type=Path,required=True)
args=ap.parse_args()
pa=args.output_root/'pass_a'; pb=args.output_root/'cards'
pa.mkdir(parents=True,exist_ok=True); pb.mkdir(parents=True,exist_ok=True)
counts={'pass_a':0,'cards':0}
seen={'pass_a':set(),'cards':set()}
for shard in sorted(args.shards_root.glob('worker_*')):
    for kind,srcdir,dstdir in [('pass_a',shard/'pass_a'/'pass_a',pa),('cards',shard/'pass_b'/'cards',pb)]:
        if not srcdir.is_dir(): continue
        for src in srcdir.glob('*.json'):
            if src.name in seen[kind]: raise RuntimeError(f'duplicate {kind}: {src.name}')
            seen[kind].add(src.name); shutil.copy2(src,dstdir/src.name); counts[kind]+=1
(args.output_root/'merge_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(counts,indent=2))
