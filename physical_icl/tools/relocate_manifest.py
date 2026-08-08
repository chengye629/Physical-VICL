#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser(description='Create a machine-local runtime manifest without changing sample IDs.')
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--video-root',type=Path,required=True,help='Local root prepended to video_relpath/project-relative video_path')
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
count=0; missing=[]
with args.input.open(encoding='utf-8') as fi, args.output.open('w',encoding='utf-8') as fo:
    for line in fi:
        if not line.strip(): continue
        r=json.loads(line)
        raw=r.get('video_relpath') or r.get('video_path')
        if not raw: raise KeyError(f"missing video path for {r.get('sample_id')}")
        p=Path(str(raw))
        if not p.is_absolute(): p=args.video_root/p
        p=p.resolve()
        if not p.is_file(): missing.append((r.get('sample_id'),str(p)))
        r['video_path']=str(p)
        fo.write(json.dumps(r,ensure_ascii=False)+'\n'); count+=1
if missing:
    raise SystemExit(f'{len(missing)} missing videos; first={missing[:5]}')
print(json.dumps({'rows':count,'missing':0,'output':str(args.output)},indent=2))
