#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--parts',type=int,required=True)
args=ap.parse_args()
rows=[]
for line in args.input.open(encoding='utf-8'):
    if line.strip(): rows.append(json.loads(line))
args.output_dir.mkdir(parents=True,exist_ok=True)
chunks=[[] for _ in range(args.parts)]
for i,row in enumerate(rows): chunks[i%args.parts].append(row)
for i,chunk in enumerate(chunks):
    p=args.output_dir/f'part_{i:02d}.jsonl'
    with p.open('w',encoding='utf-8') as f:
        for x in chunk: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(f'{p}\t{len(chunk)}')
