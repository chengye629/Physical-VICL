#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def portable(raw: str, project_root: Path) -> str:
    p=Path(raw)
    raw_posix=raw.replace('\\','/')
    cross_platform_abs=raw_posix.startswith('/') or p.is_absolute() or PureWindowsPath(raw).is_absolute()
    if not cross_platform_abs: return PurePosixPath(raw_posix).as_posix()
    try: return p.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        parts=[x for x in PurePosixPath(raw_posix).parts if x not in {'/','\\'}]
        if 'data' in parts:
            return PurePosixPath(*parts[parts.index('data'):]).as_posix()
        raise ValueError(f'Cannot sanitize absolute path outside project root: {raw}')


def transform(x: Any, project_root: Path) -> Any:
    if isinstance(x, dict):
        out={}
        for k,v in x.items():
            if k in {'video_path','query_video_path','demo_video_path'} and isinstance(v,str) and v.strip():
                out[k]=portable(v,project_root)
            else: out[k]=transform(v,project_root)
        return out
    if isinstance(x,list): return [transform(v,project_root) for v in x]
    return x


def process_file(src: Path, dst: Path, root: Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix=='.jsonl':
        with src.open(encoding='utf-8') as fi, dst.open('w',encoding='utf-8') as fo:
            for line in fi:
                if line.strip(): fo.write(json.dumps(transform(json.loads(line),root),ensure_ascii=False)+'\n')
    elif src.suffix=='.json':
        dst.write_text(json.dumps(transform(json.loads(src.read_text(encoding='utf-8')),root),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    elif src.suffix=='.csv':
        with src.open(encoding='utf-8', newline='') as fi:
            reader=csv.DictReader(fi)
            fields=reader.fieldnames or []
            rows=[]
            for row in reader:
                for key in ('video_path','query_video_path','demo_video_path'):
                    if key in row and isinstance(row[key],str) and row[key].strip():
                        row[key]=portable(row[key],root)
                rows.append(row)
        with dst.open('w',encoding='utf-8',newline='') as fo:
            writer=csv.DictWriter(fo,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    else: shutil.copy2(src,dst)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,required=True); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    if args.input.is_dir():
        for src in args.input.rglob('*'):
            if src.is_file(): process_file(src,args.output/src.relative_to(args.input),args.project_root)
    else: process_file(args.input,args.output,args.project_root)

if __name__=='__main__': main()
