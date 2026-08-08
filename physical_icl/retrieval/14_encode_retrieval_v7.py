#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit(
            "sentence-transformers is required. In the environment used for the earlier Qwen3-Embedding runs, "
            "verify with: python -c 'import sentence_transformers; print(sentence_transformers.__version__)'\n"
            f"Import error: {e!r}"
        )

    rows = read_jsonl(args.index)
    if not rows:
        raise SystemExit("empty retrieval index")

    model_kwargs = {}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = SentenceTransformer(
            args.model_path,
            device=args.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        model = SentenceTransformer(
            args.model_path,
            device=args.device,
            trust_remote_code=True,
        )
    model.max_seq_length = args.max_seq_length

    instructions = {
        "language": "Represent this detailed physics card, including objects, actions, physical processes, impacts, and mechanisms, for retrieving physically and semantically similar video events: ",
        "object": "Represent the physical objects, their roles, material cues, and initial states for physical similarity retrieval: ",
        "process": "Represent the physical process, intervention, and temporal pattern for physical similarity retrieval: ",
        "impact": "Represent the observable object impacts and state transitions for physical similarity retrieval: ",
        "mechanism": "Represent the underlying physical mechanisms for physical similarity retrieval: ",
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    ids = [r["sample_id"] for r in rows]
    (args.output_root / "sample_ids.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    meta = {
        "model_path": args.model_path,
        "sample_count": len(rows),
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "dimensions": {},
    }

    for dim in ["language", "object", "process", "impact", "mechanism"]:
        texts = [instructions[dim] + str(r.get("texts", {}).get(dim, "")) for r in rows]
        print(f"[ENCODE] {dim}: {len(texts)} texts")
        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        emb = np.asarray(emb, dtype=np.float16)
        np.save(args.output_root / f"{dim}.npy", emb)
        meta["dimensions"][dim] = list(emb.shape)

    (args.output_root / "embedding_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
