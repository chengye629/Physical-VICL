#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = None
        for key in ("data", "items", "records", "annotations"):
            candidate = raw.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None and raw and all(isinstance(value, dict) for value in raw.values()):
            rows = list(raw.values())
        if rows is None:
            raise ValueError(
                "unsupported metadata structure: expected a list or a common list wrapper"
            )
    else:
        raise ValueError("metadata JSON must contain a list or object")
    return [row for row in rows if isinstance(row, dict)]


def caption_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = caption_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("caption", "text", "instruction"):
            text = caption_text(value.get(key))
            if text:
                return text
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + chr(10)
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join WISA metadata with extracted videos and build a Pass A manifest."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--videos-root",
        type=Path,
        required=True,
        help="Root containing separately extracted shard directories; scanned recursively.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    videos_root = args.videos_root.resolve()
    output_dir = args.output_dir.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not videos_root.is_dir():
        raise NotADirectoryError(videos_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_metadata_rows = load_metadata_rows(metadata_path)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    invalid_metadata_rows: list[dict[str, Any]] = []
    identical_metadata_duplicates: list[str] = []
    conflicting_metadata_duplicates: list[dict[str, Any]] = []
    metadata_conflict_ids: set[str] = set()

    for index, row in enumerate(raw_metadata_rows):
        video_name = str(row.get("video_name") or "").strip()
        sample_id = Path(video_name).stem
        if not video_name or not sample_id:
            invalid_metadata_rows.append({"row_index": index, "reason": "missing video_name"})
            continue
        previous = metadata_by_id.get(sample_id)
        if previous is None:
            metadata_by_id[sample_id] = row
        elif canonical_json(previous) == canonical_json(row):
            identical_metadata_duplicates.append(sample_id)
        else:
            metadata_conflict_ids.add(sample_id)
            conflicting_metadata_duplicates.append(
                {
                    "sample_id": sample_id,
                    "first_video_name": previous.get("video_name"),
                    "duplicate_video_name": row.get("video_name"),
                }
            )

    videos_by_id: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(videos_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos_by_id[path.stem].append(path)

    selected_paths: dict[str, Path] = {}
    identical_file_duplicates: list[dict[str, Any]] = []
    conflicting_file_duplicates: list[dict[str, Any]] = []
    conflicting_file_ids: set[str] = set()

    for sample_id, candidates in sorted(videos_by_id.items()):
        if len(candidates) == 1:
            selected_paths[sample_id] = candidates[0]
            continue
        hashes: dict[str, list[Path]] = defaultdict(list)
        for path in candidates:
            hashes[sha256_file(path)].append(path)
        if len(hashes) == 1:
            digest = next(iter(hashes))
            kept = sorted(candidates, key=lambda value: str(value))[0]
            selected_paths[sample_id] = kept
            identical_file_duplicates.append(
                {
                    "sample_id": sample_id,
                    "sha256": digest,
                    "kept": str(kept),
                    "ignored": [
                        str(path)
                        for path in sorted(candidates, key=lambda value: str(value))
                        if path != kept
                    ],
                }
            )
        else:
            conflicting_file_ids.add(sample_id)
            conflicting_file_duplicates.append(
                {
                    "sample_id": sample_id,
                    "files_by_sha256": {
                        digest: [str(path) for path in paths]
                        for digest, paths in sorted(hashes.items())
                    },
                }
            )

    excluded_ids = metadata_conflict_ids | conflicting_file_ids
    manifest_rows: list[dict[str, str]] = []
    portable_metadata_rows: list[dict[str, Any]] = []
    missing_video_ids: list[str] = []
    empty_instruction_ids: list[str] = []

    for sample_id, metadata in sorted(metadata_by_id.items()):
        if sample_id in excluded_ids:
            continue
        video_path = selected_paths.get(sample_id)
        if video_path is None:
            missing_video_ids.append(sample_id)
            continue
        relative = video_path.relative_to(videos_root)
        source_shard = relative.parts[0] if len(relative.parts) > 1 else ""
        instruction = caption_text(metadata.get("captions"))
        if not instruction:
            empty_instruction_ids.append(sample_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "video_path": str(video_path),
            }
        )
        portable_metadata_rows.append(
            {
                "sample_id": sample_id,
                "video_name": metadata.get("video_name"),
                "instruction": instruction,
                "captions": metadata.get("captions"),
                "source_shard": source_shard,
            }
        )

    orphan_video_ids = sorted(set(selected_paths) - set(metadata_by_id))
    report = {
        "metadata_source": str(metadata_path),
        "videos_root": str(videos_root),
        "identical_metadata_duplicate_ids": sorted(set(identical_metadata_duplicates)),
        "conflicting_metadata_duplicates": conflicting_metadata_duplicates,
        "invalid_metadata_rows": invalid_metadata_rows,
        "identical_file_duplicates": identical_file_duplicates,
        "conflicting_file_duplicates": conflicting_file_duplicates,
        "missing_video_ids": missing_video_ids,
        "empty_instruction_ids": empty_instruction_ids,
        "orphan_video_ids": orphan_video_ids,
    }
    summary = {
        "metadata_rows": len(raw_metadata_rows),
        "unique_metadata_ids": len(metadata_by_id),
        "discovered_video_ids": len(videos_by_id),
        "manifest_rows": len(manifest_rows),
        "identical_file_duplicate_ids": len(identical_file_duplicates),
        "conflicting_file_ids_excluded": len(conflicting_file_ids),
        "conflicting_metadata_ids_excluded": len(metadata_conflict_ids),
        "missing_video_ids": len(missing_video_ids),
        "empty_instruction_ids": len(empty_instruction_ids),
        "orphan_video_ids": len(orphan_video_ids),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest": str(output_dir / "annotation_manifest.jsonl"),
        "portable_metadata": str(output_dir / "wisa_metadata.jsonl"),
        "duplicate_report": str(output_dir / "duplicate_report.json"),
    }

    write_jsonl(output_dir / "annotation_manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "wisa_metadata.jsonl", portable_metadata_rows)
    write_json(output_dir / "duplicate_report.json", report)
    summary["manifest_sha256"] = sha256_file(
        output_dir / "annotation_manifest.jsonl"
    )
    summary["portable_metadata_sha256"] = sha256_file(
        output_dir / "wisa_metadata.jsonl"
    )
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
