#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + chr(10)
            for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the annotation manifest for Physical-ICL wisa_test100_v2."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Local root containing the downloaded data/wisa_test100_v2 tree.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    ids_path = args.ids_file.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not ids_path.is_file():
        raise FileNotFoundError(ids_path)
    if not repo_root.is_dir():
        raise NotADirectoryError(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = [
        line.strip()
        for line in ids_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("test100_v2_ids.txt contains duplicate IDs")

    metadata_rows = read_jsonl(metadata_path)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    duplicate_metadata_ids: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    missing_video_ids: list[str] = []
    manifest_rows: list[dict[str, str]] = []
    portable_rows: list[dict[str, Any]] = []

    for index, row in enumerate(metadata_rows):
        sample_id = str(row.get("sample_id") or "").strip()
        gt_path = str(row.get("gt_path") or "").strip()
        if not sample_id or not gt_path:
            invalid_rows.append(
                {"row_index": index, "reason": "missing sample_id or gt_path"}
            )
            continue
        if sample_id in metadata_by_id:
            duplicate_metadata_ids.append(sample_id)
            continue
        metadata_by_id[sample_id] = row

        video_path = (repo_root / gt_path).resolve()
        try:
            video_path.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"gt_path escapes repo root: {gt_path}") from error
        if not video_path.is_file():
            missing_video_ids.append(sample_id)
            continue
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "video_path": str(video_path),
            }
        )
        portable_rows.append(row)

    actual_ids = {row["sample_id"] for row in manifest_rows}
    expected_id_set = set(expected_ids)
    missing_expected_ids = sorted(expected_id_set - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_id_set)
    count_matches = len(manifest_rows) == args.expected_count
    id_set_matches = not missing_expected_ids and not unexpected_ids

    write_jsonl(output_dir / "annotation_manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "test_v2_metadata.jsonl", portable_rows)
    report = {
        "duplicate_metadata_ids": sorted(set(duplicate_metadata_ids)),
        "invalid_metadata_rows": invalid_rows,
        "missing_video_ids": missing_video_ids,
        "missing_expected_ids": missing_expected_ids,
        "unexpected_ids": unexpected_ids,
    }
    write_json(output_dir / "build_report.json", report)
    summary = {
        "campaign": "wisa_test100_v2",
        "expected_count": args.expected_count,
        "metadata_rows": len(metadata_rows),
        "manifest_rows": len(manifest_rows),
        "count_matches": count_matches,
        "id_set_matches": id_set_matches,
        "metadata_sha256": sha256_file(metadata_path),
        "ids_file_sha256": sha256_file(ids_path),
        "manifest_sha256": sha256_file(output_dir / "annotation_manifest.jsonl"),
        "portable_metadata_sha256": sha256_file(
            output_dir / "test_v2_metadata.jsonl"
        ),
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if duplicate_metadata_ids or invalid_rows or missing_video_ids:
        raise SystemExit("test_v2 manifest build has metadata or video errors")
    if not count_matches or not id_set_matches:
        raise SystemExit("test_v2 manifest does not match the frozen 100-ID list")


if __name__ == "__main__":
    main()
