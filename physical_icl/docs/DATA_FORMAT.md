# Data formats

## Runtime annotation manifest

Minimum fields:

```json
{"sample_id": "...", "video_path": "/local/path/on/this/machine/video.mp4"}
```

Additional metadata may exist but Pass A does not use it in the prompt.

## Physics Card payload

One JSON file per sample. Successful payloads contain:

- `status = success`
- `observable/pass_a source`
- `physics_card`
- deterministic derived/audit fields

## Retrieval index

One row per eligible video. Important fields include:

- `sample_id`
- `process_family`
- `process_type`
- `process_resolution_level`
- `texts`
- `tokens`
- `source.video_path`

## Query-centric retrieval mapping

`retrievals_clean_v21.jsonl` contains one query per row with a `demos` list.

## Flat raw pair mapping

`pairs_clean_v21.jsonl` is the directed flattening of the query-centric mapping.

In the raw schema:

- `query_id` is the query sample ID;
- `sample_id` is the demo sample ID.

## Derived portable training mapping

`training/build_training_maps.py` writes:

- `training_pairs_1demo.jsonl`
- `query_demo_map.jsonl`
- `video_lookup.jsonl`

All exported paths use project-relative `*_video_relpath` fields.
