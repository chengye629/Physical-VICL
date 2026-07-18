# Physical-VICL

This repository evaluates whether video-generation models can use a demonstration video as in-context evidence for predicting the physical evolution of a new scene.

## Repository layout

```text
Physical-VICL/
├── configs/paths.example.yaml       # shareable path template
├── manifests/example.yaml           # one inference-item example
├── src/physical_vicl/inference/     # shared model adapter code
└── scripts/
    ├── prepare/                     # build condition-specific manifests
    ├── inference/                   # Video-As-Prompt and UniVideo entry points
    └── evaluate/                    # automatic judge entry points
```

Keep reusable implementation in `src/physical_vicl/` and keep `scripts/` as thin command-line entry points. Real datasets, generated videos, local path configs, checkpoints, and full manifests are intentionally excluded from Git.

## Next step

Run [`physiq_prelim`](https://huggingface.co/datasets/Vincwng/Physical-ICL/tree/main/data/physiq_prelim) with:

- [Video-As-Prompt](https://github.com/bytedance/Video-As-Prompt)
- [UniVideo](https://github.com/KlingAIResearch/UniVideo), using its in-context video-generation path

For every query, give the model one video from that task's `demos/` directory, the query `init_frame.png`, and the prompt paired with that demo condition. Generate a continuation of the **query image**, not of the demo. The initial benchmark is inference-only; no training or fine-tuning is needed.

## Dataset

Place the downloaded data at `data/physiq_prelim/`:

```text
data/physiq_prelim/
├── summary.json
├── case_summary.csv
└── gt_data/task_xxxx/episode_0001/
    ├── video.mp4                    # query ground truth: evaluation only
    ├── demos/<type>_demo_XX.mp4     # video context
    └── prompt/
        ├── init_frame.png           # query image condition
        ├── prompt.txt
        ├── no_demo_prompt.txt
        └── <demo_name>_<mode>_prompt.txt
```

`summary.json` is the source of truth. Its 66 query tasks currently contain:

| Demo type | Tasks | Meaning |
| --- | ---: | --- |
| `good` | 43 | Relevant, compatible physical outcome |
| `weak` | 16 | Partially related or weaker evidence |
| `opposite` | 13 | Relevant but contrastive outcome |
| `irrelevant` | 66 | Unrelated negative control |
| `bad` | 53 | Physically incorrect counterexample |

Not every task has every type. Only run demos listed in each case's `demos` / `available_demo_types`. See [`data/physiq_prelim/README.md`](data/physiq_prelim/README.md) for more details.

## Reading `summary.json`

| Field | Use |
| --- | --- |
| `case_id`, `task_name`, `episode_name` | Stable identifiers |
| `image` | Query initial frame |
| `prompt` | Plain query description |
| `gt_path` | Ground truth; evaluation only, never model context |
| `demos` | Demo paths and relation metadata |
| `available_demo_types` | Valid conditions for this query |
| `prompt_no_demo` | Baseline prompt |
| `prompt_<demo>_follow` | Generic instruction to use the demo |
| `prompt_<demo>_rule` | Instruction to infer and transfer its physical rule |
| `prompt_<demo>_typed` | Instruction explicitly stating its type |

Paths are repository-relative. Do not infer pairings from filenames: use each `demo_path` and its matching prompt key. Skip absent fields instead of synthesizing prompts; `bad` cases may only have `typed`.

## Prepare inference datasets by condition

The source dataset is organized by query task, but inference should not run directly over this mixed layout. Before launching either model, export a separate inference dataset for every demo type and prompt variant in the experiment matrix:

```text
data/inference_sets/
├── no_demo/
├── good_follow/
├── good_rule/
├── weak_typed/
├── opposite_typed/
├── irrelevant_follow/
└── bad_typed/
```

Each item should contain `case_id`, `task_name`, `init_frame`, `demo_path` (null for `no_demo`), `prompt_key`, the full generation `prompt`, and `gt_path` for later evaluation. Build every item from `summary.json`; never borrow a demo or prompt from another task to fill a missing condition.

For each condition, iterate over `summary.json`, select valid demos and prompt keys, verify all input paths, and write a JSONL or equivalent manifest. Report included, skipped, and invalid cases, and assign stable item IDs so both models run exactly the same examples. Run inference separately on each derived dataset; this makes resuming jobs, assigning GPUs, comparing coverage, and aggregating by condition less error-prone.

## Experiment matrix

Use identical cases, demos, prompts, seeds, sample counts, duration policy, FPS, and resolution policy for both models.

| Condition | Video context | Prompt |
| --- | --- | --- |
| `no_demo` | None | `prompt_no_demo` |
| `good` | Every available good demo | Run both matching `prompt_*_follow` and `prompt_*_rule` |
| `weak` | Every available weak demo | Matching `prompt_*_typed` |
| `opposite` | Every available opposite demo | Matching `prompt_*_typed` |
| `irrelevant` | Every available irrelevant demo | Matching `prompt_*_follow` |
| `bad` | Every available bad demo | Matching `prompt_*_typed` |

These prompt choices define the main experiment. Do not run other prompt variants unless an additional ablation is explicitly requested.

The no-demo baseline still uses the same query initial frame. If a model requires a video, use its native image-to-video path or a documented null-video condition; do not repeat the query image to create a fake demo.

Recommended order:

1. Export and validate the separate condition-specific inference datasets.
2. Implement one adapter per model.
3. Smoke-test one task with `no_demo`, `good`, and `irrelevant`.
4. Run all 66 no-demo baselines.
5. Run the complete demo matrix with the prompt variants specified above.
6. Score and aggregate the results.

## Model adapter contract

```python
generate(
    init_frame: str,
    prompt: str,
    demo_video: str | None,
    output_path: str,
    seed: int,
) -> None
```

- **Video-As-Prompt:** demo -> reference/video prompt; query frame -> reference image; selected prompt -> text.
- **UniVideo:** demo + query frame + selected prompt -> in-context video-generation inputs. Adapt the official `in_context_video_gen` example and preserve its input ordering.
- **No-demo:** query frame + `prompt_no_demo`, without ground-truth frames or demo-derived images.

Keep model-native preprocessing unless it changes semantic input. Record resizing, cropping, frame sampling, truncation, and prompt rewriting.

## Outputs

```text
outputs/generations/<model>/<prompt_mode>/<demo_type>/<task_name>/
└── <demo_name-or-no_demo>/seed_<seed>/
    ├── video.mp4
    └── metadata.json
```

Metadata must include `model`, `checkpoint`, `case_id`, `task_name`, `demo_type`, `demo_path`, `init_frame`, `prompt_key`, full `prompt`, `seed`, and all generation settings. Save checkpoint and repository revisions when possible. Never overwrite another configuration. Log failed attempts with inputs, settings, and exceptions so runs can resume.

## Evaluation

The main result is the paired change from the same model's no-demo baseline:

```text
delta(condition) = score(query, condition) - score(query, no_demo)
```

Evaluate physical plausibility, physical-outcome correctness, prompt consistency, query-scene preservation, temporal/visual quality, and unwanted following of irrelevant, opposite, or bad demos. Use `gt_path` only for evaluation. Match the query `duration` where supported by both models.

For the main physics judge, uniformly sample each generated video at **4 FPS** and pass the frames in temporal order to **Qwen3-8B-VL-Instruct or a comparable vision-language model**. Also provide the exact generation prompt / task instruction used for that output: it gives the judge the intended event and relevant physical context. Do not provide the demonstration video or ground-truth query video to the judge.

Use this general-purpose judge prompt:

```text
You are a strict, calibrated evaluator of the PHYSICAL REALISM of AI-generated videos.

You are given the generation prompt / task instruction and uniformly sampled frames from one generated video in temporal order. Use the prompt to understand the intended scene, event, and physical interaction. Judge whether the visible video realizes that event with physically realistic behavior. Focus the score on physical realism rather than visual style or general text-video similarity. Be conservative: reserve 5 for clearly flawless and temporally consistent physics, and 1 for severe or pervasive physical violations.

Generation prompt / task instruction:
{generation_prompt}

Task: Judge the PHYSICAL REALISM of this AI-generated video with respect to the intended event described above.

Criteria (address each criterion and note any other physical issue):

1. Entity and material consistency: objects, bodies, and materials remain structurally coherent; they do not inexplicably melt, fuse, split, stretch, or deform.

2. Scene and object continuity: the environment and objects remain temporally stable, without unexplained flicker, teleportation, morphing, duplication, disappearance, or identity changes.

3. Interaction and contact realism: collisions, support, grasping, containment, cutting, tearing, pouring, and other contacts behave plausibly. Objects do not interpenetrate, pass through barriers, float without support, or react before contact.

4. Motion and physical-law consistency: motion follows applicable gravity, inertia, momentum, friction, rigidity/deformation, and fluid behavior. Causes precede effects, trajectories are continuous, and outcomes are physically plausible.

Score (integer 1-5): 1 = gross or pervasive violations; 2 = major violations; 3 = noticeable local inconsistencies; 4 = minor issues only; 5 = no visible violation.

Reason first, then score. Output a JSON object with two keys: `reasoning` (string) and `physical_adherence` (integer 1-5).
```

Existing judge entry points are:

```bash
python scripts/evaluate/run_videoscore2.py --help
python scripts/evaluate/run_videophy2.py --help
```

They expect an older type-grouped layout and do not currently discover `outputs/generations/`; update their collectors or add a conversion step first.

Report model x demo type, model x demo type x prompt mode, physical-category slices, and per-task deltas. Include representative qualitative comparisons for all five demo types.

## Reproducibility rules

- Use identical seeds across models and conditions; three seeds is a reasonable first full run if compute permits.
- Never expose `gt_path` to a generator.
- Never replace missing demos or prompt variants with another task's data.
- Do not count unavailable conditions as failures.
- Validate every input path before expensive inference.

## Definition of done

Both adapters accept the intended inputs; every valid condition with its specified prompt variant and every no-demo baseline has been attempted; successful outputs have complete metadata; failures are resumable; and both models have scores, paired deltas, and a qualitative report.
