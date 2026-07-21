# Video-ICL model adapters

This branch adds a common inference path for four models that can condition video
generation on a demonstration or on a representation derived from it. The benchmark
item contract is:

```text
demo video (optional) + query init_frame + condition-matched prompt -> query continuation
```

The ground-truth query video is never passed to a generator. Build manifests first:

```bash
python scripts/prepare/build_manifests.py \
  --dataset-root /data/physiq_prelim \
  --output-root /work/manifests
```

## Verification status

| Model | Demo carrier | Physical-VICL status | Important limitation |
| --- | --- | --- | --- |
| VAP | native demo video through the MoT reference expert | end-to-end validated on RTX 4090 | 49 frames took about 19 min with sequential CPU offload |
| VINO | native video through official `tiv2v` | official schema and launcher validated; generation not run locally | editing semantics may preserve the demo scene instead of the query scene; inspect frame 0 and scene identity |
| UniVideo | 8 uniformly sampled demo frames through official `multiid` | official pipeline interface validated; generation not run locally | no native demo-video input in `multiid`; temporal information is compressed into frames |
| VACE | demo video plus query image through official `animate_anything` | experimental official-pipeline adapter; generation not run locally | designed for motion/appearance control, not a trained physics-ICL objective |

Only VAP should currently be described as locally end-to-end validated. The other three
adapters deliberately retain experimental labels until their first H20 smoke tests pass.

## 1. Video-As-Prompt (VAP)

Upstream: <https://github.com/bytedance/Video-As-Prompt>  
Weights: `ByteDance/Video-As-Prompt-CogVideoX-5B`

Use an isolated environment and install the repository's vendored Diffusers fork. Do
not create the venv with `--system-site-packages`; pin Torch inside it.

```bash
git clone https://github.com/bytedance/Video-As-Prompt.git /models/Video-As-Prompt
cd /models/Video-As-Prompt
python -m venv .venv
. .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ./diffusers
pip install -r requirements.txt
```

Run a with-demo item:

```bash
python scripts/inference/run_vap.py \
  --manifest /work/manifests/good_follow.jsonl --index 0 \
  --vap-root /models/Video-As-Prompt \
  --checkpoint /models/Video-As-Prompt/ckpts/Video-As-Prompt-CogVideoX-5B \
  --cpu-offload
```

Run the same command on `no_demo.jsonl` for the base CogVideoX-I2V control. The adapter
selects the MoT pipeline only when `demo_path` is non-null.

## 2. VINO

Upstream: <https://github.com/SOTAMak1r/VINO-code>  
Weights: `SOTAMak1r/VINO-weight`

Follow the upstream installation and weight download instructions. The adapter writes
the verified official task schema:

```json
{
  "task": "tiv2v",
  "caption": "<condition-matched Physical-ICL prompt>",
  "ref_image_paths": ["<query init frame>"],
  "ref_video_path": "<demo video>"
}
```

Launch or inspect the generated command without allocating GPUs:

```bash
python scripts/inference/run_vino.py \
  --manifest /work/manifests/good_follow.jsonl --index 0 \
  --vino-root /models/VINO-code --nproc-per-node 8 --dry-run
```

Remove `--dry-run` on the H20 machine. The no-demo manifest maps to official `i2v`.
Before a batch run, verify that output frame 0 and the continuing scene match the query
image rather than the demo; `tiv2v` is an editing path, so this is a required gate.

## 3. UniVideo

Upstream: <https://github.com/KlingAIResearch/UniVideo>  
Weights: `KlingTeam/UniVideo`

Install the upstream `environment.yml` and download the hidden variant. UniVideo's
released in-context generation mode is `multiid` (Image x N + Text), not native
Video + Image + Text. The adapter therefore samples eight demo frames, appends the
query initial frame, and calls the official pipeline. The no-demo arm uses native I2V.

```bash
python scripts/inference/run_univideo.py \
  --manifest /work/manifests/good_follow.jsonl --index 0 \
  --univideo-root /models/UniVideo \
  --transformer-checkpoint /models/UniVideo/ckpts/transformer.pt
```

Keep the sampling count identical across conditions and models when reporting an
ablation. The extracted frames are saved next to each result for auditability.

## 4. VACE

Upstream: <https://github.com/ali-vilab/VACE>  
Weights: `Wan-AI/Wan2.1-VACE-14B`

VACE is included as a control-transfer route. The with-demo adapter uses the official
`animate_anything` preprocessor with the demo video as the motion source and the query
initial frame as the target reference. The no-demo arm uses `frameref/firstframe`.

```bash
python scripts/inference/run_vace.py \
  --manifest /work/manifests/good_follow.jsonl --index 0 \
  --vace-root /models/VACE \
  --checkpoint /models/VACE/models/Wan2.1-VACE-14B \
  --dry-run
```

Install both `requirements.txt` and `requirements/annotator.txt`, plus the official
VACE annotator weights, before removing `--dry-run`. `salientbboxtrack` is the default
because it provides an explicit motion track; it may fail on multi-object physics cases.
Record such preprocessing failures rather than silently falling back to another demo.

## Smoke-test gate

For each model, run one `good_follow`, one `irrelevant_follow`, and the matching
`no_demo` item before a batch launch. Check:

1. frame 0 preserves the query image;
2. demo objects/background do not replace the query scene;
3. the output contains the requested physical event;
4. output metadata records model revision, seed, dimensions, frame count, and all input paths;
5. failures are resumable and do not overwrite another condition.

Use identical item IDs and seeds across the four models. Treat the generated videos,
not adapter startup, as the definition of an end-to-end pass.
