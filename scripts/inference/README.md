# Model inference

The adapters in this directory connect one canonical Physical-VICL manifest item to
the official model repository. They do not vendor model code or weights.

| Entry point | Conditioning used for the with-demo arm | Status |
| --- | --- | --- |
| `run_vap.py` | native reference video + query initial frame | validated locally |
| `run_vino.py` | native `tiv2v` reference video + query image | interface-validated |
| `run_univideo.py` | sampled demo frames + query image (`multiid`) | interface-validated |
| `run_vace.py` | `animate_anything`: demo motion video + query image | experimental adapter |

See [`docs/video_icl_models.md`](../../docs/video_icl_models.md) for environments,
weights, exact commands, limitations, and the verification matrix.
