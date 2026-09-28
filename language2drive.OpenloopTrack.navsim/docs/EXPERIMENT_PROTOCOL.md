# Five-model evaluation protocol

Status: frozen before the first complete model run. Benchmark: `l2d-offline-navsim` 1.0.0.

## Evaluation unit and aggregation

- 38 PCUs, two candidate instructions per PCU.
- One canonical wording and three approved English paraphrases per arm.
- 304 queries per model (`38 × 2 × 4`).
- Each instruction is passed verbatim to the model. No private contract, witness trajectory,
  sidecar, map, future frame, or ground-truth future is available to inference.
- Decoding uses seed 42. Deterministic decoding is used when the official interface supports
  it. AutoVLA and diffusion planners retain their released sampling interfaces with the seed
  reset before every query.
- Rewrites are averaged equally inside a PCU, then PCUs are averaged equally. Metrics are
  SemSISR, SemCFPS, SCS, PCS, and paired imbalance, as frozen in `METRIC_SPEC.md`.

## Common trajectory output

All outputs are converted to eight rear-axle poses in the anchor ego frame at
`[0.5, 1.0, ..., 4.0]` seconds. Model-provided headings are retained where available; otherwise
headings are computed from successive positions. Parse failures remain invalid predictions and
receive zero task completion.

## Model adapters

| Model | Released input retained | Language injection | Output conversion |
|---|---|---|---|
| Qwen-Drive-1.0-4B | 4 frames × front/front-left/front-right; 16-point interpolation of the observed 1.5 s history | `Language driving instruction: [...]` in the released scene prompt | RL planner, reasoning mode, one sample; 10 Hz points 5,10,...,40 |
| AutoVLA | 4 frames × front/front-left/front-right; current velocity and acceleration | Released `driving instruction` field | First eight released 0.5 s action-token poses |
| DriveMA-2B | Current front-left/front/front-right; observed history, velocity, acceleration | Released `Current high-level intent` field | Released two-turn decision/trajectory protocol; 1 Hz points linearly sampled at 0.5 s |
| ReCogDrive | Current front image; four observed poses; current velocity and acceleration | Full text replaces the discrete command phrase in the released VLM prompt | Released 8-pose DDIM planner output |
| OneVL | Current front image; three historical poses; current velocity and acceleration | Full text in the released `Command` field | Released latent-CoT generation of eight 0.5 s poses |

Qwen-Drive and ReCogDrive also require a non-language command tensor. It is fixed to the
straight/neutral state for every arm and rewrite, so the evaluated language text is the only
candidate-dependent input.

## Released artifacts

| Model component | Hugging Face revision |
|---|---|
| Qwen/Qwen-Drive-1.0-4B | `28484089a7cc8c335cf5089fb0745cf7c49b6eaa` |
| Qwen/Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Zewei-Zhou/AutoVLA | `a7d7ba3ed7529b248d2694c2defa31b35208340f` |
| zwc2003/DriveMA-2B | `d5f4905d2032da943a7ae8f22d5b2b8efbe4b0f7` |
| owl10/ReCogDrive-VLM-2B | `16873acca08e3c04ab229b3d973f39aeba9db68d` |
| owl10/ReCogDrive-2B-RL | `7346ede73f03ed577811024f1a5d16fb063a47b9` |
| xiaomi-research/OneVL_NAVSIM | `c1e893e9a03b8312f9328d2b51267836a9f7bf8a` |

All downloads use `hf-mirror.com` with every proxy environment variable removed. Per-repository
download manifests record expected byte counts and completeness.

## Code revisions

- Qwen-Drive: `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`
- AutoVLA: `ba34eed74ce6729e7986592d0e66cbaca397b4fa`
- DriveMA: `de20e6c64878bc3413d26d69362c3be9b3a91533`
- ReCogDrive: `6b8d8f5e01346c71094651c81dcaf66405dbc04e`
- OneVL: `f4e6f05572b2f66611259bcd5563529833f30469`

PyTorch 2.5.1+cu124 is used on NVIDIA A100 40 GB GPUs. Qwen-Drive uses Transformers 5.14.1,
OneVL uses 4.57.0, and the legacy adapters use the released-compatible 4.49 development build.
