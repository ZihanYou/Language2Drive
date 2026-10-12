---
pretty_name: Language2Drive
language:
- en
license: other
license_name: component-specific
license_link: https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.carla/THIRD_PARTY_NOTICES
tags:
- autonomous-driving
- vision-language-action
- instruction-following
- counterfactual-evaluation
- carla
- realsim
- nuscenes
- navsim
- waymo
- benchmark
- language2drive
---

<div align="center">

# 🚘 Language2Drive

### Do Vision-Language-Action Models Really Follow the Instructions in Autonomous Driving?

**Same scene. Same seed. Different instructions.**

[![Code](https://img.shields.io/badge/Code-GitHub-202c39?style=flat-square&logo=github)](https://github.com/ZihanYou/Language2Drive)
[![Benchmark](https://img.shields.io/badge/Benchmark-5_Backends-2b7a78?style=flat-square)](#benchmark-tracks)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Training_Free_Baselines-68549c?style=flat-square)](#main-results--leaderboard)
[![Dataset](https://img.shields.io/badge/Dataset-Versioned_Release_Coming_Soon-e6a43a?style=flat-square)](#release-roadmap)

**👥 Authors**

Zihan You<sup>1,2,*,†</sup>, Zehua Fan<sup>3,*</sup>, Wenqi Lyu<sup>4,*</sup>, Aoqi Wang<sup>5,*</sup>,<br>
Ni Wang<sup>6</sup>, Yifei Yang<sup>7</sup>, Linge Zhao<sup>8</sup>, Huan Li<sup>9</sup>, Sining Ang<sup>10</sup>, Yuguang Yang<sup>1</sup>,<br>
Jueyu Chen<sup>10,11</sup>, Xiangyu Li<sup>12</sup>, Qianshan Wei<sup>13</sup>, Jieyuan Pei<sup>1,14,15</sup>, Yuxin Wang<sup>16</sup>, Zhiwen Chen<sup>15</sup>,<br>
Jiaqi Li<sup>17</sup>, Wenxuan Song<sup>15</sup>, Yanyuan Qiao<sup>18</sup>, Benjin Zhu<sup>19</sup>, Haoang Li<sup>15</sup>, Yan Wang<sup>1,‡</sup>

**🏛️ Affiliations**

<sup>1</sup> THU · <sup>2</sup> SEU · <sup>3</sup> SJTU · <sup>4</sup> Adelaide · <sup>5</sup> BISTU · <sup>6</sup> Amazon · <sup>7</sup> BJTU<br>
<sup>8</sup> WHU · <sup>9</sup> HUST · <sup>10</sup> USTC · <sup>11</sup> NIO · <sup>12</sup> UT Austin · <sup>13</sup> CAS<br>
<sup>14</sup> HiThink · <sup>15</sup> HKUST(GZ) · <sup>16</sup> AMD · <sup>17</sup> HKU · <sup>18</sup> EPFL · <sup>19</sup> Li Auto

<sub>* Core contributors &nbsp; † Project Lead &nbsp; ‡ Corresponding author</sub>

<img src="https://raw.githubusercontent.com/ZihanYou/Language2Drive/main/docs/assets/framework.png" alt="Language2Drive: paired counterfactual instructions, four scopes and five evaluation backends" width="100%">

**4 instruction scopes · 3 runtime groups · 5 evaluation backends**

</div>

## 🧭 Benchmark in 30 seconds

Language2Drive evaluates whether changing a natural-language instruction changes the feasible behavior selected and executed by a driving VLA. Its **Paired Counterfactual Unit (PCU)** fixes the initial driving scene and changes only the instruction, separating individual task completion from completion of both counterfactual alternatives.

- **Matched pairs:** same scene, initial state, non-language inputs and simulator seed; different mutually exclusive instructions.
- **Four scopes:** atomic actions, persistent behaviors, scene-grounded navigation and long-horizon instruction programs.
- **Five backends:** CARLA and RealSim closed loop; nuScenes, NAVSIM and WOD-E2E open loop.
- **Training-free evaluation:** all reported baseline systems use the training-free protocol.
- **Adapt once:** the unified SDK connects a model to all five native evaluators through one trajectory interface.

## 🔀 PCU: the core evaluation unit

| Shared across the pair | Changed between the arms | Evaluated |
|---|---|---|
| Scene, initial state, non-language inputs and seed | Active natural-language instruction | Requested behavior and rejection of the opposite contract |
| Checkpoint and observation configuration | Feasible target behavior | Native safety, progress, deadline and stability requirements |

At a shared junction, one arm can request **“Turn left”** and the other **“Go straight.”** Each arm starts independently with reset model memory. Strict paired success requires completing both requests. The model receives public observations and the active instruction; geometric targets, evaluator contracts and the opposite-arm instruction remain with the evaluator.

| Scope | What the instruction requires | Example task |
|---|---|---|
| **A · Atomic action** | Select a local maneuver | Turn, change lane or park |
| **B · Persistent behavior** | Maintain a constraint over time | Speed band, lane choice or following behavior |
| **C · Scene-grounded navigation** | Resolve a destination or relation in the observed scene | Parking bay, named landmark or obstacle-relative destination |
| **D · Long-horizon instruction program** | Execute stages in the specified order | Conditional destinations and multi-stage maneuvers |

### 🏗️ Benchmark construction

<img src="https://raw.githubusercontent.com/ZihanYou/Language2Drive/main/docs/assets/process_png.png" alt="Source curation, geometric grounding, paired specification, compilation and validation" width="100%">

Source scenes are grounded geometrically, paired with feasible instructions, compiled into executable contracts and validated before evaluation. Paraphrases test sensitivity to equivalent wording.

## 🛣️ Benchmark Tracks

| Track | Evaluation | Manuscript / frozen scope | Model output | Native guide |
|---|---|---|---|---|
| **CARLA-L2D** | Closed loop; six purpose-designed maps | **64 paper PCUs**, A–D; 384 core rollouts/system, three expressions | Ego XY → declared controller | [CARLA](https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.carla/README.md) |
| **RealSim** | Closed loop; AlpaSim / NuRec | **30 PCUs / 23 scenes**, A–B; 60 slots/system, seed 1701 | 6 s / 60 points, 10 Hz | [RealSim](https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.RealSim/README.md) |
| **nuScenes** | Open loop | **75 release PCUs**: 15 dev + 60 held-out; paper uses held-out canonical | 3 s / 12 points, 4 Hz | [nuScenes](https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.OpenloopTrack.nScenes/README.md) |
| **NAVSIM** | Open loop | **38 PCUs / 19 recordings**, four wordings | 4 s / 8 points, 2 Hz | [NAVSIM](https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.OpenloopTrack.navsim/README.md) |
| **WOD-E2E** | Open loop | **241 P0 scored PCUs**: 76 STOP/GO + 165 speed-choice | 5 s / 20 points, 4 Hz | [WOD-E2E](https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.OpenloopTrack.WOD-E2E/README.md) |

**Scopes are instruction categories; tracks are evaluation environments.** Counts are not interchangeable. CARLA's source inventory is broader than the 64-PCU paper cohort; record tier, split and seed. The SDK defaults to `base` and evaluates dev/test separately. CARLA PSM uses 24 additional routes and requires the native diagnostic evaluation. WOD's public workload can include construction candidates beyond the frozen 241-PCU scored subset.

## 📦 Resources in this repository

This Hugging Face repository hosts benchmark workload definitions, replay bundles and runtime resources. The official versioned dataset and model/checkpoint releases are coming soon.

| Resource group | Browse / download | Contents |
|---|---|---|
| **CARLA-L2D** | [CARLA resources](https://huggingface.co/datasets/Mike087/Language2Drive/tree/main/language2drive.carla) | Benchmark and runtime archives, manifests, checksums and integration sources |
| **RealSim** | [RealSim resources](https://huggingface.co/datasets/Mike087/Language2Drive/tree/main/langauge2drive.RealSim) | PCUs, public workload, run plan, evaluator sidecar and runtime configuration |
| **nuScenes** | [nuScenes resources](https://huggingface.co/datasets/Mike087/Language2Drive/tree/main/language2drive.OpenloopTrack.nScenes) | Asset manifest and `nuscenes-paper-1.0-replay.zip` |
| **NAVSIM / WOD-E2E** | [NAVSIM code](https://github.com/ZihanYou/Language2Drive/tree/main/language2drive.OpenloopTrack.navsim) · [WOD-E2E code](https://github.com/ZihanYou/Language2Drive/tree/main/language2drive.OpenloopTrack.WOD-E2E) | Native evaluator guides, source-data requirements and release profiles |

The historical RealSim directory is spelled `langauge2drive.RealSim`; use that exact prefix when downloading.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Mike087/Language2Drive",
    repo_type="dataset",
    local_dir="./language2drive-resources",
    allow_patterns=["langauge2drive.RealSim/**"],
)
```

Select `language2drive.carla/**` or `language2drive.OpenloopTrack.nScenes/**` for the other available resource groups. Repository authorization uses your Hugging Face account. Model checkpoints are obtained from their original publishers. Source scenes follow the access terms of their upstream datasets.

## 🗂️ Dataset / Evaluation Format

| Layer | Contents | Consumer |
|---|---|---|
| Public observation | Cameras, observed ego state/history and active instruction | Model adapter |
| Trajectory request | Track, ego frame, reference point, timestamp grid and model seed | Model adapter |
| Evaluator metadata | PCU/arm identity, contracts, pair binding and expected inventory | Track runner / scorer |
| Submission | Native trajectory records | Native scorer |
| Result | Scores, scheduled coverage, provenance and protocol | Leaderboard / analysis |

The SDK accepts **+x forward, +y left, metres**, with explicit frame, reference point and timestamps. For example, a NAVSIM output uses the rear axle and eight points at 2 Hz:

```json
{
  "coordinate_frame": "ego_at_anchor",
  "reference_point": "rear_axle",
  "timestamps_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
  "xy_m": [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]]
}
```

nuScenes uses `ego_pose_origin`; RealSim uses `rig_origin`. The [adapter guide](https://github.com/ZihanYou/Language2Drive/blob/main/docs/ADAPTER.md) documents each output contract, and the [protocol](https://github.com/ZihanYou/Language2Drive/blob/main/docs/PROTOCOL.md) specifies native submission files and cohort identities.

## 🏆 Main Results / Leaderboard

**All reported systems use the training-free protocol.** Values below are transcribed from the supplied manuscript; all scores are **0–100, higher is better**. Systems are compared **within each track and profile**. [Project-page leaderboard source](https://github.com/ZihanYou/Language2Drive/blob/main/docs/index.html) · [Machine-readable results](https://github.com/ZihanYou/Language2Drive/blob/main/docs/data/results.json) · [Submission guide](https://github.com/ZihanYou/Language2Drive/blob/main/docs/LEADERBOARD.md).

Click a model name to read its paper. Explore the [interactive metric guide](https://ZihanYou.github.io/Language2Drive/#metric-explorer) for paired success, graded compliance and the L2D score.

<details open>
<summary><b>📊 CARLA-L2D</b></summary>

paper: 64 PCUs · 3 expressions · seed 1.

| System | SISR | CFPS | SemISR | Sem-CFPS | SCS | PCS | PSM | DrivingScore | L2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Alpamayo 1.5](https://arxiv.org/abs/2511.00088 "Alpamayo family paper") · [model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | 8.07 | 1.04 | 42.19 | 30.73 | 26.18 | 12.50 | 70.10 | 62.69 | 12.20 |
| [SimLingo](https://arxiv.org/abs/2503.09594) | 7.81 | 1.04 | 48.70 | 35.94 | 23.51 | 11.94 | 62.30 | 65.04 | 12.15 |
| [LMDrive](https://arxiv.org/abs/2312.07488) | 12.24 | 0.00 | 29.17 | 18.23 | 21.45 | 9.26 | 88.84 | 50.12 | 10.39 |
| [OneVL](https://arxiv.org/abs/2604.18486) | 0.52 | 0.00 | 25.00 | 3.65 | 16.69 | 7.05 | 77.72 | 50.69 | 4.87 |
| [Qwen-Drive-1.0-4B](https://arxiv.org/abs/2609.00111) | 1.82 | 0.00 | 15.62 | 3.12 | 17.05 | 7.09 | 76.68 | 43.67 | 4.31 |
| [Impromptu-VLA](https://arxiv.org/abs/2505.23757) | 0.26 | 0.00 | 19.79 | 6.25 | 14.48 | 4.61 | 77.43 | 42.02 | 3.87 |

</details>

<details>
<summary><b>📊 RealSim</b></summary>

realsim-paper-1.0: 30 PCUs · seed 1701.

| System | SISR | CFPS | SemISR | Sem-CFPS | SCS | PCS | Safety | DrivingScore | L2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Alpamayo2-Super](https://arxiv.org/abs/2511.00088 "Alpamayo family paper") · [model card](https://huggingface.co/nvidia/Alpamayo2-Super) | 13.33 | 0.00 | 18.33 | 0.00 | 21.43 | 0.00 | 93.33 | 97.10 | 9.22 |
| [Alpamayo 1.5](https://arxiv.org/abs/2511.00088 "Alpamayo family paper") · [model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | 11.67 | 0.00 | 15.00 | 0.00 | 18.12 | 1.11 | 100.00 | 97.17 | 8.38 |
| [SimLingo](https://arxiv.org/abs/2503.09594) | 6.67 | 3.33 | 13.33 | 6.67 | 16.67 | 10.00 | 68.33 | 83.09 | 7.58 |
| [AutoVLA](https://arxiv.org/abs/2506.13757) | 1.67 | 0.00 | 8.33 | 6.67 | 9.73 | 6.67 | 41.67 | 76.94 | 3.36 |
| [Qwen-Drive-1.0-4B](https://arxiv.org/abs/2609.00111) | 0.00 | 0.00 | 1.67 | 0.00 | 4.44 | 0.00 | 43.33 | 69.16 | 0.92 |
| [Impromptu-VLA](https://arxiv.org/abs/2505.23757) | 0.00 | 0.00 | 1.67 | 0.00 | 3.71 | 0.00 | 30.00 | 68.85 | 0.78 |
| [OneVL](https://arxiv.org/abs/2604.18486) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 43.33 | 43.82 | 0.44 |
| [ReCogDrive](https://arxiv.org/abs/2506.08052) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 33.33 | 43.57 | 0.38 |

</details>

<details>
<summary><b>📊 nuScenes</b></summary>

v1.0-test · 60 held-out PCUs · canonical.

| System | SemSISR | SemCFPS | SCS | PCS |
|---|---:|---:|---:|---:|
| [Impromptu-VLA](https://arxiv.org/abs/2505.23757) | 15.83 | 0.00 | 20.21 | 0.00 |
| [UniDriveVLA](https://arxiv.org/abs/2604.02190) | 16.67 | 0.00 | 25.00 | 0.00 |
| [OpenDriveVLA-0.5B](https://arxiv.org/abs/2503.23463) | 13.33 | 3.33 | 19.06 | 4.30 |
| [OmniDrive](https://arxiv.org/abs/2405.01533) | 7.50 | 0.00 | 29.24 | 8.61 |

</details>

<details>
<summary><b>📊 NAVSIM</b></summary>

paper historical scorer · 38 PCUs · 4 wordings.

| System | SemSISR | SemCFPS | SCS | PCS |
|---|---:|---:|---:|---:|
| [Qwen-Drive-1.0-4B](https://arxiv.org/abs/2609.00111) | 0.000 | 0.000 | 5.472 | 0.000 |
| [AutoVLA](https://arxiv.org/abs/2506.13757) | 3.947 | 0.000 | 8.172 | 0.000 |
| [DriveMA-2B](https://arxiv.org/abs/2605.31271) | 8.553 | 0.000 | 11.244 | 2.242 |
| [ReCogDrive](https://arxiv.org/abs/2506.08052) | 0.000 | 0.000 | 4.400 | 0.000 |
| [OneVL](https://arxiv.org/abs/2604.18486) | 3.618 | 0.000 | 8.428 | 0.000 |

</details>

<details>
<summary><b>📊 WOD-E2E</b></summary>

P0 · 241 PCUs · canonical.

| System | SemSISR | SemCFPS | SCS | PCS |
|---|---:|---:|---:|---:|
| [DriveMA-2B](https://arxiv.org/abs/2605.31271) | 38.38 | 1.24 | 49.33 | 7.88 |
| [Fast-dDrive](https://arxiv.org/abs/2605.23163) | 34.85 | 0.00 | 46.42 | 5.27 |
| [Qwen-Drive-SFT](https://arxiv.org/abs/2609.00111) | 34.44 | 0.00 | 50.58 | 9.37 |
| [Qwen-Drive-RL](https://arxiv.org/abs/2609.00111) | 35.06 | 0.00 | 49.58 | 7.58 |

</details>

The CARLA table includes the six principal systems; the paper's Senna high-level bridge is a separate diagnostic configuration. **NAVSIM numbers use the historical manuscript scorer.** New SDK runs use `score_corrected.py`; keep corrected evaluations in a separate leaderboard profile. Full protocol, coverage and failure denominators must accompany submissions. Rankings follow each track’s native metrics and evaluation profile.

### 📐 Reading the metrics

**SISR / CFPS** measure strict individual / paired completion. **SCS / PCS** measure graded individual / paired compliance. Closed-loop **SemISR / Sem-CFPS** are relaxed proxies; open-loop **SemSISR / SemCFPS** apply exclusive semantic success. Wording robustness is evaluated using equivalent paraphrases. **L2D** combines the native language and driving score families with the track's nonlinear aggregation. Full definitions are in the [evaluation protocol](https://github.com/ZihanYou/Language2Drive/blob/main/docs/PROTOCOL.md).

## 🚀 Evaluate your model

Install the coordinator from the source repository or a release wheel, select your runtime groups, and implement one model adapter:

```sh
git clone https://github.com/ZihanYou/Language2Drive.git
cd Language2Drive
python -m pip install -e '.[openloop]'
language2drive init --workspace ./l2d-workspace
language2drive setup --workspace ./l2d-workspace --tracks all --runtime --download
```

`setup` selects official Hugging Face / PyPI endpoints when reachable and mirror / Tsinghua endpoints otherwise. Authenticated benchmark downloads use the official Hugging Face endpoint. CARLA and RealSim use separate simulator environments. Licensed source data, GPU services and model checkpoints are configured according to the selected backend.

- [Quick start and all five native reproduction commands](https://github.com/ZihanYou/Language2Drive#quick-start)
- [Full GPU evaluation guide](https://github.com/ZihanYou/Language2Drive/blob/main/docs/GPU_EVALUATION.md)
- [Installation and resource setup](https://github.com/ZihanYou/Language2Drive/blob/main/docs/INSTALLATION.md)
- [Leaderboard submission guide](https://github.com/ZihanYou/Language2Drive/blob/main/docs/LEADERBOARD.md)
- [Interactive project preview, paired cases and demonstration film](https://github.com/ZihanYou/Language2Drive/blob/main/docs/SHOWCASE.md)

## 🗓️ Release Roadmap

| Status | Deliverable |
|:---:|---|
| ✅ | Benchmark source, five native evaluators and a unified model-adapter SDK |
| ✅ | Wheel build and selectable environment/resource setup |
| ✅ | Training-free baseline tables and paired case visualizations |
| ✅ | Evaluation resources in this Hugging Face repository |
| ⏳ | Official versioned dataset release |
| ⏳ | Official model and checkpoint release |

## 📚 Citation

If Language2Drive supports your research, please cite our paper.

```bibtex
@misc{language2drive,
  title = {Language2Drive: Do Vision-Language-Action Models Really Follow the Instructions in Autonomous Driving?},
  author = {You, Zihan and
           Fan, Zehua and
           Lyu, Wenqi and
           Wang, Aoqi and
           Wang, Ni and
           Yang, Yifei and
           Zhao, Linge and
           Li, Huan and
           Ang, Sining and
           Yang, Yuguang and
           Chen, Jueyu and
           Li, Xiangyu and
           Wei, Qianshan and
           Pei, Jieyuan and
           Wang, Yuxin and
           Chen, Zhiwen and
           Li, Jiaqi and
           Song, Wenxuan and
           Qiao, Yanyuan and
           Zhu, Benjin and
           Li, Haoang and
           Wang, Yan},
  howpublished = {Research manuscript},
  url = {https://github.com/ZihanYou/Language2Drive}
}
```

## 📄 License

Original Language2Drive code uses the [MIT License](https://github.com/ZihanYou/Language2Drive/blob/main/LICENSE). Dataset, simulator, runtime and third-party integration resources retain their respective licenses. See the [CARLA resource license](https://huggingface.co/datasets/Mike087/Language2Drive/blob/main/language2drive.carla/LICENSE), [third-party notices](https://huggingface.co/datasets/Mike087/Language2Drive/blob/main/language2drive.carla/THIRD_PARTY_NOTICES) and [RealSim notices](https://github.com/ZihanYou/Language2Drive/blob/main/language2drive.RealSim/NOTICE.md). This card preserves the repository's **component-specific** license designation. Manuscript figures are research materials and are separate from the code license.
