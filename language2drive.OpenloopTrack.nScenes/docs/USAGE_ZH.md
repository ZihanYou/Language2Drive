# nuScenes 开环轨迹评测使用说明

本目录对应论文的 nuScenes 开环任务，使用 75 个 PCU（15 dev、60 heldout）。评分名称和尺度为 SemSISR、SemCFPS、SCS、PCS（0–100），并提供 SRG、NBD、轨迹差异和覆盖率。

## 一键部署与论文复算

```sh
sh scripts/setup.sh --with-assets
sh scripts/reproduce.sh
```

需先安装 conda，并具有私有数据集 `Mike087/Language2Drive` 的读取权限。通过凭据环境设置 `HF_TOKEN` 后可一键运行；令牌不会写入代码或资产。环境安装使用清华源；资产下载默认使用 `https://hf-mirror.com` 并关闭环境代理。约 14 MB 的资产包含冻结输入、评测契约、评分地图和四个模型的预测。复算只用 CPU，不需重新下载权重或运行 GPU 推理。

若已有兼容环境，可以直接运行：

```sh
python scripts/download_assets.py --prompt-token
python scripts/reproduce_paper.py --check-paper --output-dir outputs/my_replay
```

输出包括表 9 的 CSV/Markdown、完整精度指标、置信区间、逐 PCU 结果、有效预测覆盖率和文件哈希。`--check-paper` 仅在计算完成后检查论文展示值，不参与评分。重复执行请使用新的输出目录。

## 接入模型

实现 `predict(public_input, instruction, seed)`，返回 3 秒内每 0.25 秒一个点，共 12 个二维坐标。坐标为 anchor 时刻自车坐标，x 向前、y 向左，参考点为 `ego_pose_origin`。公开推理接口不传入私有契约或 witness。

通过 `run_model.py` 或支持断点恢复的 `model_eval/run_resumable.py` 运行；使用 `evaluate_predictions.py` 评分。默认 headline 为 heldout canonical、seed 0；另外三个改写和 masked/shuffled 分别报告，四表述均值作为单独补充。完整命令见根目录 README。

实际模型推理需自行取得 nuScenes 原始传感器数据及对应模型代码/权重，设置 `L2D_NUSCENES_ROOT`。启动 GPU 推理前检查空闲显存，选择满足模型需要的空闲卡。

数据与代码采用各自许可，见 `NOTICE.md`。本目录的文件和脚本不会修改 GitHub 或 Hugging Face 仓库的可见性。
