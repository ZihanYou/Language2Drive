# 一键创建环境与常见故障处理

本项目不分发现有服务器的 Python 虚拟环境。使用者在自己的机器上根据固定版本创建环境，避免绝对路径、系统库和 CUDA 差异造成不可移植的问题。

有两个不同入口：

| 需要做什么 | 命令 | 是否下载模型/启动实验 |
| --- | --- | --- |
| 离线评分、CPU 单元测试、合成示例 | `sh scripts/setup.sh` | 不下载模型，不启动实验 |
| 创建 AlpaSim + RealSim GPU 闭环所需的 Python 环境 | `sh scripts/setup_runtime.sh --install-rust` | 安装运行依赖，不下载权重，不启动实验 |

第二条命令不是“自动跑完 30 PCU”的声明。场景、模型权重、渲染器与驱动权限就绪后，才可按 [闭环运行接口](RUNTIME_SETUP.md) 启动服务和提交运行计划。

## 1. 系统准备

运行环境目标为 **Linux x86_64、Python 3.12**。启动安装脚本的 Python 可以为 3.10 或更新版本；脚本通过 uv 为闭环环境选择 Python 3.12。

建议至少准备 30 GB 可用空间用于 Python 依赖、工具链和安装缓存。模型、场景与渲染器需要另外预留空间。不要将“环境可安装”理解为所有模型都能在一张 24 GB 卡上运行；显存需求取决于模型和部署方式。

一次性系统依赖（Ubuntu 示例，需管理员授权）：

```sh
sudo apt-get update
sudo apt-get install -y git build-essential pkg-config python3-venv libglib2.0-0 libgl1
```

环境脚本自身不调用 sudo，不安装或升级系统 NVIDIA 驱动，不修改用户登录脚本，也不更改现有 Conda/实验环境。

真正运行 NuRec GPU 渲染器还需：

- NVIDIA 驱动支持 CUDA 12.8；固定上游版本要求驱动至少为 570 系列。
- Docker Engine、Compose 插件和 NVIDIA Container Toolkit。
- 当前用户可访问 Docker daemon。加入 docker 组相当于授予较高权限，应由管理员决定，不能用 `chmod 666 /var/run/docker.sock` 代替权限管理。

## 2. 一键安装

在本项目目录中运行：

```sh
sh scripts/setup_runtime.sh --install-rust
```

默认使用 `.runtime-env/`，依次执行：

1. 检查 Linux 架构、Git 和 C 编译器。
2. 缺少 Rust 时，仅在指定目录内安装 Rust 1.90.0；Rust 下载优先使用 rsproxy.cn。
3. 在独立 bootstrap 环境安装 uv 0.11.13；Python 包安装器优先使用清华镜像。
4. 获取 AlpaSim 官方代码，切换到固定提交 `affc2eab209fa43bdfa2f26c0f8d437922d78a68`；跳过 Git LFS 自动下载，所需资产另行按清单获取。
5. 上游仓库不跟踪 `uv.lock`，因此将本发布包提供的 `config/alpasim-runtime.uv.lock` 放入独立检出，再执行 `uv sync --frozen --extra all`，不重新求解锁定版本。
6. 编译上游 gRPC protobuf。
7. 安装本项目及兼容覆盖项 `casadi==3.7.2`。
8. 检查 Python 版本、运行模块、原生扩展、模型注册和兼容项，再运行 CPU 回归测试。

注意：发布包冻结锁中的下载地址保持原样，部分依赖来自官方 GitHub/PyPI。`--index-url` 控制 bootstrap 和 RealSim 集成安装，并不把冻结锁内的所有 URL 自动改成镜像。

成功时生成 `.runtime-env/setup-result.json`，状态为 `runtime_python_ready`。脚本会明确记录 `gpu_closed_loop_verified: false`，因为环境创建不等于完整闭环实验验收。

安装日志为 `.runtime-env/setup-*.log`。失败会停止并返回非零退出码；修复原因后重新运行同一命令即可，不会自动删除目录或重置已有代码。

## 3. 复用已有资源、指定位置

机器已有相同 AlpaSim Git 仓库和 Python 3.12 时，可以减少下载：

```sh
sh scripts/setup_runtime.sh \
  --work-dir /data/l2d-runtime \
  --upstream-source /data/existing/alpasim \
  --python /path/to/python3.12 \
  --install-rust
```

`--upstream-source` 只复用本地 Git 对象，创建自己的独立检出，不复制旧 `.venv`、模型缓存或未提交改动，也不修改原仓库。选用 uv 0.11.13 的现成可执行文件时，可追加 `--uv /path/to/uv`。

第一次创建必须使用尚不存在的工作目录。已有目录只有带本脚本的 ownership marker、且固定版本一致时才允许继续，防止误用一个正在运行实验的环境。

先看计划、不执行任何下载或写入：

```sh
sh scripts/setup_runtime.sh --plan --work-dir /data/l2d-runtime
```

默认删除继承的代理设置；需要使用代理时必须明确传入 `--keep-proxy`。更换 bootstrap/集成包镜像用 `--index-url https://pypi.org/simple`。不要为解决网络问题删除 `--frozen` 或随意升级 torch、transformers、protobuf 等核心依赖。

完全离线安装需缓存完整依赖、本地 upstream Git、可用 Python 3.12、Rust 和 uv：

```sh
sh scripts/setup_runtime.sh \
  --offline --upstream-source /data/existing/alpasim \
  --python /path/to/python3.12 --uv /path/to/uv
```

`--offline` 不等于打包后的离线安装介质。缺少缓存时脚本会失败，不会宣称网络不需要。

## 4. 环境验收

```sh
.runtime-env/venv/bin/python scripts/doctor_runtime.py \
  --upstream .runtime-env/alpasim

.runtime-env/venv/bin/python scripts/doctor_runtime.py \
  --upstream .runtime-env/alpasim --require-gpu --require-docker
```

前者只检查 Python 层；后者还检查 NVIDIA 驱动、PyTorch CUDA 可见性、Docker 服务、Compose 和 NVIDIA runtime 注册。这些检查不会加载模型权重，也不会启动仿真容器，因此仍不能替代真正的闭环 smoke test。

## 5. 获取 30 PCU 核心数据

核心资产存放在 Hugging Face 私有 **dataset** 仓库。仓库内目录拼写为 `langauge2drive.RealSim/`，不是代码目录名 `language2drive.RealSim/`。

先取得仓库的读取权限，再执行：

```sh
.runtime-env/venv/bin/python scripts/hub_assets.py download \
  --repo Mike087/Language2Drive \
  --directory assets --prompt-token
```

如已明确允许使用本机代理，可加 `--keep-proxy`。Token 交互输入不回显、不保存，不能写进脚本或提交到 Git。

数据位于 `assets/langauge2drive.RealSim/`：30 PCU、23 个场景引用、seed 1701 的 60 个双臂执行槽及 7 个运行配置分组。数据包保留任务、评分几何和 oracle。历史审核材料只是外部引用，不是运行依赖。

清单中的 NuRec 场景和官方模型权重需要分别取得上游许可。例子：

```sh
.runtime-env/venv/bin/python scripts/hub_assets.py scenes \
  --bundle assets/langauge2drive.RealSim \
  --directory assets/scenes --prompt-token

.runtime-env/venv/bin/python scripts/hub_assets.py model \
  --bundle assets/langauge2drive.RealSim \
  --model alpamayo2 --directory model_zoo/alpamayo2 \
  --endpoint https://hf-mirror.com --anonymous
```

匿名镜像下载若返回 401/403，不能据此断言权重不存在；检查是否为受限资源，并在官方站完成许可与认证。模型版本应在正式运行前固定并记录。

## 6. 常见故障

| 现象 | 优先检查和处理 |
| --- | --- |
| `No module named venv` / `ensurepip` | 安装与启动 Python 对应的 venv 系统包，或用 `L2D_PYTHON` 指向可用 Python |
| 缺少 `git` / `cc` | 安装 Git / build-essential；脚本不会自动提权 |
| 缺少 `cargo` | 加 `--install-rust`，或先安装 Rust 并确保 cargo 在 PATH |
| Rust 镜像超时 | 检查 rsproxy.cn 连通性；也可用自己配置好的 Rust，再运行脚本而不加 `--install-rust` |
| GitHub / PyPI 超时 | 查看失败阶段；优先用 `--upstream-source`、本地缓存和镜像。确有授权才加 `--keep-proxy` |
| Git LFS smudge 下载失败 | 环境安装不需要自动拉取全部 LFS 资产，检出前设置 `GIT_LFS_SKIP_SMUDGE=1`；新脚本已默认设置。失败的半成品检出应保留并换用新的工作目录 |
| `Unable to find lockfile` | 使用完整发布包中的 `config/alpasim-runtime.uv.lock`；不能仅下载上游仓库后直接运行 `--frozen` |
| `No space left on device` | `df -h` 检查工作目录和缓存所在磁盘，换到更大目录；不要删除原实验日志或全局缓存 |
| ownership marker / wrong pin | 选择一个新 `--work-dir`；不要删除检查逻辑或强制覆盖旧环境 |
| protobuf 或 `pkg_resources` 导入错误 | 确认上游冻结依赖安装成功，保留其 protobuf/setuptools 约束，再重跑脚本 |
| `casadi.tools.SX/MX` 导入错误 | 重新应用 `config/runtime-compatibility.txt`，确认 CasADi 为 3.7.2 |
| `utils_rs` 导入失败 | 检查 Rust 构建日志；不要把另一台机器的虚拟环境目录直接复制过来 |
| `libGL.so.1` / `libgthread` 找不到 | 由管理员安装对应系统库（Ubuntu 常见为 libgl1、libglib2.0-0） |
| `torch.cuda.is_available()` 为 false | 检查 nvidia-smi、驱动、CUDA_VISIBLE_DEVICES 和容器 GPU 映射；不需要先重装评分代码 |
| Docker permission denied | 请管理员授权，重新登录后验证 `docker info`；不要全局开放 socket 权限 |
| Docker 无 GPU runtime | 按 NVIDIA 官方指南配置 Container Toolkit，必要时由管理员重启 Docker |
| HF 401 / 403 | 检查 token、私有仓库读取权限、NuRec gated 授权；不要把写权限 token 发给镜像 |
| 环境检查通过但模型 OOM | 根据实际可用显存安排模型分片/设备；环境创建不保证任意模型适配 24 GB 显卡 |

需要报告问题时，提供失败阶段、脱敏日志末尾和检查输出。不要附上 token、代理密码或完整环境变量。

## 7. 手动配置的对应步骤

已安装 uv、Rust、系统库后，以下步骤与脚本的主要安装流程对应：

```sh
export GIT_LFS_SKIP_SMUDGE=1
# 在本发布包根目录执行；只复制依赖锁，不复制已有 Python 环境。
git clone https://github.com/NVlabs/alpasim.git /data/l2d-alpasim
git -C /data/l2d-alpasim checkout --detach affc2eab209fa43bdfa2f26c0f8d437922d78a68
cp config/alpasim-runtime.uv.lock /data/l2d-alpasim/uv.lock
cd /data/l2d-alpasim
uv sync --frozen --extra all --python 3.12
cd src/grpc
../../.venv/bin/python -c 'from scripts.compile_protos import compile_protos; compile_protos()'
```

回到本项目根目录，安装覆盖项与本项目：

```sh
uv pip install --python /data/l2d-alpasim/.venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -e '.[adapter,test]' -r config/runtime-compatibility.txt
/data/l2d-alpasim/.venv/bin/python scripts/doctor_runtime.py --upstream /data/l2d-alpasim
/data/l2d-alpasim/.venv/bin/python -m pytest -q tests
```

这里手动方式的 `.venv` 路径与自动脚本的 `.runtime-env/venv` 不同，二者不要混用。后续不要使用自动重新同步依赖的 `uv run` 覆盖 CasADi 兼容项；运行服务时直接使用该环境的 Python，或明确使用 `uv run --no-sync`。
