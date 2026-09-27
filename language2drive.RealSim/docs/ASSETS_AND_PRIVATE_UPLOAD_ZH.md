# 数据资产与私有仓库上传

## 仓库约定

- Hugging Face 仓库类型：`dataset`。
- 仓库名：`<账号>/Language2Drive`，必须为 **private**。
- 仓库内前缀：`langauge2drive.RealSim/`。这里的拼写是发布者指定的，不要自动改名。
- 本文脚本不会修改 GitHub 仓库或其可见性。

## 网络与凭据

`scripts/hub_assets.py` 默认会移除大小写的 HTTP、HTTPS、ALL proxy 环境变量，禁用环境代理发现，并给网络请求设置连接超时。明确允许使用代理时加 `--keep-proxy`。公开下载可以使用镜像；认证与私有上传使用 Hugging Face 官方 HTTPS 地址。

不要把 token 写入代码、命令行参数、容器构建参数或数据包。交互执行时使用 `--prompt-token`，输入不回显，也不写入登录缓存。自动化可以使用安全注入的 `HF_TOKEN`。

公开权重通过镜像下载的例子：

```sh
python scripts/hub_assets.py model \
  --bundle /path/to/bundle \
  --model alpamayo2 \
  --directory /path/to/checkpoints/alpamayo2 \
  --endpoint https://hf-mirror.com --anonymous
```

`--anonymous` 显式禁止携带环境或缓存中的 token；受限资源如果拒绝匿名请求，需按上游要求获取访问许可并从官方站下载，不能用镜像绕过许可。

## 初始化与上传

在可以直连官方站的机器上安装 `huggingface_hub`，然后执行：

```sh
python scripts/hub_assets.py init --prompt-token
python scripts/hub_assets.py upload \
  --directory /path/to/bundle \
  --prompt-token
```

本次约 9.15 MB 核心包已上传到私有 `Mike087/Language2Drive`，路径为
`langauge2drive.RealSim/`，初始提交 `51867e4a15be8146910dbe3975d95a2e66c2de28`。
本次核心数据上传获准使用代理；Python 环境、模型与容器没有随之上传。

脚本通过认证确定账号，只允许向该账号的 `Language2Drive` 数据集仓库写入。创建时设为 private；既有仓库也会确认/设为 private，并在上传前读取可见性进行核验。不满足条件则停止。上传不会删除仓库中其他目录的文件，但同一前缀下的同名文件会产生新版本。

成功输出包含 `repo`、`private`、`prefix` 和上传 `commit`。没有成功的 commit 记录时，不能把本地打包或网络探测视为上传成功。大文件上传协议本身可能计算校验值；脚本不额外遍历大型资产计算哈希。

下载自己的私有资产：

```sh
python scripts/hub_assets.py download \
  --repo <账号>/Language2Drive \
  --directory /path/to/downloads \
  --prompt-token
```

## 核心数据包的内容

`package_benchmark.py` 从冻结的可信源和实际运行配置生成下列文件：

| 文件 | 用途 |
| --- | --- |
| `pcus.json` | 30 个 PCU 的指令、语义目标、几何标注及原有三种子 oracle 配置 |
| `compiled/public_workload.json` | 编译后的公开工作负载 |
| `compiled/sidecar.json` | 评测器专用标注，不能提供给被测模型 |
| `compiled/language.json` | 模型服务需要的会话到指令时间表映射，不包含评分目标和 oracle |
| `compiled/run_plan.json` | seed 1701 下的 60 个双臂执行槽 |
| `compiled/leakage_audit.json` | 对应同一单种子集合的 30 组成对输入检查 |
| `runtime/groups.json` | 覆盖 30 个 PCU 的运行分组及配置 |
| `runtime/controller.json`、`runtime/eval.json` | 控制器和评估配置 |
| `runtime/driver-*.json` | 模型服务模板；本地路径用占位符表示 |
| `assets.json` | 上游场景文件、模型仓库与渲染器镜像来源 |
| `manifest.json` | 数量、版本和打包边界说明 |

运行时需解析 `${SCENES}`、`${OUTPUT}`、`${CHECKPOINT}`。`${AUTHORING_ROOT}` 仅用于历史审核引用，其文件不在核心包中，也不是运行或评分依赖；不要把它当作已经随包分发的文件。

导出保留任务定义、评分几何、oracle、配额和覆盖范围声明。源文件不被修改。历史机器路径被重定位，因此新导出包的 source binding / session UUID 可能不同于历史实验；不要混用不同 binding 的 sidecar 和轨迹。

## 完整交付与许可边界

现有 Python 环境不上传，以 [一键创建环境脚本](ENVIRONMENT_SETUP_ZH.md) 代替。
核心数据包与环境安装检查不等于完整 GPU 实验验收；不能仅凭它们成功便宣称闭环推理已完成。

场景 USDZ、模型权重和 NVIDIA 渲染器不是核心包的内置文件。使用者须按各上游条款取得访问权限，依据清单下载。可下载不自动代表可转发；完整标注中的上游来源内容、重建图像以及第三方容器也应核实分发权限。不能把服务器工作目录、模型缓存或现有容器整体归档当作可公开发布的环境。
