# RealSim 部署与使用说明

## 本包包含什么

本目录是独立的发布代码，包含 PCU 数据结构、编译器、运行时接口、
Alpamayo/SimLingo 模型接口、轨迹验证、评分、补充诊断和 CPU 回归测试。
评测公式及证据有效性检查保留，不通过修改结果或删除验证条件进行匿名化。

不包含个人/单位介绍、仓库远程地址、提交历史、账号密码、令牌、服务器地址、
本机绝对路径、内部实验日志、运行缓存、模型权重或第三方源码树。
`private_annotations` 是评测器掌握的隐藏目标几何这一技术字段，不能删除：
它不表示作者身份信息，也不应进入模型输入。

## 最快检查

在本目录中执行：

```sh
sh scripts/setup.sh
```

无需 GPU、Docker、下载模型或访问原开发机器。示例是明确标注的合成数据，
用于测试速度、车道、路线三种评分接口，不是论文实验成绩。
脚本会创建 `.venv`、安装依赖、执行回归测试及示例、检查评分入口。
默认使用清华 PyPI 镜像，不继承代理。已有环境可以重复使用；每次示例
输出进入新的 `artifacts/deployment-*` 子目录，不覆盖历史输出。
指定解释器可用 `L2D_PYTHON=python3.12 sh scripts/setup.sh`。
离线部署和排错见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 使用自己的实验数据

```sh
sh scripts/score.sh --output scores.json \
  --pcus path/to/frozen_pcus.json --experiment path/to/experiment.json
```

按 `DATA_FORMAT.md` 准备冻结 PCU、60 个槽位的组件记录及归一化轨迹。
轨迹路径可相对于 experiment.json 所在目录填写。默认论文入口返回
SISR、CFPS、SemISR、Sem-CFPS、SCS、PCS、Safety、Progress、DrivingScore、L2D，
全部为 0–100 分；保留有效槽位和零计分槽位数量。
失败/不可用执行保留在固定分母中，计零分，不以有效子集替代总体。
缺少槽位、声称有效但缺少必需证据的数据会直接报错。
指标先在 PCU 内平均改写，再对 PCU 等权平均。L2D 在最后使用未舍入的
总体指标计算，不是单次执行分数的平均。SRG/NBD 保留为库级补充诊断。
公式与论文对应关系见 `METRICS.md` 和 `PAPER_ALIGNMENT.md`。
Safety 使用碰撞/驶离道路口径，不包含闯红灯、超速、逆行。

完整闭环推理需要另外准备有权访问的场景资产、冻结评测集和模型权重，
参照 `RUNTIME_SETUP.md` 对接。本包不是包含全部实验数据的复现镜像。

## 上传前检查

```sh
python scripts/check_anonymity.py
```

建议提交配套的 ZIP，而不是打包开发目录。配套 ZIP 使用固定归档时间、
相对文件名，不携带原机器所有者、文件修改时间、Git 历史和隐藏运行目录。
审稿时仅上传本代码包及单独审核过的数据材料，不要上传原工程根目录。

自动检查覆盖常见直接身份/凭据标记，不保证研究内容本身绝对不可被关联。
后续添加 PDF、图片、Notebook、日志或数据时，要另外检查作者元数据和路径。
本项目原始代码沿用父仓库既有 MIT 许可证，第三方依赖适用各自条款。
