# PlaneAutoOccupancy

独立的心肌 Binary Occupancy 仓库：病例 Auto-Decoder、位置/phase 生成 plane latent、按物理层间距离插值、单头心肌占据场。当前方法基于2026-10-11修复版，安装、训练、评估和 Slurm 部署均不依赖 PlaneLatentSDF。

## 方法与默认协议

- 128维case code，128维plane latent；宽256、五Linear/四Softplus、拼接skip的decoder。正logit为心肌，心腔/背景为负；零值面是占据分类边界，模型没有SDF距离约束。
- 每层每次更新32768点：心肌16384、心腔8192、背景8192。观测轮廓联合bbox+20mm内按区域面积均匀提议，只排除≤0.05mm边界带；near/far仅统计。
- 每层面积池262144点，无放回抽样，8步完整覆盖，80步刷新；三观测层每次98304独立输入、131072次decoder查询。
- 端点为Direct，内部层为0.75 Direct+0.25 LOO；总目标为1.5×各层均值+1e-4×mean(code²)。4096点分块累计，全病例完成后clip5和Adam.step各一次。
- 默认随机初始化seed19、共享/code学习率0.001、2000次完整更新、Seam0，无早停和best-checkpoint选择。
- 仅观测轮廓参与训练；隐藏层在固定最终模型训练结束后评估。训练LOO是邻居latent一致性监督，目标层标签参与训练。
- 主评估160²/+5mm/logit0，另做+20mm审计；最终128³网格不外推、不封盖。各观测层Dice≥0.90为事后工程验收。

默认参数见 [plane_auto_occ/config.json](plane_auto_occ/config.json)，原始修复规划见 [docs/REPAIR_PLAN_20261011.md](docs/REPAIR_PLAN_20261011.md)。迁移保留模型、loss、采样算法、坐标规范和优化设置的原始源码字节；变更集中在包路径、运行准备和可迁移的文件引用。

## 安装和命令

使用已有Torch CPU/CUDA环境，在仓库根目录安装：

```bash
python -m pip install -e . --no-deps
python -m plane_auto_occ --help
plane-auto-occ --help
```

新环境先安装适合机器的Torch及 [requirements.txt](requirements.txt) 中依赖。Python≥3.10。

```bash
python -m plane_auto_occ test
python -m plane_auto_occ test-repair
python -m plane_auto_occ test-migration
```

各命令也支持模块入口，例如 `python -m plane_auto_occ.train --help`。包安装后可以从任意工作目录调用。

## 准备单病例与提交 Slurm

输入为occupancy schema `plane_auto_occ_v1` NPZ，需要真实Epi/Endo轮廓、坐标/axis/s及原归一化元数据。已有缓存可直接使用；shell-schema缓存可用 `convert-cache` 转换。转换只提供数据格式适配，不导入 PlaneLatentSDF 代码。

```bash
python -m plane_auto_occ prepare-run \
  --case /absolute/path/002_ED.npz --observed 0,3,6 \
  --output /absolute/path/PlaneAutoOccupancy/runs/002_ED_next

export TASK_EXPERIMENT=/absolute/path/PlaneAutoOccupancy/runs/002_ED_next
export TASK_PY=/absolute/path/existing_cuda_environment/bin/python
sbatch --job-name=plane_auto_occ_002_ED \
  --output="$TASK_EXPERIMENT/logs/auto_occ_%j.out" \
  --error="$TASK_EXPERIMENT/logs/auto_occ_%j.err" \
  scripts/single_case_job.sh
```

prepare-run复制缓存、配置和独立源码快照到新run，建立观测面积池及SHA256锁。可用 `--config` 指定配置，默认设置不变。后续Slurm读取该run的冻结源码，不读取开发目录或旧仓库。可选 `--reference-initial` 显式复制初始化对照checkpoint；没有隐式的V1目录依赖。准备及训练都要求新/空输出目录。

部署先做独立两步cold smoke，再重新初始化正式训练；里程碑仅观察观测层/无标签间隙。完整协议见 [SSH_DEPLOYMENT.md](SSH_DEPLOYMENT.md)。

## 直接训练、评估和导出

```bash
python -m plane_auto_occ train \
  --case /absolute/path/002_ED.npz --observed 0,3,6 \
  --area-pool /absolute/path/generation_000.npz \
  --output /absolute/path/new_fit --device cuda --steps 2000 \
  --milestones 0,50,100,200,500,1000,2000
python -m plane_auto_occ evaluate \
  --checkpoint /absolute/path/new_fit/last.pt --case /absolute/path/002_ED.npz \
  --output /absolute/path/new_evaluation --device cuda
python -m plane_auto_occ extract-mesh \
  --checkpoint /absolute/path/new_fit/last.pt --case /absolute/path/002_ED.npz \
  --output /absolute/path/myocardium.obj --device cuda --resolution 128
```

checkpoint格式仍兼容旧 `plane_auto_occ_v1`。移动旧checkpoint无需改写权重；诊断优先从新run内按文件SHA256找到对应面积池。缓存字节必须与拟合清单相同。

## GitHub

按用户要求，独立项目覆盖原GitHub仓库的main： https://github.com/hwu918945-alt/PlaneLatentSDF 。GitHub仓库名称沿用原名称，主分支内容为本独立PlaneAutoOccupancy项目；旧main保留备份分支。真实缓存、权重和runs不push。

## 仓库与本地运行数据

```text
plane_auto_occ/  可安装的独立Python包、默认配置与测试
scripts/        Slurm及迁移验证工具
docs/           方法规划、来源、迁移和验证记录
runs/           本地缓存、冻结源码、权重、日志、网格和分析（Git忽略）
```

本地已复制V1和V2完整运行结果到 `runs/plane_auto_occ_002_ED_20261011_v1/` 和 `runs/plane_auto_occ_repair_002_ED_20261011_v2/`，原文件保留。历史提交来源和校验记录见 [docs/MIGRATION.md](docs/MIGRATION.md)。这些历史runs不随Git克隆下发，真实缓存和权重不上传。

V2完成2000步，观测/严格隐藏平均contour-proxy Dice分别0.8018/0.8018，已形成闭合环，仍有壁厚高估，尚未通过逐层0.90验收。执行成功不等于几何拟合或跨病例先验成功。当前修复训练路径为单病例；多病例面积池和独立code的Adam动量隔离未在本轮迁移中扩展。
