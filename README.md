# PlaneAutoOccupancy

独立的心肌 Binary Occupancy 仓库：病例 Auto-Decoder、位置/phase 生成 plane latent、按物理层间距离插值、单头心肌占据场。支持 2026-10-12 的成对多尺度采样、ROI Dice、最后隐藏层 LeakyReLU 和特征残差。安装、训练、评估和 Slurm 部署均不依赖 PlaneLatentSDF。

## 2026-10-12 集成方法与五组实验

完整定义见 [部署文档](PlaneAutoOcc_Integrated_Deployment.md)；本站安装和资源调度说明见 [部署记录](docs/INTEGRATED_DEPLOYMENT_NOTES.md)。新配置使用 `python -m plane_auto_occ.paired_suite`，原默认配置及入口保留兼容性。

|组|新成对多尺度 + ROI 均匀采样|BCE_all + Dice_ROI|最后隐藏激活|h2→h4 残差|
|---|---|---|---|---|
|A|是|仅 BCE|Softplus|无|
|B|是|Dice 权重 1|Softplus|无|
|C|是|同 B|LeakyReLU(0.01)|无|
|D|是|同 B|LeakyReLU(0.01)|相加后激活|
|FULL|是|同 D|同 D|同 D，独立从零复跑|

每层 32768 点：Epi 两侧、Endo 两侧各 3072 点，独立 ROI 均匀点 20480。BCE 使用全部点，整层 soft Dice 只使用 source=4 的 ROI 点；两遍分块计算保留整层 Dice 精确梯度。五组复用同一 500 代点池，配额、点对、真实标签及来源对应关系保留。默认不以常数场监测调整 loss、学习率或停止训练。

本轮 10 位患者的 ED/ES 分别从头训练，5×20 个独立模型，每模型 2000 次更新，全缓存 SAX 监督。原隐藏子集在这里已参与拟合，评分不能称为独立泛化。最终 step2000、固定原三层 bbox+20mm、160²/logit0、96³ 原生 mesh，无封盖、平滑或修复。

```bash
python -m unittest plane_auto_occ.paired_tests -v
python -B -m plane_auto_occ.paired_suite prepare --previous-experiment /absolute/previous_single_case_run --output /absolute/new_suite
cd /absolute/new_suite/runtime
python -B -m plane_auto_occ.paired_suite precache --suite /absolute/new_suite --workers 3
python -B -m plane_auto_occ.paired_suite run --suite /absolute/new_suite --device cuda
python -B -m plane_auto_occ.paired_suite summarize --suite /absolute/new_suite
```

本站按 A、B、C、D、FULL 各提交一个 GPU job，每个版本串行完成十位患者的 ED/ES（20 次独立拟合），五个版本结束后汇总。启动脚本见 `scripts/paired_worst10_arm.sh`；调度器最多同时运行四个版本。

缓存的最终 float32 坐标会重新核验几何有效性和唯一性；ROI 候选拒收补足，边界点在原锚点上成对重试，重试写入 metadata。修复说明见 [部署记录](docs/INTEGRATED_DEPLOYMENT_NOTES.md)。

配置模板位于 `configs/paired_worst10/`；实际配置由 prepare 读取原 metadata 后冻结。真实数据、点池、训练输出、历史分析和安装备份不进入 Git。

## 原版默认协议（兼容入口）

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

## 多病例心脏与25slice实体实验

新增多病例训练入口 `python -m plane_auto_occ run-experiment --experiment /absolute/run --device cuda`。
协议见 [docs/FULL_AND_SOLID_PROTOCOL.md](docs/FULL_AND_SOLID_PROTOCOL.md)：心脏采用患者级285/36/36划分、共享G/D和独立case/phase latent，先训练200epoch；验证/测试冻结G/D，用首/中/末层优化新code2000次，以其余层评估。验证选择checkpoint后运行测试。

bunny及气球狗分别随机初始化，用全部25观测层拟合4000次，每1000次导出PNG/OBJ/checkpoint。保留多个独立轮廓与孔洞，使用实体内外balanced BCE；不读取三维参考网格做监督。两类实验共享同一G/D与Direct/LOO目标，使用独立Adam latent、按病例访问次数刷新面积池及可恢复Slurm部署。原`train`入口继续支持修复版单病例。

```bash
python -m plane_auto_occ test-population
PYTHONPATH=. python scripts/prepare_requested_runs.py --source-root /absolute/compatible/cache-repo --repo-root .
PYTHONPATH=. python scripts/freeze_experiment.py --experiment /absolute/run
PYTHONPATH=. python scripts/submit_fast_heart.py --experiment /absolute/heart_run
# Legacy solid runs retain the direct experiment_job.sh submission.
```

## 全量心脏训练加速

新生成的全量心脏run默认先离线准备全部训练点池，再启动依赖缓存成功的GPU训练；缓存采用mmap、CPU预取与524288点查询上限。每病例完整损失、点数及一次Adam更新不变。新增Numba依赖用于精确轮廓分类加速。A40的小规模整流程试跑为0.1125秒/训练更新、0.0489秒/冻结code更新；这不是完整570-phase epoch的实测结果。方法、验证和缓存阶段管理见 [docs/FAST_TRAINING.md](docs/FAST_TRAINING.md)。
