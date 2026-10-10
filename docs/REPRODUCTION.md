# 主流程与单病例实验复现

以下命令从仓库根目录开始，`python` 指向已安装依赖的环境。准备好的缓存随仓库发布，无需重新预处理。先运行 `python scripts/verify_release.py` 检查输入与代码。

## 多患者心脏训练

```bash
python train.py --config config.json --data data/plane_sdf --output runs/main --device cuda
python evaluate.py --checkpoint runs/main/best.pt --data data/plane_sdf --split val --output outputs/main/val --device cuda --meshes --resolution 160
python evaluate.py --checkpoint runs/main/best.pt --data data/plane_sdf --split test --output outputs/main/test --device cuda --meshes --resolution 160
```

200 epochs，每 epoch 完整遍历 570 个训练时相文件，batch_cases=4，共 143 次更新。验证每 5 epochs 及末轮检查全部 Val，用实际 direct/LOO 零轮廓与距离场/符号项选 checkpoint；缺失轮廓保留 coverage 与选模惩罚。恢复必须使用相同配置：

```bash
python train.py --config config.json --data data/plane_sdf --output runs/main --device cuda --resume runs/main/last.pt
```

## 002 配对单患者直接优化

```bash
python scripts/prepare_single_patient.py --data data/plane_sdf --case-id 002 --validation-case-id 012 --output preflight/single_patient
python train.py --config configs/paired_initialization_fit_check.json --data preflight/single_patient --output runs/single_002 --device cuda
python evaluate.py --checkpoint runs/single_002/last.pt --data preflight/single_patient --split train --output outputs/single_002/train --device cuda --meshes --resolution 128
python evaluate.py --checkpoint runs/single_002/last.pt --data preflight/single_patient --split val --output outputs/single_002/val --device cuda --meshes --resolution 128
```

该配置在 002 的 ED/ES 上优化 400 epochs，每 epoch 一次参数更新；012 属于原 Val，不参与反传。优化整个 student，teacher 仅 EMA 更新，latent 由编码器计算，没有病例 latent 参数表。固定 `last.pt` 评价；Train 是拟合效果，Val 是未见患者诊断。

`configs/` 还保留原随机初始化、延长拟合、早期初始化和短程全损失检查配置。这些属于不同验证阶段，不构成仅改变轮数的严格消融。

## Bunny v1：16 层 / 2000 步

```bash
cd experiments/stanford_bunny_single_case_v1
python scripts/fit_bunny.py --device cuda
python scripts/evaluate_bunny.py
cd ../..
```

固定最终步评价，使用 v1 的 45 个未见切面。输出为该实验 `runs/main/` 与 `outputs/`。v1 是固定路径的原始入口，不支持自动恢复；重复实验请使用新的克隆目录。

## Bunny / Balloon Dog：16/25/50/75 层 / 6000 步

```bash
cd experiments/stanford_bunny_slice_sweep_v2
for count in 16 25 50 75; do
  python scripts/run_group.py --slices "$count"
done
python scripts/aggregate.py
cd ../..

cd experiments/balloon_dog_slice_sweep_v1
python scripts/run_all.py
cd ../..
```

每组从头运行 6000 步，在 2000/4000/6000 步保存并评价三个预定快照，不根据未见切面结果选快照。两套对照各用冻结的 121 个未见切面、160³ 网格、96³ 体素及固定查询范围。更多层增加每步计算量，并非相同计算量对照。v1 的 EMA horizon 与未见切面集合不同，不能混入新对照表。

`run_group.py` 先核对原 `manifest.json` 的代码/输入 SHA-256，默认从完整 checkpoint 恢复。输出在 `slices_NNN/runs/main/` 和 `slices_NNN/outputs/step_NNNNNN/`，`scripts/aggregate.py` 在本机生成汇总。CUDA gate 与数据验证记录是原部署/输入校验元数据，不代表新机器已完成拟合。

## Slurm

新克隆的主流程使用相对路径入口，先创建日志目录；partition/qos 按目标集群指定：

```bash
mkdir -p logs
sbatch --partition=aiaca40 --qos=4a40 --export=ALL,PYTHON_BIN="$(command -v python)" sbatch/train_portable.sh
```

脚本校验代码/输入与核心方法、检查 CUDA、持有输出目录锁，随后训练并分别评价 Val/Test。权重与运行完成标记没有发布，新克隆不会误跳过训练。

几何实验也可用目标环境直接提交，例如从 Balloon Dog 实验目录：

```bash
mkdir -p logs
sbatch --partition=aiaca40 --qos=4a40 --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=08:00:00 --output=logs/sweep_%j.out --error=logs/sweep_%j.err --wrap='python scripts/run_all.py'
```

Bunny 的 `--wrap` 可使用四组循环并随后汇总。确保作业环境中的 `python` 是目标环境。

原 `sbatch/train_main.sh`、实验 `sbatch/` 和 preflight shell 脚本作为历史 HPC 部署代码保留，含原 GPFS 路径与环境。原 `provenance/code_sha256.txt` 是历史运行锁，其中 README 与部署报告属于旧文档；新克隆请使用上述入口。新发布由 `provenance/release/manifest.json` 校验，并额外核验历史锁中的方法源代码与配置，方法本身没有因整理而改动。
