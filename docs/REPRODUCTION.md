# 当前方法复现

当前 SAX-only 心肌 shell 的远端数据准备、CUDA smoke、训练、续训、Slurm 和评价命令见 [SHELL_DEPLOYMENT.md](SHELL_DEPLOYMENT.md)。

发布已包含完整 shell 输入，先运行 `python scripts/verify_release.py`，无需重新读取原 static 仓库。部署接口与恢复包方法文件的对应关系见 `provenance/release/import_24c0be0.json`。

## 002 单患者拟合

```bash
python scripts/prepare_single_patient.py --data data/plane_sdf_shell --output preflight/shell_single_002
python train.py --config configs/shell_single_002.json --data preflight/shell_single_002 --output runs/shell_single_002 --device cuda
python evaluate.py --checkpoint runs/shell_single_002/last.pt --data preflight/shell_single_002 --split train --output outputs/shell_single_002/train --device cuda --meshes --resolution 160
python evaluate.py --checkpoint runs/shell_single_002/last.pt --data preflight/shell_single_002 --split val --output outputs/shell_single_002/val --device cuda --meshes --resolution 160
# 或从仓库根目录提交完整拟合/评价流水线：
mkdir -p logs
sbatch sbatch/train_single_shell.sh
```

拟合只更新 002 的 ED/ES，400 epochs 共 400 次更新。012 的 ED/ES 仅验证，不反传；固定最后一步评价，不按 012 指标选择拟合结果。它仍更新完整 Encoder/Decoder/Head，并通过原 EMA 更新 Teacher，没有病例 latent 参数表。

## 多患者训练

```bash
sbatch sbatch/train_main.sh
```

主配置保持 200 epochs、batch_cases=4，每轮 143 次更新，合计 28600；从随机权重开始，不使用旧 Epi 或单患者权重。Val 为 36 人，Test 为 36 人；仅 Val 选择 `best.pt`。单/多患者的日志、输出与锁分离。

两个提交入口都检查发布 hash、已完成连续源轮廓审计的输入清单，并在各自 Train 数据上执行全损失 CUDA smoke。首次部署额外使用 `sbatch/gate_shell_deployment.sh` 检查训练、epoch 边界续训和评价接口；门禁权重不进入正式训练。

原始 Epi 方法及初始化实验属于历史版本 `14ec357`；Bunny/balloon-dog 实验仍使用 `experiments/*/runtime` 内的独立源码快照。不要把旧 Epi 缓存或旧初始化配置传入新主训练入口。
