# PlaneLatentSDF

当前主流程为 **SAX-only、ED/ES、Epi + Endo 心肌 shell**：共享逐点 MLP + Mean/Max Pool 编码每层 128D latent，按真实高度线性插值，单标量条件 SDF 解码后以 Marching Cubes 提取全部零值面。

方法依据：[确认规格](docs/SAX_Myocardium_Spec.md)。实施与远端命令：[部署说明](docs/SHELL_DEPLOYMENT.md)。

- 每层离线 8192 点，Encoder 分层取 2048；Decoder 监督 640（512 非边界 + Epi/Endo 各 64 零值）。
- Gaussian 保留 1/3/6 mm、0.4/0.4/0.2；Seam 每侧 16，初始偏移 0.1 mm 且不超过相邻层距的 1/20。
- Eikonal 从第一个训练 step 生效，查询 50% 体积均匀、25% Epi 引导、25% Endo 引导。默认每病例 128 queries、权重 0.1，属于待实验的初始参数。
- 只保留真实隐藏 SAX 的 EMA 分布蒸馏；删除 gap 概率插值、球形初始化及 Gap Geometry Loss。
- 原 `observed_loss()` 自缩放距离项及 `factor > 0.15` 暂时保留，并记录通过率与场梯度。

**当前输入是 `data/plane_sdf_shell`：570/72/72 个 ED/ES 文件，患者固定划分 285/36/36。** 双轮廓 manifest、分层缓存、输入签名和 `DATASET_READY.json` 随仓库发布，新克隆可以直接训练。原 `data/plane_sdf` 与 `data/aligned_sax_manifest.json` 只有 Epi，仅保留历史复现用途。

新缓存只读取 v11_5static 的已对齐真实 Epi/Endo 与平面几何，不读取图像、LAX 或辅助心肌 mask。缺失或不严格嵌套的 Endo 不会填成实心心肌；排除记录见 `data/aligned_sax_shell_manifest_audit.json`。只查询保留的首末有效 SAX 之间，不外推或强加封口。

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python verify_core.py --output preflight/shell_core.json
python scripts/check_shell_ready.py
python scripts/smoke_shell.py --data data/plane_sdf_shell --device cuda
python train.py --config config.json --data data/plane_sdf_shell --output runs/shell --device cuda
python evaluate.py --checkpoint runs/shell/best.pt --data data/plane_sdf_shell --split test --output outputs/shell/test --device cuda --meshes
```

固定患者划分仍为 285/36/36，ED/ES 同划分，Test 不参与 checkpoint 选择。`evaluate.py` 分别报告 Direct/LOO、Epi/Endo 边界误差、心肌 Dice 与拓扑覆盖率；`extract_mesh.py` 保存完整 shell，不删除小连通面。没有完整 3D GT 或未观测端部时不报告完整心脏体积/EF。

## 正式运行入口

```bash
mkdir -p logs
sbatch sbatch/train_single_shell.sh  # 002 ED/ES：400 epochs / 400 updates；012 仅验证
sbatch sbatch/train_main.sh          # 285 人：200 epochs / 28600 updates；Val 选模
```

单患者配置是 `configs/shell_single_002.json`，从主配置复制，只调整 epochs、batch_cases 与验证频率。两项训练均保持恢复包方法与随机初始化，互不共享权重；拟合评价使用固定 `last.pt`，多患者评价使用 Val 选出的 `best.pt`。

对应运行/结果目录为 `runs/shell_single_002` / `outputs/shell_single_002` 和 `runs/shell` / `outputs/shell`。数值报告、日志、权重和重建结果只在运行机器上生成，不发布到 GitHub。旧方法输出保持原目录。

恢复包提交为 `24c0be0`，本地导入与方法文件校验值见 [import_24c0be0.json](provenance/release/import_24c0be0.json)。部署层新增并行预处理、数据完整性与 CUDA 门禁、单患者提交入口；`core/`、主训练/评价/提取、原始采样器和主配置逐文件与恢复包一致。

Bunny/balloon-dog 的 `experiments/*/runtime`、旧 Epi 缓存和归档 ZIP 保留为历史实验；其独立源码快照不受主方法修改影响。旧初始化配置及脚本不适用于当前训练入口。
