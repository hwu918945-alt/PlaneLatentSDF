# PlaneLatentSDF

当前主流程为 **SAX-only、ED/ES、Epi + Endo 心肌 shell**：共享逐点 MLP + Mean/Max Pool 编码每层 128D latent，按真实高度线性插值，单标量条件 SDF 解码后以 Marching Cubes 提取全部零值面。

方法依据：[确认规格](docs/SAX_Myocardium_Spec.md)。实施与远端命令：[部署说明](docs/SHELL_DEPLOYMENT.md)。

- 每层离线 8192 点，Encoder 分层取 2048；Decoder 监督 640（512 非边界 + Epi/Endo 各 64 零值）。
- Gaussian 保留 1/3/6 mm、0.4/0.4/0.2；Seam 每侧 16，初始偏移 0.1 mm 且不超过相邻层距的 1/20。
- Eikonal 从第一个训练 step 生效，查询 50% 体积均匀、25% Epi 引导、25% Endo 引导。默认每病例 128 queries、权重 0.1，属于待实验的初始参数。
- 只保留真实隐藏 SAX 的 EMA 分布蒸馏；删除 gap 概率插值、球形初始化及 Gap Geometry Loss。
- 原 `observed_loss()` 自缩放距离项及 `factor > 0.15` 暂时保留，并记录通过率与场梯度。

**仓库原有 `data/plane_sdf` 和 `data/aligned_sax_manifest.json` 只有 Epi，不含 Endo；本版本不会将它们用于心肌训练。** 完整 shell 数据需要从远端已有的原始对齐缓存重新生成。缺失或不嵌套的 Endo 不会被填成实心心肌，排除记录保存在 adapter audit。只查询保留的首末有效 SAX 之间，不外推或强加封口。

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py --hashes-only
python verify_core.py --output preflight/shell_core.json
bash scripts/prepare_shell.sh /path/to/LV_Epi_CrossSDF_MeshPrior_MnM2_v11_5static
python scripts/smoke_shell.py --data data/plane_sdf_shell --device cuda
python train.py --config config.json --data data/plane_sdf_shell --output runs/shell --device cuda
python evaluate.py --checkpoint runs/shell/best.pt --data data/plane_sdf_shell --split test --output outputs/shell/test --device cuda --meshes
```

固定患者划分仍为 285/36/36，ED/ES 同划分，Test 不参与 checkpoint 选择。`evaluate.py` 分别报告 Direct/LOO、Epi/Endo 边界误差、心肌 Dice 与拓扑覆盖率；`extract_mesh.py` 保存完整 shell，不删除小连通面。没有完整 3D GT 或未观测端部时不报告完整心脏体积/EF。

Bunny/balloon-dog 的 `experiments/*/runtime`、旧 Epi 缓存和归档 ZIP 保留为历史实验；其独立源码快照不受主方法修改影响。旧初始化配置及脚本不适用于当前训练入口。
