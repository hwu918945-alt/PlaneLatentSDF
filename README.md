# PlaneLatentSDF

从已对齐的平面轮廓学习连续三维隐式场。本次发布包含**主流程代码、单病例实验代码和准备好的预处理缓存**，克隆后可直接读取输入，无需依赖原服务器的相邻仓库。

当前已实现的方法遵循 [AGENT_DEPLOYMENT.md](AGENT_DEPLOYMENT.md)：SAX-only、LV epicardial envelope 的填充区域、ED/ES 共享单标量场。逐平面编码器产生 latent，按真实轴向距离插值后送入共享解码器；损失包含真实层监督、严格内部层 LOO、距离加权 EMA 分布蒸馏、弱几何、接缝与完整三维 Eikonal。保留已部署的配对方向球初始化。

[SAX/LAX shell 修订文档](PlaneLatentSDF_SAX_LAX_Shell_PointByPoint_Revision.md) 是后续方法规格，**尚未在当前代码中实现**。当前目标不是 epi/endo 心肌壳层；只提取首末观测之间的区域，不外推未知端点，也不人工封口。

## 目录与入口

| 目录 / 文件 | 用途 |
| --- | --- |
| `adapt_static_cache.py` → `prepare_cache.py` | 已有对齐缓存适配、真实平面 2D SDF 与查询点采样 |
| `data/aligned_sax_manifest.json` | 自包含的已对齐 SAX 轮廓与平面几何 |
| `data/plane_sdf/{train,val,test}` | 714 个可直接训练的 NPZ：570 / 72 / 72 |
| `core/` | 数据契约、编码器/插值/解码器、损失、训练目标、网格与评价 |
| `train.py` / `evaluate.py` / `extract_mesh.py` | 主训练、direct/LOO 评价、零等值面提取 |
| `config.json` / `configs/` | 主配置与心脏单患者、初始化、短程验证配置 |
| `experiments/stanford_bunny_single_case_v1/` | Bunny：16 个可见切面，固定 2000 步 |
| `experiments/stanford_bunny_slice_sweep_v2/` | Bunny：16/25/50/75 切面，2000/4000/6000 步快照 |
| `experiments/balloon_dog_slice_sweep_v1/` | Balloon Dog：同一切面密度与步数对照 |
| `vendor/` / `provenance/` | 已内置的评价依赖、来源与输入/代码校验清单 |
| `sbatch/train_portable.sh` | 可从新克隆目录提交的主训练入口 |

三套几何实验均包含原始参考数据、准备好的可见切面、冻结的未见切面、配置、预处理/拟合/评价脚本和独立 `runtime/` 方法快照。输入与代码保持原实验 SHA-256；Bunny 源目录使用仓库内相对软链接。

## 快速开始

需要 Python 3.10+。先为目标机器安装合适的 PyTorch CPU/CUDA 版本，再安装其余依赖；实际验证版本见 [environment.json](provenance/release/environment.json)。

```bash
git clone https://github.com/hwu918945-alt/PlaneLatentSDF.git
cd PlaneLatentSDF
python -m pip install -r requirements.txt
python scripts/verify_release.py
mkdir -p .release-check
python verify_core.py --output .release-check/core.json
```

缓存已准备好，可直接训练：

```bash
python train.py --config config.json --data data/plane_sdf --output runs/main --device cuda
python evaluate.py --checkpoint runs/main/best.pt --data data/plane_sdf --split test --output outputs/main/test --device cuda --meshes --resolution 160
```

主配置为 200 epochs，患者固定划分 285/36/36；患者 ED/ES 始终在同一划分。checkpoint 仅用 Val 选择，Test 不参与。主流程、002 单患者、Bunny / Balloon Dog 和 Slurm 命令见 [复现说明](docs/REPRODUCTION.md)。SDF 查询点、标签和坐标约定见 [缓存说明](docs/DATA.md)。

发布不包含数值报告、结果图、训练日志、权重或重建输出。它们仍保留在原工作目录；运行代码会在本机重新生成 `runs/`、`outputs/`、`logs/` 和实验结果。
