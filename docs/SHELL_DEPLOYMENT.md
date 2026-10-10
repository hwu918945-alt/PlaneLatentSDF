# SAX-only 心肌 shell 部署与验证

本版实现 `SAX_Myocardium_Spec.md`，替代主目录原 Epi 实心区域方法。原始历史实验的独立 `runtime/` 保留。当前版本不含 LAX、双头、额外网络或新距离回归公式。

## 1. 环境和位置

需要 Python 3.10+ 与适合远端 CUDA 的 PyTorch。先激活已有 DL 环境，再安装依赖：

```bash
cd /gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneLatentSDF
git pull --ff-only
python -m pip install -r requirements.txt
python scripts/verify_release.py --hashes-only
python verify_core.py --output preflight/shell_core.json
```

恢复包的原工程验证是 Linux CPU；本次远端部署使用 `provenance/release/environment.json` 记录的环境，并在真实 A40 上执行独立 CUDA 门禁。完成信息保存在本机 `preflight/shell_deployment/cuda_gate/CUDA_GATE_COMPLETE.json`，不将工程通过等同于心脏重建质量。Windows 下载位置可在 PowerShell 使用：

```powershell
Set-Location D:\
git clone https://github.com/hwu918945-alt/PlaneLatentSDF.git
```

## 2. 已准备好的双轮廓数据与可选重建

GitHub 的原 `data/plane_sdf` 和 `data/aligned_sax_manifest.json` 只有 Epi，不能恢复缺失 Endo。本版通过新版 adapter 读取原始 `data/cache/<id>/endpoint_case.npz` 的真实双轮廓；不读图像、LAX 或 `training_myo_views`。

本次发布另包含 `data/aligned_sax_shell_manifest.json`、审计元数据与完整 `data/plane_sdf_shell`，新克隆无需再执行转换。先运行 `python scripts/check_shell_ready.py`。以下命令只用于拥有原 static 缓存、需要重新生成输入的机器。

```bash
bash scripts/prepare_shell.sh /gpfs/work/aac/haoyuwu24/Heart_Reconstruction/LV_Epi_CrossSDF_MeshPrior_MnM2_v11_5static
```

也可分开执行：

```bash
python adapt_static_cache.py --source /path/to/static_repo --output data/aligned_sax_shell_manifest.json
python prepare_cache.py --manifest data/aligned_sax_shell_manifest.json --output data/plane_sdf_shell --resume
python scripts/verify_dataset.py
```

adapter 校验 `endpoint epi+endo` 元数据与每层连续多边形严格嵌套，验证 channel 0/1 的外/内边界含义。无效/缺失/相交的轮廓以原因和原始切片索引记录，不构造假 shell；每时相仍至少需要三层，否则明确失败。完整划分要求仍是 570/72/72 时相。

输出：

| 路径 | 内容 |
|---|---|
| `data/aligned_sax_shell_manifest.json` | 有效双轮廓、平面几何、固定划分 |
| `data/aligned_sax_shell_manifest_audit.json` | 通道证据、排除层和量化误差 |
| `data/plane_sdf_shell/{train,val,test}` | 新 schema 的 8192 分层候选池 |
| `preflight/shell_dataset_audit.json` | 全部 SDF 与连续源轮廓、配对/划分一致性 |

审计报告中的排除层可能缩小首末有效观测范围，需要检查原因；不存在的 cavity 与缺失标注无法区分时，这版只采用双轮廓有效层。

预处理固定尺度 100 mm，中心取首层有效平面原点，不依赖隐藏层形状均值。共轴近似造成的轮廓投影超过 0.01 mm 时，默认失败；只有明确接受近似时才在独立 `prepare_cache.py` 命令中添加 `--project-parallel`。投影后重新计算真实平面 SDF，报告最大投影误差。

`--resume` 只复用输入摘要、种子、projection 设置匹配的完整 NPZ；病例随机种子独立，不受中断或处理顺序影响。发布仅包含准备好的模型输入；原 source 缓存、图像和训练输出不上传。

部署新增 `scripts/prepare_shell_cache_parallel.py`，直接调用恢复包的 `convert()` 和同一病例随机种子；002 ED/ES 与串行原入口逐字节一致。`scripts/verify_shell_cache_parallel.py` 按患者拆分原连续多边形审计，汇总时强制完整 285/36/36 划分和 714 时相覆盖。原方法与采样实现没有改动。

```bash
# 已有双轮廓 manifest，按分配到的 CPU 数并行生成/审计：
python scripts/prepare_shell_cache_parallel.py --workers 4 --resume
python scripts/verify_shell_cache_parallel.py --workers 4
# 或在本集群提交 CPU-only 任务：
sbatch sbatch/prepare_shell_cache.sh
```

## 3. 训练和 Slurm

先测试训练集的新缓存与 CUDA 二阶梯度：

```bash
python scripts/smoke_shell.py --data data/plane_sdf_shell --device cuda --output preflight/shell_cuda.json
python train.py --config config.json --data data/plane_sdf_shell --output runs/shell --device cuda
```

继续同一个新版本的训练：

```bash
python train.py --config config.json --data data/plane_sdf_shell --output runs/shell --device cuda --resume runs/shell/last.pt
```

旧方法 checkpoint 不可用于续训；新运行目录为 `runs/shell`，不覆盖 `runs/main`。默认 200 epochs，128D latent、256 width。配置中 Eikonal 每病例每 iteration **128 queries、权重 0.1** 沿用旧预算作为初始选择，不是本轮用户确定的最优值；`batch_cases=4` 时一个 optimizer step 最多处理 512 个 Eikonal queries。正权重从首步生效，DINO/Seam 继续原 warmup/ramp。

```bash
mkdir -p logs
sbatch sbatch/train_main.sh
# 其他集群可自行指定 partition/qos：
sbatch --partition=YOUR_PARTITION sbatch/train_portable.sh
```

Slurm 入口检查发布哈希与已审计缓存的完整输入清单，再在真实 Train 数据上执行 CUDA 全损失 smoke，随后训练与评价。耗时的连续轮廓审计在准备阶段一次性完成，并用 `DATASET_READY.json` 锁定输入；不重复占用 GPU 做同一 CPU 审计。首次部署的核心测试与独立 CUDA 训练/续训/评价门禁单独保留。

002 单患者使用 `sbatch/train_single_shell.sh`，配置 `configs/shell_single_002.json`（400 epochs、batch_cases=2），输出 `runs/shell_single_002` / `outputs/shell_single_002`。多患者入口保持 `sbatch/train_main.sh`、`config.json` 与 200 epochs。旧单患者 `preflight/single_patient` 和旧权重不复用。

## 4. 指标与诊断

每个 epoch 的 `metrics.jsonl` 包含：`weighted_eikonal`、`gradient_norm_median`、`field_abs_mean`、`near_field_fraction`、`factor_pass_fraction`、`calibration_valid_fraction`、Direct/LOO 和 teacher/latent 诊断。梯度记录为各病例查询的 median 后按 batch/epoch平均，不是全数据合并的 median。

需要重点观察 Eikonal 首步启用后梯度是否逐渐接近 1、旧门限是否开始放行，以及心腔是否恢复。**工程 smoke 通过不代表它已解决梯度坍缩或证明心脏重建质量。**

```bash
python evaluate.py --checkpoint runs/shell/best.pt --data data/plane_sdf_shell --split val --output outputs/shell/val --device cuda --meshes --resolution 160
python evaluate.py --checkpoint runs/shell/best.pt --data data/plane_sdf_shell --split test --output outputs/shell/test --device cuda --meshes --resolution 160
python extract_mesh.py --checkpoint runs/shell/best.pt --case data/plane_sdf_shell/test/138_ED.npz --output outputs/shell/138_ED.obj --device cuda
```

推理/验证每层也使用确定性分层 2048 点，不编码全 8192 点。Direct/LOO 分开；两条闭合嵌套预测轮廓分别配对到 Epi/Endo。缺失边界、额外岛/孔洞、开口或负值延伸至平面域边界为无效拓扑，计入覆盖率和 checkpoint 选择惩罚；不通过只保留最大组件掩盖失败。`direct_contour_dice`/`loo_contour_dice` 是心肌负值区域的 Dice，查询分辨率在报告中注明；`*_epi_*` / `*_endo_*` 是独立边界指标。mesh 保留所有零值面，不强制封口。

## 5. 本地可复现的工程测试

```bash
python verify_core.py --output preflight/shell_core.json
python make_synthetic.py --output preflight/synthetic_shell
python scripts/verify_dataset.py --data preflight/synthetic_shell --manifest preflight/synthetic_shell/manifest.json --allow-subset --output preflight/synthetic_shell_audit.json
python scripts/smoke_shell.py --data preflight/synthetic_shell --device cpu
python train.py --config configs/smoke_full_losses.json --data preflight/synthetic_shell --output preflight/smoke_run --device cpu --epochs 2 --max-steps 2
```

测试覆盖连续 shell 符号、8192/2048/640 配额、真实零值、固定推理点数、隐藏层不进入 LOO latent、非等间距与倾斜轴采样、完整条件场坐标导数、首步 Eikonal、所有损失反传、Teacher stop-gradient/EMA 和双层 mesh。真实 138 ED/ES 仅用于数据转换/数值检查，不用于 checkpoint 拟合、调参或质量结论。
