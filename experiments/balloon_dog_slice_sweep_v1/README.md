# Balloon Dog 单病例切面密度 × 优化步数对照

本次 GitHub 发布仅包含代码、预处理缓存与输入校验元数据；文中历史运行结果、submission 和日志仍保留在原服务器，未上传。新克隆运行命令见 [统一复现说明](../../docs/REPRODUCTION.md)。

本实验用 PlaneLatentSDF 原方法直接优化一个 Balloon Dog，比较 16、25、50、75 个可见平行切面，在第 2000、4000、6000 次更新分别提取模型和指标，合计 12 个设置。各组从相同 seed 初始化，在 6000-step 轨迹上保留三个预定 checkpoint，不按测试结果选模型。

## 数据来源与参考几何

模型选用 [OReX 官方数据](https://github.com/haimsaw/OReX)中的 `Data/csl_with_ref/reference/balloondog_scaled.stl`，仓库 commit `089db1f757022606399d5f070d65bd172fe69362`；不使用 `balloondog_box_scaled.stl`，也不合成替代模型。原 STL、原论文 CSL 和仓库 README 保存在 `source/`，精确下载 URL、大小、SHA-256 在 `source/download_provenance.json`。

请引用：Sawdayee, Vaxman, Bermano. *OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields*. CVPR 2023。OReX 官方 README 声明 Data 下提供论文切面及参考模型。

原 STL 合并重复顶点后有 10001 个顶点、19998 个三角形；闭合、绕向一致、Euler 数 2、一个连通分量。**没有补洞、平滑、简化或添加连接几何。** 保留原 Z 轴作为竖直轴，朝向预览为 `source/orientation_preview.png`。

源文件已有尺度处理，物理单位未核实。本实验固定约定 `1 源坐标单位 = 100 benchmark mm`，不再按本物体 bbox 单独归一化；网络坐标和距离仍同时除以固定 100 benchmark mm。准备好的参考为 `source/balloon_dog_reference_mm.ply/.obj`，缩放与拓扑报告见 `data/preparation_report.json`。这些 benchmark mm 不能当作核实过的实物毫米，也不能用来比较动物模型的真实尺寸。

原 CSL 提供 15 个论文观测切面，仅用于保留来源；**本实验重新生成平行切面，不复现 OReX/CrossSDF 原有切面布局或模型结果。** [CrossSDF 论文](https://arxiv.org/html/2412.04120v2) Table 2 在 Balloon Dog 上使用 25 个切面；25 为文献参考密度，16/50/75 为本实验的密度对照。

## 网络、损失和优化

`runtime/` 物理复制已部署主方法，逐文件与主代码 SHA-256 一致：共享逐点编码器、逐层 128 维 latent、物理位置插值、共享单标量解码器、直接与内部层 LOO 重建、距离加权 EMA 分布蒸馏、弱几何、接缝、完整三维 Eikonal 均不变。沿用已有球初始化，优化整个 student；teacher 仅 EMA 更新，latent 由编码器算出，不存在单病例 latent 参数表。

配置相对临床主配置仅更改 `epochs=6000`、`batch_cases=1`。各组使用相同学习率、每层查询和边界采样数、loss 权重、warmup/ramp 与 6000-step EMA horizon。phase 固定 0，仅为加载接口条件，模型没有心脏 ED/ES 相位。每次完整处理同一病例产生一次参数更新，因此这里一轮就是一步。

| 可见切面数 | 间距 / benchmark mm | 留存与评价步数 |
|---:|---:|---|
| 16 | 10.0849 | 2000、4000、6000 |
| 25 | 6.3031 | 2000、4000、6000 |
| 50 | 3.0872 | 2000、4000、6000 |
| 75 | 2.0442 | 2000、4000、6000 |

各层缓存 1024 个查询点和 256 个边界点；源切面若有多个轮廓，使用所有真实边界线段的最小距离与跨环奇偶规则定义内外，不添加连接线监督。中心只由各组自己的可见边界计算，固定尺度一致。更多层在相同步数下提供更多监督点、需要更多计算，因此不是相同计算量的对照。

## 评价范围与产物

所有四组固定使用相同 121 个未见测试切面，距离任意组的训练切面至少 0.03 benchmark mm。训练不读取 heldout 或完整参考模型；完整参考只用于生成可见输入和之后评价。测试指标不能影响训练或 checkpoint 保存。

共同查询范围来自所有预设可见轮廓的包围盒加 15 mm；轴向限于首末真实观测层，每端原始尖端约 0.75 mm 被排除。零值面提取 160³、填充体素 Dice 96³、切面 Dice 0.75 mm 栅格；参考同范围裁切，不封口。逐层记录覆盖率与环数；无预测轮廓的层保留 Dice 和失败覆盖记录，切面距离只能对有预测轮廓的层计算。3D MCD/HD95 用两侧各 5000 个面积均匀样本到另一侧实际三角形的最近距离。保留所有预测连通分量，重点观察腿、颈部、尾巴连接是否恢复。

上部 20% 高度范围 Dice 只是固定高度的辅助项，不能代替语义部位分割或人工判断连接。所有设置统一镜头，不按视觉或测试结果隐藏失败输出。临床主训练与 Bunny 实验的代码、数据、配置和 job 不受本实验修改。

## 检查与运行

四组数据检查真实边界零距离、查询标签、多环内外符号、教师冻结、encoder/decoder/head 梯度及独立 Eikonal 梯度。75 层 CUDA gate 另外检查全部 loss 开启、恢复与连续训练的模型/teacher/center/optimizer/RNG 完全一致，以及真实 OBJ、图片、指标输出。gate 输出独立位于 `slices_075/runs/gate*`、`slices_075/outputs/gate`，不计入正式对照。

由于 QOS `4a40` 的每用户提交任务数量上限为 8，最初四任务 array 提交被调度器拒绝，改为一个 Slurm job 顺序执行 16、25、50、75 层四组，再在同一 job 中生成汇总。资源为 1 A40、4 CPU、24 GB、8 小时上限；每组运行到 6000 步后评价全部三档 checkpoint。job ID 与提交记录在 `submission.json`；代码与输入 hash 锁在 `manifest.json`。部署检查通过不等于形状拟合成功，正式完成以 `results/SWEEP_COMPLETE.json` 与 `PIPELINE_RUN_COMPLETE.json` 为准。

训练日志：`logs/sweep_<job_id>.out/.err`，其中含明确的分组开始标志。

例如 25 层、4000 步：

- `slices_025/runs/main/step_004000.pt`：完整 checkpoint，`last.pt` 用于恢复。
- `slices_025/runs/main/metrics.jsonl`：全部训练步指标。
- `slices_025/outputs/step_004000/balloon_dog_comparison.png`：参考／输入／预测。
- `slices_025/outputs/step_004000/heldout_contours.png`：未见切面真值／预测。
- `slices_025/outputs/step_004000/balloon_dog_reconstruction.obj`：全部分量的预测 mesh。
- `slices_025/outputs/step_004000/metrics.json`、`slice_metrics.json`：汇总与逐切面指标。
- `slices_025/outputs/step_004000/eval.out/.err`：评估日志，失败时保留 `EVALUATION_FAILED.json`。

汇总位于 `results/comparison.csv`、`comparison.json`、`comparison_meshes.png` 和 `results/README.md`。这些正式产物在对应 job 完成后生成。

复现 Python 为 `/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python`。预处理入口 `scripts/prepare_dataset.py`，标签与梯度验证 `scripts/verify_dataset.py`，CUDA 验证 `scripts/gate_sweep.py`，正式入口 `sbatch/sweep.sh` 调用 `scripts/run_all.py`。汇总自动在同一任务内调用 `scripts/aggregate.py`；`sbatch/aggregate.sh` 仅保留为独立重做汇总的入口，此次未额外提交。恢复会核对原数据 hash、完整配置和代码 manifest，已有正式目录使用 `--resume`，不得将旧输出与新轨迹混合。
