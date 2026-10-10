# Bunny 切面密度与优化时间对照

本次 GitHub 发布仅包含代码、预处理缓存与输入校验元数据；文中历史运行结果、submission 和日志仍保留在原服务器，未上传。新克隆运行命令见 [统一复现说明](../../docs/REPRODUCTION.md)。

本实验比较同一 Bunny 的 16、25、50、75 个平行可见切面，以及每组第 2000、4000、6000 次参数更新的重建，合计 12 个预先指定的设置。原单病例 v1 的输入、模型与结果保留；临床主代码、配置和训练 job 未改动。

## 数量选择与方法

[CrossSDF 论文](https://arxiv.org/html/2412.04120v2)的 Table 2 使用 25 个切面测试较粗的结构；Table 4 比较 25、50、75 个切面。这里参考其数量设置用于 Bunny，另保留 16 层稀疏组。这些数量不是 Bunny 的统一标准，本实验也没有改用 CrossSDF 的网络、loss 或采样器。

`runtime/` 完整复制 v1 的主方法代码，编码器、latent 插值、单标量解码器、直接与 LOO 监督、EMA 分布蒸馏、弱几何、接缝和完整三维 Eikonal 均不变。仍用主方法已有球初始化；固定归一化尺度 100 benchmark mm，中心只由各组实际可见轮廓计算，phase 固定为 0。Bunny 单位延续此前约定，不能当作独立核实过的实物毫米。

| 可见切面数 | 相邻间距 / benchmark mm | 固定评价步数 |
|---:|---:|---|
| 16 | 10.1889 | 2000、4000、6000 |
| 25 | 6.3681 | 2000、4000、6000 |
| 50 | 3.1191 | 2000、4000、6000 |
| 75 | 2.0653 | 2000、4000、6000 |

每组从头运行一次 6000-step 轨迹，在三个里程碑保存 student、teacher、蒸馏 center、优化器、RNG 和配置，并评价全部三个快照，不挑选测试指标最好的模型。同一病例一次更新即这里的一轮。所有组使用相同 seed、学习率、权重、每层采样数、warmup/ramp 和 6000-step EMA cosine horizon。旧 v1 的 horizon 为 2000，因此新 16 层组重新训练；旧 2000-step 结果不混进新表中。16 层输入 NPZ 与 v1 逐字节相同。

相同步数下，更多切面带来更多实际监督点和更高计算成本，不能称为相同计算量的对照；对照目的是分别观察继续优化与增加切面能否改善耳根连接和耳尖缺失。

## 公平评价与参考数据

所有组使用**同一批 121 个未见切面**，在轴向范围内分层随机取样，距离任意一组的训练切面至少 0.03 benchmark mm。训练和 checkpoint 保存均不读取 heldout 或完整参考 mesh。评价使用同一 XY 查询范围（所有可见轮廓的共同包围盒加 15 mm）、同一轴向范围、0.75 mm 切面 Dice 栅格、160³ mesh 和 96³ 体素比较。未见切面是新冻结集合，与 v1 的 45 层不同，不能直接把两者的平均值作严格差值。

`source/` 显式链接到 v1 保留的 Stanford 官方原文件和已记录的五孔修补参考；不再改变参考几何。只重建首末观测之间，不外推未知端点，不人工封口，不删除小分量。模型提取失败、无零值面或缺失轮廓均保留记录；连通分量和上部 20% 高度范围 Dice 用于辅助检查耳朵，上部高度指标不是语义耳朵分割。

## 验证、任务与输出

四组数据都验证真实边界零距离、多环内外符号、查询标签一致性、教师冻结、完整 loss 反传和 Eikonal 独立进入 encoder。最大 75 层组另外执行 CUDA 全 loss gate、包含 teacher/center/optimizer/RNG 的恢复等价检查，以及同一个新评价脚本的真实 OBJ/图片/指标输出。gate 输出独立放在 `slices_075/runs/gate*` 与 `slices_075/outputs/gate`，不计入正式对比。

Slurm array 请求每组 1 A40、4 CPU、24 GB、4 小时，最多同时运行两组。每组自动训练到 6000 步后评价三档 checkpoint；CPU 汇总 job 使用 `afterany` 依赖，生成所有预先指定设置的汇总，也记录失败项。`submission.json` 保存真实 job ID；`CUDA_GATE_COMPLETE.json` 仅表示部署检查通过，正式完成以 `results/SWEEP_COMPLETE.json` 的 `EXECUTION_COMPLETE` 为准。

训练日志：`logs/sweep_<array_id>_<task_id>.out/.err`；任务索引 0/1/2/3 分别对应 16/25/50/75 层。

每组具体路径（例如 50 层、4000 步）：

- `slices_050/runs/main/step_004000.pt`：完整 checkpoint；`last.pt` 为恢复点。
- `slices_050/runs/main/metrics.jsonl`：连续训练指标。
- `slices_050/outputs/step_004000/bunny_comparison.png`：参考、输入、预测，同一视角。
- `slices_050/outputs/step_004000/heldout_contours.png`：固定未见切面的参考与预测轮廓。
- `slices_050/outputs/step_004000/bunny_reconstruction.obj`：保留全部分量的预测 mesh。
- `slices_050/outputs/step_004000/metrics.json`、`slice_metrics.json`：汇总与逐层指标。
- `slices_050/outputs/step_004000/eval.out/.err`：该快照评估日志；失败时有 `EVALUATION_FAILED.json`。

全部完成后的对比表与统一镜头大图在 `results/comparison.csv`、`comparison.json`、`comparison_meshes.png`、`results/README.md`。指标不会因测试结果影响后续训练或 checkpoint 选择。

## 复现入口

Python：`/gpfs/work/aac/haoyuwu24/.conda/envs/haoyu/bin/python`。

预处理 `scripts/prepare_sweep.py`，单组数据检查 `scripts/verify_bunny.py`，CUDA gate `scripts/gate_sweep.py`。正式入口为 `sbatch/sweep.sh`，执行 `scripts/run_group.py`，依次训练、评估；汇总入口为 `sbatch/aggregate.sh`。默认保护已有训练目录，恢复使用 `--resume` 并核对数据 hash 与配置；保存最近完整快照之后的未保存日志只在恢复时回退，以保持步数唯一。改动锁定代码或输入会使 manifest 校验拒绝继续运行。
