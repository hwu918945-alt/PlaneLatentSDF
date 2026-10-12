# 2026-10-12 集成部署说明

方法、预算和五组定义见仓库根目录 [PlaneAutoOcc_Integrated_Deployment.md](../PlaneAutoOcc_Integrated_Deployment.md)。代码来源为同名交付包，12 个 overlay 文件保持其交付 SHA-256；几何核心 SHA-256 为 `e5567b511cd6a32de835cb0428bc70aff7e2ee8d030d66502335b527bc1ac315`。

安装时发现 manifest 的四个基准 SHA 与声明 commit `9367dadef3c4e9a10c904590a01a24b39378dee4` 不符。反向应用随包 patch 重建其基准后，确认差异仅为各文件末尾增加一个空行。先保存实际 Git 基准文件，再规范这个空行，原安装器的 check-only 与安装均通过；安装后逐个核对 12 个交付 SHA。未变更几何、loss 或网络定义。原文件保存在本地 `revision_backups/`。

本轮从实际有 `outputs/single_case/` 的旧从头单例实验读取 metadata，而不是文档示例变量指向的冻结网络目录；只读取配置及真实 NPZ 来源，不加载旧权重或 optimizer。首轮检查目录为 `experiments/paired_worst10_20261012_v1`；缓存修复后的正式目录为 `experiments/paired_worst10_20261013_v2`。20 个阶段数据按 SHA 复制，100 份任务配置与运行代码冻结。

初轮首代全容量检查使用 cpudebug 完成。用户随后要求正式缓存和训练全部使用 GPU Slurm job；正式预缓存改为 aiaca40 / 4a40、GPU1 / CPU4 / 16GB / 24h，复用通过 hash 核验的完整池。缓存几何沿用原 Shapely/NumPy CPU 实现。拟合须同时具有 500 池完成标记和五组全预算 CUDA 验证标记。

GPU QoS 最多同时提交 8 个任务、同时运行 4 个 GPU 任务。本站按版本提交五个独立 job：A、B、C、D、FULL，每个通过 [paired_worst10_arm.sh](../scripts/paired_worst10_arm.sh) 串行执行对应十位患者的 ED/ES，共 20 次独立拟合。四个版本可以同时运行，第五个由调度器排队；没有数组，也不按 25 个训练划分 worker。每次通过冻结 `paired_suite run --task-index` 重新初始化 G/D/z/Adam。失败保留并继续其他独立任务；五个版本都结束后才汇总。D/FULL 同配置、同 seed，FULL 从零独立运行。

训练及评分完成之前，`SUITE_COMPLETE.json` 不会存在。完成汇总包含全部 20 阶段，无零面阶段作为低分结果保留；执行错误或缺失阶段会使该组主均值为 null。此版本仍是最差十位患者的观测拟合诊断。

缓存、checkpoint、历史与新输出、分析产物、交付 ZIP 和安装备份均只保存在本地，并由 Git 忽略。此前被 Git 跟踪的合成验证图和验证结果 JSON 退出当前版本的跟踪；本地文件保留。Git 历史不改写。

## 2026-10-13 缓存坐标适配修复

原首代检查有四阶段失败：206_ED 的 Endo cavity 两个相邻采样单位映射成同一 float32 点；029_ED、352_ES、138_ED 在最终训练坐标中存在无效点。029_ED 的一个 ROI 点离 Epi 为 0.049985077mm，未满足严格 >0.05mm 排除带。按用户要求修复缓存后提交五组训练。

新增 `paired_pool_coordinates.py`，只修改 `paired_pools.py` 的最终坐标适配。最终 float32 坐标决定标签、ROI/边界带有效性及唯一性。ROI 数值无效或重复候选拒收并按原均匀提议补足；边界重复/数值无效时，在原弧长锚点上成对尝试更细的偏移缩减和原范围内的方向调整。每对仍共享偏移，保持原 scale_id、点对 id、来源与配额；不能支持的锚点仍明确报错。metadata 记录最初无效/重复点、实际重试单位、弧长锚点和实际偏移。

`paired_geometry.py` 交付字节 SHA 保持 `e5567b511cd6a32de835cb0428bc70aff7e2ee8d030d66502335b527bc1ac315`；其他网络、loss、采样预算、80次刷新、8次覆盖、排除带、ROI和优化设置保持原协议。原始交付 manifest 保留；本地修复的旧/新 SHA 单独记在 `INTEGRATED_SOURCE_MANIFEST.json` 的 local_repairs 中，不把修复后适配文件宣称为交付原字节。

缓存 contract 新增 `coordinate_adapter_version=float32_world_valid_unique_v2`，生成新的缓存地址，避免误复用旧数值适配缓存。旧冻结实验不改写；新 v2 实验重新冻结代码和100份配置，每份与 v1 只允许离线池绝对根路径不同。五组共用同一500代池。

三项回归覆盖有效池及随机状态不变、ROI无效/重复拒收可重现、边界碰撞后原锚点/scale和成对标签保留。原六项集成测试继续覆盖整层Dice精确梯度、来源选择、模型恢复和8次覆盖。正式提交前另检查20阶段首代全容量缓存，并独立重算四个原失败阶段的全部最终坐标。CPU缓存与CUDA验证是准备任务，五个训练job分别按版本串行跑20次，不产生25次worker划分。

正式流程按用户最新要求使用一个GPU预缓存job，再加A/B/C/D/FULL五个GPU训练job。仅资源申请变更为A40；五个训练job只依赖全量预缓存成功，CUDA检查在提交前完成并通过标记复核。实际job ID和提交后状态保存在本地实验README与SUBMISSION.json，不进入Git。
