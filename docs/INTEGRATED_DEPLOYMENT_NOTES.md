# 2026-10-12 集成部署说明

方法、预算和五组定义见仓库根目录 [PlaneAutoOcc_Integrated_Deployment.md](../PlaneAutoOcc_Integrated_Deployment.md)。代码来源为同名交付包，12 个 overlay 文件保持其交付 SHA-256；几何核心 SHA-256 为 `e5567b511cd6a32de835cb0428bc70aff7e2ee8d030d66502335b527bc1ac315`。

安装时发现 manifest 的四个基准 SHA 与声明 commit `9367dadef3c4e9a10c904590a01a24b39378dee4` 不符。反向应用随包 patch 重建其基准后，确认差异仅为各文件末尾增加一个空行。先保存实际 Git 基准文件，再规范这个空行，原安装器的 check-only 与安装均通过；安装后逐个核对 12 个交付 SHA。未变更几何、loss 或网络定义。原文件保存在本地 `revision_backups/`。

本轮从实际有 `outputs/single_case/` 的旧从头单例实验读取 metadata，而不是文档示例变量指向的冻结网络目录；只读取配置及真实 NPZ 来源，不加载旧权重或 optimizer。新实验目录为 `experiments/paired_worst10_20261012_v1`。20 个阶段数据按 SHA 复制，100 份任务配置与运行代码冻结。

本集群 cpudebug 同时只能提交一个 CPU 作业；先运行 20 个首代全容量池检查，再提交余下全量 500 代缓存。完整缓存任务在 59 分钟时限前可重排队，复用已完整发布且通过 hash 核验的池。GPU 拟合须同时具有 500 池完成标记和五组全预算 CUDA 验证标记。

GPU QoS 最多同时提交 8 个任务、同时运行 4 个 GPU 任务，不能提交 100 元素数组。因此本站使用 [paired_worst10_workers.sh](../scripts/paired_worst10_workers.sh)：4 个 Slurm worker，各依次执行 25 个原 scientific task_index，合计仍为 0–99。每次通过冻结 `paired_suite run --task-index` 重新初始化 G/D/z/Adam。失败保留并继续其他独立任务；四个 worker 都结束后才汇总。D/FULL 同配置、同 seed，FULL 从零独立运行。

训练及评分完成之前，`SUITE_COMPLETE.json` 不会存在。完成汇总包含全部 20 阶段，无零面阶段作为低分结果保留；执行错误或缺失阶段会使该组主均值为 null。此版本仍是最差十位患者的观测拟合诊断。

缓存、checkpoint、历史与新输出、分析产物、交付 ZIP 和安装备份均只保存在本地，并由 Git 忽略。此前被 Git 跟踪的合成验证图和验证结果 JSON 退出当前版本的跟踪；本地文件保留。Git 历史不改写。
