# PlaneAutoOccupancy 独立仓库迁移（2026-10-11）

新根目录为 Heart_Reconstruction/PlaneAutoOccupancy，主包为plane_auto_occ，独立.git和main分支。提取了原子目录的三条相关Git历史，不携带无关PlaneLatentSDF提交或对象依赖。原仓库和所有旧实验保留。

| 原提交 | 提取后提交 | 内容 |
| --- | --- | --- |
| f050ac4 | 07a135b | 初始Auto-Decoder occupancy |
| 9b6add3 | 4dde928 | 观测诊断和单病例部署 |
| 989601a | 03085d8 | 面积采样与逐层Direct/LOO修复 |

上述历史后追加独立包迁移提交。完整原来源、源码hash和提取提交SHA保存在 MIGRATION_PROVENANCE.json。原始方法规划保留为 REPAIR_PLAN_20261011.md，其中旧路径是原文历史背景。

## 代码变化边界

模型、用户objective.py、几何、数据接口、推理、评估、mesh、面积池、分块梯度和默认config均与989601a中的源文件逐字节一致。train和repair_tests仅将子进程模块名从experiments.plane_auto_occ改为plane_auto_occ。diagnostics增加按SHA256解析移动后的池路径。deployment与Slurm入口改成独立包/冻结run，配置、case名和观测索引由run记录读取，初始化比较改为显式可选输入。

新增pyproject、统一CLI、prepare-run、路径解析和迁移测试。prepare-run复制完整独立源码，不生成指向旧仓库的symlink或worktree指针。模型、loss、默认训练预算和标签协议不因迁移修改。

## 历史运行数据

V1/V2共380个非runtime文件复制时逐文件SHA256验证；两套冻结源码也根据原source_lock验证。复制的checkpoint、缓存、采样池、日志、原指标和网格未改写。

runs/MIGRATION_MANIFEST.json记录复制前后位置及每个文件hash；runs/ANALYSIS_ADAPTERS.json列出三个仅为新包导入/移动池路径适配而修改的分析脚本。runs内旧JSON/log/checkpoint中的原绝对路径保留作历史元数据，活动推理与诊断使用显式新case路径和新run内池，不依赖那些原路径。历史冻结runtime保存原包布局，仅用于来源核验；当前开发和新部署均使用根plane_auto_occ包。

代码与运行数据分离：runs、真实缓存、权重、日志、训练输出全部Git忽略。迁移没有重新提交训练job。

## 验证

完整本地验证结果写入 VALIDATION_STANDALONE.json：安装包的14项测试、独立CPU合成两步部署、迁移后V2全部7层最终指标重算一致，以及代码/配置和artifact hash校验。验证完成后以该JSON为准。

用户随后明确要求覆盖原GitHub仓库、缓存不push。因此origin继续使用 git@github.com:hwu918945-alt/PlaneLatentSDF.git，将其main根目录替换为独立PlaneAutoOccupancy项目。GitHub仓库显示名称仍为PlaneLatentSDF；代码布局、包和本地Git已经独立。覆盖前的远端main为213f9c6092bbe96b8200159a27dff199545e656b，已保留到备份分支 `archive/plane-latent-sdf-before-auto-occupancy-20261011`，再用带该SHA保护的force-with-lease推送新main。原实验分支保留，本地旧PlaneLatentSDF不改动。
