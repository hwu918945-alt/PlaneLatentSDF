# Stanford Bunny 单病例稀疏切面拟合

此目录发布实验代码、原始/预处理参考模型及输入缓存，不包含运行结果。主方法代码的固定副本位于 `runtime/`，逐文件 SHA-256 记录在 `runtime/source_snapshot.json`。

## 数据来源与处理

来源：[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)，Stanford University Computer Graphics Laboratory。保留 `bunny.tar.gz`、高分辨率 `bun_zipper.ply` 和原说明。原 Bunny 底部五孔已明确修补：沿原孔边界 PCA 投影与耳切三角化，仅新增 213 个面；处理记录在 `data/preparation_report.json`。原模型和修补参考分别保存在 `source/bunny_original_mm.ply` 与 `source/bunny_reference_mm.ply`。

约定 `1 源坐标单位 = 1000 benchmark mm`，未独立核实实物单位；旋转后原 Y 为堆栈 Z，固定归一化尺度 100 benchmark mm，中心仅取自可见边界。输入为 16 个平行切面，每层 1024 查询点与 256 边界点；评价在各间隙 1/4、1/2、3/4 位置，共 45 个未见切面。多环 SDF 使用真实线段距离与奇偶符号，不添加环间连接线监督。`phase=0` 仅是加载接口常量。

## 方法与运行

逐平面编码器、latent 物理位置插值、共享标量解码器、直接/LOO、EMA 分布蒸馏、几何、接缝、完整三维 Eikonal 与主代码一致。优化全部 student，teacher 仅 EMA 更新，保留主方法球初始化；没有可学习病例 latent 表，完整参考模型不用于拟合 loss。

`config.json` 相对主配置仅改 `epochs=2000`、`batch_cases=1`，每轮一次更新。固定最终步评价，不使用未见切面选择 checkpoint。预处理/验证脚本为 `scripts/prepare_bunny.py`、`scripts/verify_bunny.py`。缓存已准备好，从本目录直接执行：

```bash
python scripts/fit_bunny.py --device cuda
python scripts/evaluate_bunny.py
```

本机输出为 `runs/main/` 和 `outputs/`，固定路径原始入口不支持自动恢复。评价保留全部分量，限制首末观测层范围，不人工封口。其他实验与 Slurm 运行见 [统一复现说明](../../docs/REPRODUCTION.md)。
