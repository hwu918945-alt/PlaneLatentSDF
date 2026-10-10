# 本版工程验证记录

本次在云端 Linux、Python 3.12、PyTorch 2.14.1+CPU 完成；没有访问用户 Windows D 盘，没有在用户 CUDA/DL 环境训练。

| 检查 | 结果 |
|---|---|
| 7 项核心行为测试 | PASS；池/取样配额、符号、隐藏层隔离、坐标全导数、Eikonal 首步反传、所有损失/EMA、确定性 2048 推理、双零值面/心腔 |
| 真实原始 138 ED/ES 适配与双轮廓 SDF 复算 | PASS；最大 SDF 误差 2.575e-5 mm，最大轴向残差 6.087e-6 mm |
| 合成 ED/ES 的完整训练入口 | PASS；optimizer/clip/checkpoint/validation 正常，无非有限值 |
| 所有损失开启的独立 optimizer smoke | PASS；Eikonal 从首步为正权重，Seam/DINO ramp=1 时有限 |
| epoch 边界 checkpoint 恢复 vs 连续三步 | PASS；Student/Teacher/center/RNG 逐位一致 |
| 评价入口与失败报告 | PASS；三步权重未产生有效心肌双边界，如实报告 coverage=0 与无符号变化 mesh failure |
| 解析 shell 的真实网格提取入口 | PASS；保留两层面与轴向开放边界，拒绝人工封口 |
| 历史 epi cache 误传新主入口 | 按预期拒绝，指向重新生成双轮廓数据的说明 |

138 属于原 Test 划分，本次只做预处理和几何标签审计，没有拿它优化网络、选择 checkpoint 或得出效果结论。固定完整 285/36/36 shell 数据尚需在持有原始对齐双轮廓缓存的远端生成并审计。

短程 smoke 的 `factor_pass_fraction` 和 `calibration_valid_fraction` 仍为 0，梯度范数约 0.002–0.004。它证明 Eikonal 的计算/反传已接通，**不能证明首步启用 Eikonal 已解决梯度坍缩**。旧距离缩放、符号项与 `factor > 0.15` 保持原公式；后续以真实训练曲线判断。

工程命令与测试源见 [部署说明](SHELL_DEPLOYMENT.md)、`verify_core.py`、`tests/test_shell.py`。CUDA、完整数据、Slurm 和临床重建质量未在本次云端验证，远端入口会在长训练前执行真实 Train 数据 CUDA smoke。
