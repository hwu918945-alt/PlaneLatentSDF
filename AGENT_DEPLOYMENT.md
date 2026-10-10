# 当前主方法的部署约束

以 [SAX_Myocardium_Spec.md](docs/SAX_Myocardium_Spec.md) 为用户确认的规格，以 [SHELL_DEPLOYMENT.md](docs/SHELL_DEPLOYMENT.md) 为实现/运行说明。

SAX-only 心肌 shell；单输出标量 SDF、共享逐点 MLP + Mean/Max Pool、128D latent 与物理距离插值。禁止恢复 gap 概率插值、球形初始化或 Gap Geometry Loss。禁止替换用户暂留的自缩放距离项和 `factor>0.15`。任何旧 Epi cache 不可冒充 shell 输入。

Eikonal query 数和权重属于可调实验配置，当前 128/0.1 是初始值，不是已确定或已验证的最优值。固定点数和区域配额、推理一致性、真实隐藏层 LOO 信息路径、Epi/Endo 等权必须保持。随机初始化与 Eikonal 从首步启用的效果由后续实验判断。
