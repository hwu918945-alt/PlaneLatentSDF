# 源依据与实现边界

- 已读取：用户提供的 `PlaneLatentSDF(1).zip` 与 `PlaneLatentSDF_DualDecoder_20261011.zip`，以及双 Decoder README。
- 原 `core/model.py` SHA256：`0e411167b1b8967d6a97b1f3580764486bf9adfdcbd92d85f5934b692ad6e767`。
- 原 `core/data.py` SHA256：`6b20af57e7faaef59a4a43f328433533deb7bb13e923baddfb4685a96d17594b`。
- 双 Decoder `core/dual_model.py` SHA256：`4f46860dbfcc68fd3137deebc4cdfeaf718e6209e76186c57f0ed7df3b630464`。
- `OccupancyDecoder` 保持所给基础 `SDFDecoder` 的层数、宽度、Softplus 和输入拼接结构，改名及输出语义；插值沿用源公式并保留坐标梯度。没有把旧 SDF 对象导入新训练图。
- 多边形距离/填充与 stack basis 改编自所给基础源码；使用 chunk 和重复轮廓点清理；仅用于缓存符号审计、分层采样、评估，不用于三维 SDF 学习。
- Auto-Decoder case code、位置生成器、Occupancy 目标、冻结拟合、新缓存/评估/导出入口是本次新代码，使用 PyTorch/NumPy/scikit-image 标准 API。位置条件 plane-latent + LOO 的组合是此次实验设计，不能归称为任何标准论文的官方实现。
- 没有读取 SSH 端实际 HEAD 或远程 GitHub 最新内容。本包采取仅新增实验目录的 Git commit，避免对未知本地版本做字符串替换。
- 不附旧 zip 的 Git 历史、缓存、模型权重或任何凭据；bundle 独立根提交，patch 可追加到当前仓库。
