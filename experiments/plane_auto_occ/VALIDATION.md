# 本次已执行验证

验证环境：CPU，Python 3.12，PyTorch 2.14.1+cpu。未访问 SSH GPU，未运行真实 M&Ms-2。

## 数值契约测试

`python -m experiments.plane_auto_occ.tests`：6 个测试通过。

- 不等间距插值与 LOO 权重正确，插值对 query 轴向坐标的梯度保留，域外 query 拒绝。
- Decoder 是5个 Linear/4个 Softplus，原始输入 concat skip 保留。
- Shell 缓存转换、512/256/256类预算、观测层选择正确；旧输出拒绝覆盖；filled Epi 标签冒充 Shell 被拒绝。
- Direct/LOO/可选 Seam 前向反向有限，梯度到达 generator、decoder 和 case code。
- 冻结模型后 code 可更新，所有共享参数不变。
- 修改隐藏切片的全部 Occupancy 标签、保留观测切片时，训练目标逐位一致，固定随机种子下没有隐藏标签进入目标。
- 正值内部的解析球 Mesh 提取验证朝向、mm反归一化、无边界边；解析环检测两个闭合曲线。

此外，将源 `SDFDecoder` 的 state_dict 载入 `OccupancyDecoder`，对同一随机输入前向：`torch.equal=True`。说明此次 Decoder 层结构与所给基础源码严格一致。

## 实际合成训练链路

合成病例：7个 SAX 的光滑椭圆心肌 Shell，训练只选0、3、6三层。使用默认128D/256width/1024points、Direct0.5/LOO1/L2 1e-4、Seam0，seed19，CPU训练800步。

| 项目 | 结果 |
|---|---|
| 最后一步 total loss | 0.259465 |
| 最后一步 Direct BCE | 0.221395 |
| 最后一步 LOO BCE | 0.148768 |
| 观测3层均匀96×96网格平均代理 Dice | 0.863475 |
| 隐藏4层平均代理 Dice | 0.910529 |
| 训练中间层 LOO 代理 Dice | 0.930294 |
| 7个真实层 + 2个中间位置 | 均为两个闭合曲线、零开曲线 |
| 48³ Mesh | 13160 vertices，25846 faces |
| Mesh nonmanifold edges | 0 |
| Mesh boundary edges | 474，观测轴向截断处开放 |
| Mesh lateral boundary contacts | 0 |

隐藏层更高并不代表泛化优于拟合；规则形状的中间层更易预测，首末层重建仍有误差。此结果仅说明训练与评估链路能执行，且该合成环未坍缩为常数场。

`mean_plane/reversed_plane/zero_plane` 扰动能改变一部分预测，详细数字见 `evidence/synthetic_metrics.json`。这表明本次合成场对 plane latent 有敏感性，仍不能等同于跨病例先验有效性的证据。

同时通过：两合成病例共享网络/两case code的2步训练 smoke；新合成病例冻结网络仅拟合 code 的3步 smoke；前后 checkpoint 每一项共享state_dict `torch.equal=True`。这两项只检查接口和更新范围，不报告泛化效果。

## 复现实验

在仓库根目录，以下路径必须是新目录：

```bash
python -m experiments.plane_auto_occ.make_synthetic --output runs/auto_occ_check/raw
python -m experiments.plane_auto_occ.convert_cache \
  --input-root runs/auto_occ_check/raw --output-root runs/auto_occ_check/occ
python -m experiments.plane_auto_occ.train --case runs/auto_occ_check/occ/train/train_ED.npz \
  --observed 0,3,6 --output runs/auto_occ_check/fit --steps 800 --device cpu
python -m experiments.plane_auto_occ.evaluate --checkpoint runs/auto_occ_check/fit/last.pt \
  --case runs/auto_occ_check/occ/train/train_ED.npz --output runs/auto_occ_check/eval --resolution 96
python -m experiments.plane_auto_occ.extract_mesh --checkpoint runs/auto_occ_check/fit/last.pt \
  --case runs/auto_occ_check/occ/train/train_ED.npz --output runs/auto_occ_check/myo.obj --resolution 48
```

不同 PyTorch/硬件可有数值差异；CPU默认2线程。此包不附合成checkpoint，避免把合成权重用于真实病例训练。

## 未验证项目

真实心肌环、不同病例解剖差异、M&Ms-2多病例先验、新真实病例latent拟合、真实MRI raster Dice、完整闭合3D心脏、临床体积/EF。需要按 README 在 SSH 端开展单病例短程验证后再判断。
