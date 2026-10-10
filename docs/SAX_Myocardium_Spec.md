# PlaneLatentSDF — SAX-only Myocardium Shell 方法与采样规格

> 版本：2026-10-10 讨论确认稿；**用于下一步修改原 `PlaneLatentSDF(1).zip`，本文件本身不代表已经修改代码或完成实验。**
>
> 范围：仅 SAX；M&Ms-2 已有对齐轮廓缓存；ED / ES；Epi + Endo 心肌 shell。暂不加入 LAX。
>
> 设计原则：一切修改都必须由任务的真实几何、可观测性、信息流或已验证的问题推导，追求简洁、可解释，不以“最少改动”为目标，也不以新增组件为目的。

## 1. 当前方法主体

```
对齐的真实 SAX epi + endo 轮廓/平面几何
    ↓   计算真实二维心肌 SDF，并按区域分层建离线采样池
每张真实 SAX 的点集 (xyz, 2D myo-SDF, ED/ES phase)
    ↓
共享逐点 MLP + Mean/Max Pool → 每张 SAX 一个 128D latent
    ↓
按实际 SAX 高度执行无参数相邻 latent 线性插值
    ↓
共享 Conditional Scalar SDF Decoder [xyz, 128D latent, phase] → F(q)
    ↓
真实平面 Direct + 隐藏真实 SAX 的 LOO 几何监督
真实隐藏平面的 DINO-style EMA 蒸馏
三维 Eikonal（训练第一步起）+ 真实 SAX 的 Epi/Endo Seam
    ↓
完整三维空间查询 → Marching Cubes (level 0) → Epi + Endo 零值面
```

保留原 Plane Encoder、128D latent、物理距离插值和单标量 Decoder 结构。**不加入 LAX、不加入双 Decoder、双输出头或未经论证的新模块。**

### 1.1 心肌 shell 的目标 SDF

仅当 epi、endo 均为有效闭合且严格嵌套的轮廓时，采用（区域内部为负值）：

\[
\Omega_{myo}=\Omega_{epi}\setminus\Omega_{endo},\qquad
 d_{myo}^{2D}(u,v)=\max\bigl(d_{epi}^{2D}(u,v),-d_{endo}^{2D}(u,v)\bigr).
\]

- 心肌内部：`d_myo < 0`。
- 心腔内部及 epi 外背景：`d_myo > 0`。
- Epi 与 Endo 边界：`d_myo = 0`。
- `polygon_sdf` 必须使用实际连续线段距离；真实边界采样点标签直接设置成 **0**（避免浮点距离残差）。
- **缺失 endo 标注不等于心腔确实不存在。** 必须审计源缓存的解剖通道和有效性；缺少足够标注时，不得构造虚假的完整 shell SDF。
- 已对齐的真实物理坐标保留；坐标和距离使用固定 `100 mm / normalized unit` 缩放，不能按病例 bbox 独立缩放大小。

## 2. 离线候选点池：每张真实 SAX 8192 个

每张 SAX 按 **采样来源** 建立七个类别的独立点池，保存 `(xyz, d_myo^2D, sampling_source)`；取出时重新/直接利用依据真实轮廓计算的 SDF 标签。

| 采样来源 | 离线候选数 | 每次 Encoder 取数 | 每次 Decoder 监督取数 |
|---|---:|---:|---:|
| Epi 真正零值点 | 512 | 128 | 64 |
| Epi Gaussian 邻域 | 1536 | 384 | 128 |
| Endo 真正零值点 | 512 | 128 | 64 |
| Endo Gaussian 邻域 | 1536 | 384 | 128 |
| Myocardium 内部 | 2048 | 512 | 128 |
| Cavity 内部 | 1024 | 256 | 64 |
| Epi 外 Background | 1024 | 256 | 64 |
| **总计** | **8192** | **2048** | **640** |

**区别**：`sampling_source` 是采样的来源，而不是 SDF 正负的真值。例如 Epi Gaussian 扰动后可能穿进 cavity，SDF 必须依真实 epi/endo 轮廓重算，不能照来源赋符号。

### 2.1 边界零值点

- 沿真实 Epi / Endo 轮廓**按弧长采样**；每条边界建立 512 个候选，训练每条边界给 Encoder 128 个，零值监督另外抽 64 个。
- 为避免重复的固定 contour vertices，512 个候选应是沿连续 polygon 线段的弧长位置采样，而不是复制 256 个原点以凑数。
- Epi 和 Endo 的两个零值池分别管理。

### 2.2 Epi / Endo Gaussian 邻域

保留原采样尺度（单位 mm）：

\[
\sigma\in\{1,3,6\},\quad P(\sigma)=(0.4,0.4,0.2).
\]

每个边界中心点沿 contour 弧长取样，二维平面内加各向同性高斯扰动 `N(0, sigma² I₂)`，**暂不采用依壁厚调整 sigma 的规则**。

对每张 SAX 的每个 Epi 或 Endo Gaussian 池，各生成 1536 个不同的候选点；每次 Encoder 取 384，监督取 128。实际 384 个中各尺度的期望数约为 154/154/77（具体随机或整数配额实现须保证总计 384）。

### 2.3 Myocardium / Cavity / Background 区域点

- Myocardium：真实 epi 内、真实 endo 外的面积内部采样，排除边界。
- Cavity：真实 endo 内部采样。
- Background：真实 epi 外部、覆盖目标平面合理 bounding rectangle 的区域采样；初始空间范围可沿用原 bbox 每方向 `±10 mm` 扩展。
- 可以用拒绝采样实现，确保足够候选点；数据异常时明确报错，不静默把缺失类别替换成其他类别。
- 所有内部点通过真实两条轮廓计算连续二维 SDF；不是分类头的类别标签。

### 2.4 训练阶段抽样和可复现性

- 每个训练 iteration，从各独立候选池按表格配额随机抽样并打乱，合计 Encoder 输入 **2048** 点。
- Decoder 监督单独抽样 **640** 点：512 个非边界 + 128 个真实边界零值点。Direct 和 LOO 对于同一目标平面使用**同一监督坐标**，仅条件 latent 不同。
- Student/Teacher 视图采用同一分层规则，但随机子集独立采样；严禁被隐藏的真实 SAX 点进入 Student 的可见编码输入。
- Train 随机抽样；Validation/Test/mesh inference 使用固定种子或确定性子集，且 Encoder 输入数量与配额保持 **2048**，避免 Mean/Max Pool 对点数变化的敏感性。
- 尽可能在同一 step 对 Encoder/Decoder/Seam 子集执行无放回取样，但**独立抽样**的含义是不同样本视图，不必强制同一步绝对零重合；优先正确配额和稳定实现。
- 离线 NPZ 应保存物理几何、中心/尺度与采样来源元数据，可审计类别点数、坐标和 SDF 符号。

## 3. 真实平面 Decoder 几何监督：每 plane 640 点

已确定的几何目标：一个标量场同时恢复 Epi、Endo。

**零值损失**（Epi/Endo 分别计算均值，再等权）：

\[
L_{zero}^{myo}=\tfrac12\mathbb E_{q\in\Gamma_{epi}}\rho(F(q))+
\tfrac12\mathbb E_{q\in\Gamma_{endo}}\rho(F(q)).
\]

Epi、Endo 各取 64 个零值监督点。零值 loss 的贡献**不根据零值点占总查询比例简单折算**，仍是独立 loss 项。

其余 512 个非边界 SDF 监督点沿用原 `observed_loss()` 结构，输入的 `d2` 与真实 inside/outside 符号改为 **myocardium SDF**。

**特别标记：原 `observed_loss()` 的自缩放距离约束和 `factor > 0.15` 梯度门限，用户决定目前保留。** 待新训练跑完后，观察从 step 0 开启 Eikonal 的实际效果，再决定是否修改；不得在本轮部署中自行替换新的距离监督公式。

## 4. 3D Eikonal Query Sampling

**目的**：约束真正的三维连续隐式场；不为无 GT 的层间位置制造 SDF 标签。

- 50%：SAX 覆盖体积中的 3D Uniform Volume；轴向高度 `h` 在真实 `s_1 ... s_N` 区间均匀采样，平面内覆盖全部真实轮廓的联合 bbox 与合理 padding。
- 50%：真实 Epi/Endo 边界引导的 3D query。先选高度和相邻真实 SAX，再从一侧真实 Epi 或 Endo 独立抽取边界点，沿共同 SAX 轴移到目标高度，并添加平面内扰动；Epi/Endo 可各占这一部分的一半。
- gap 的选择概率与物理层间距成比例：`P(gap i) = (s_{i+1}-s_i)/(s_N-s_1)`；再在该 gap 内均匀采样高度，从而总体沿轴向均匀。
- 不使用 `(1-t)*b[i,j] + t*b[i+1,j]` 同序号点对应，也不将层间 proposal 视为真实曲面点或 `F=0` 标签。
- Eikonal 使用完整条件场对 query 坐标的导数（包括 latent 随查询高度变化的链式路径），计算 `(||grad_q F|| - 1)^2`。
- **从第一个 optimizer step 启用 Eikonal**，不与 DINO/Seam 共用一个延迟启用的 ramp；`lambda_eik > 0`，但**具体系数尚未决定**。
- **每 step 的 Eikonal 总 query 数尚未决定**，应作为明确可调配置记录；不得编造“已经讨论确认”的固定数值。

## 5. Seam Loss：只在真实内部 SAX 的 epi/endo 边界

**明确删除** `gap_geometry_loss()`（通用层间法线/曲率平滑），只保留 Seam。

- 每个具备上下真实 SAX 邻居的有效内部 plane，沿弧长从 Epi 取 16 个、Endo 取 16 个真实边界点。
- 对同一物理边界点 `q`，在 SAX 轴 `a` 上采用 `q ± epsilon*a` 的**小偏移有限差分形式**。本轮并未决定更换为解析单侧梯度。
- `epsilon_mm` 初始 **0.1 mm**，同时 `epsilon_mm ≤ min(adjacent_gap_mm)/20`；换算到 `100 mm` 归一化坐标，0.1 mm 对应 0.001。
- `n± = grad F(q±) / (||grad F(q±)|| + delta)`，`l_seam = 1 - dot(n-,n+)`（在数值稳定实现中处理近零梯度与 dot 范围）。

\[
L_{seam}^{myo}=\tfrac12\mathbb E_{\Gamma_{epi}}l_{seam}
+\tfrac12\mathbb E_{\Gamma_{endo}}l_{seam}.
\]

- 不将 epi 法线与 endo 法线互相匹配；只比较**同一边界点**在实际 SAX knot 两侧的法线。
- 仅在有有效两侧邻居时计算；排除数据无效或已知真实不光滑的边界点。
- 若 endo 标注缺失，不能直接假设解剖没有 endo；须遵守有效性语义。若边界确实不存在且有证据，损失只由有效边界组成，权重重新归一化而不是把缺失项视为零损失。

## 6. DINO 以及初始化：确定删除项

### DINO

**彻底删除无 GT gap 的 Teacher 概率线性插值伪目标**：

```
target_gap = (1-a) * teacher_prob[k] + a * teacher_prob[k+1]
```

没有真实中间 plane Teacher 目标，不能把概率插值当几何监督。**仅保留真实隐藏 SAX 的同位置 DINO-style EMA Teacher/Student 蒸馏**：Teacher 看真实目标 plane，Student 只能利用可见 SAX 的邻层 latent 插值。

### 初始化

彻底删除/禁用原 paired-directions / sphere 几何初始化的调用及其特定权重配置（尤其不能人为将 latent 通道初始化压制到 `1e-4`）；采用正常的框架随机参数初始化。训练仍正常更新 Encoder/Decoder，并非固定 Decoder 权重。

### 训练组合示意

\[
L_{total}=\lambda_{self} L_{self}+\lambda_{LOO}L_{LOO}
+\lambda_{E}L_{eik}
+r(t)(\lambda_{D}L_{DINO,true\ masked}+\lambda_{S}L_{seam}).
\]

示意保留原方案中其他项的训练权重/调度接口；**只保证 Eikonal 不被 `r(t)` 延迟**。具体权重及其他 warmup 未在本次讨论中重新确定。

## 7. 输出与边界

- 只在真实 SAX stack 的轴向有效范围内查询（本版本不加入 LAX）。
- 使用共享单输出标量隐式场，Marching Cubes `level=0` 获得 Epi + Endo 零值面。
- 不只保留最大连通分量；两侧零值面可能是不同连通分量。
- 不擅自构造真实心尖/心底几何或强加封口。

## 8. 部署时的数据与实验检查

此处是实施/验证要求，不是新增网络或新增训练损失：

1. **数据通道审计**：旧 adapter 只读取 `channel 0` 的 epi。先确认源 cache 的 epi/endo 索引、闭合性、平面映射、真实缺失与真实不存在的区别；不擅自将 channel 1 判为 endo。
2. **缓存一致性**：按患者/ED/ES 固定划分；每张 SAX 的 8192 候选池分类点数精确、inside/outside 符号正确、零值点 `d=0`、归一化单位一致；保留可逆 world-mm 映射。
3. **隐藏层泄漏审计**：原数据 center 由整例全部真实轮廓计算，可能将隐藏平面的全局统计泄漏给 Student。对严格 masked-plane 测试需避免这种隐蔽信息通路；它是数据协议正确性问题，不是新增架构模块。
4. **训练数值观测**：追踪逐 step / epoch `Eikonal`、`median(||grad F||)`、预测场幅度、近场比例与旧 `factor > 0.15` 通过率，验证 Eikonal 首步启用后是否真的解决旧场尺度退化。
5. **几何评价**：分别测试真实 SAX 的 Direct 与内部 SAX 的 LOO、Epi/Endo 边界误差、心肌符号正确率、心腔是否保留、层间 mesh 开裂/合并与表面连续性。
6. **训练/推理点数统一**：Train/Val/Test 的 Encoder 均 2048 点且同配额；Val/Test 使用确定性抽样。

## 9. 已确定 vs 尚未固定

**已确定**：SAX-only；单标量 myocardium SDF；Encoder 2048；离线 pool 8192 且分层；零值点 Encoder Epi/Endo 各128；Decoder 512非边界+64/64边界；Gaussian 1/3/6 mm（0.4/0.4/0.2）；Seam Epi/Endo 各16、等权、小偏移0.1 mm初值；3D Eikonal 50% uniform + 50% boundary-guided；Eikonal step0；删除 gap DINO 概率插值、删球形初始化、删 Gap Geometry Loss；保留核心网络和潜变量插值。

**作为初始配置而非已验证最优**：Gaussian 比例、2048/8192 容量、Decoder640与Seam32、Eikonal 50/50、Seam 0.1mm 都是目前认可的首版训练设置，性能仍需验证。

**刻意不修改/尚未决定**：原 `observed_loss()` 的自缩放距离项与梯度筛选；Eikonal 每 step 查询总数、Eikonal 相对 loss 权重；任何关于新型 LAX 融合的方案。禁止把这些未决定项写成用户确认的硬性规格。
