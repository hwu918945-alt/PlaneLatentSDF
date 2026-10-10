# PlaneLatentSDF：SAX + LAX 心肌 shell 逐项修改方案

版本：2026-10-10。对象：提交的 `PlaneLatentSDF(1).zip`。用途：交给实现 agent 修改现有项目。

**本文件给出一个确定的主方法、推导、数据契约和逐文件修改要求，不提供实验矩阵。** 数学组件已经做数值核验；修改后的完整 PyTorch 网络和 MnM2 训练尚未执行。初始超参数是部署起点，不是经过真实数据调参的最优值。

## 1. 最终决定：保留什么，修改什么

保留现有研究路线：共享逐点 MLP 编码真实平面；相邻 SAX 的 latent 按物理距离插值；人为隐藏真实 SAX 的几何监督与 EMA 表征蒸馏；连续三维隐式场；Eikonal；Marching Cubes。

最终输出只有一个标量：

\[
F_\theta(q;\mathcal O,p)\in\mathbb R,
\qquad F<0\text{ 为心肌，}F>0\text{ 为心腔或外部背景。}
\]

\(\mathcal O\) 包含当前允许学生使用的 SAX 和 LAX 观测，\(p\) 为 ED/ES。零值面是心肌区域的边界，可同时包含 epi、endo 和按统一解剖约定定义的 basal rim。

| 当前源码/问题 | 确定修改 |
|---|---|
| adapter 只读 SAX 的 contour channel 0，声明 `LV_epi_envelope` | 恢复 epi + endo，读取真实 LAX；数据目标改为 `LV_myocardium` |
| 自缩放距离目标 `F ≈ detach(||P∇F||) d2`，并用 `factor > 0.15` 关闭监督 | 删除该距离回归及其筛选；采用真实边界零值、固定温度区域符号和投影法向方向约束 |
| Eikonal 延迟，与几何正则一起 ramp | 从第一步计算完整场的 Eikonal；真实边界和体积采样都覆盖 |
| 坐标几何初始化、latent 权重约 `1e-4` | 使用普通随机初始化和非零条件调制；仅校准输出梯度尺度，去掉初始球形几何偏置 |
| 只有上下 SAX latent | 上下 SAX 距离插值 + LAX 第三项残差修正；查询时再融合有平面位置含义的 LAX 条件 |
| `observed_loss()` 直接调用固定 latent 的 decoder | 所有监督都调用统一 `query_sdf(q, observation_state)`，包括 LAX、direct 和 masked |
| 无 GT gap 的 teacher 概率插值 | 完全删除；蒸馏只配对人为隐藏的真实 SAX |
| 用同一批已编码 latent 做 LOO | 隐藏层从整个学生 observation state 中移除；teacher 单独读取目标层 |
| 训练随机 512 点、验证完整 1024 点 | 同一个分层采样器和固定点数；验证/推理固定 seed |
| 点对点插值临时轮廓生成 gap proposals | 改为无对应关系假设的邻层边界投影混合 + 体积采样；从不生成层间 GT |
| 网格轴向范围硬限制首末 SAX | 范围来自当前可见 SAX + LAX；提取时不钳制查询坐标，不添加人工正值切片 |

两个判断需要纠正：**单一标量 SDF 并不只能生成一个外表面；单病例拟合好也不证明 plane encoder 和插值路径有效。** 当前只输出 epi 是数据目标的直接结果；单病例还允许共享 decoder 的全部参数专门记忆一个患者。

## 2. 输出解剖定义与数据前提

### 2.1 心肌区域的定义

对具有完整闭合 epi/endo 的 SAX 平面，设外膜填充区域为 \(\Omega_{epi}\)，心腔为 \(\Omega_{endo}\)。要求 \(\Omega_{endo}\subset\Omega_{epi}\)。

\[
\Omega_{myo}=\Omega_{epi}\setminus\Omega_{endo}.
\]

心腔和外部背景都为正值，但在采样、编码和评价中仍保留各自的区域 ID。这个 ID 是输入标签/评价分组，不是另一个预测头。

若 \(d_{epi}^{2D}\)、\(d_{endo}^{2D}\) 都采用内部为负的符号约定，则对满足嵌套条件的平面区域：

\[
d_{myo}^{2D}(u,v)=\max\left(d_{epi}^{2D}(u,v),-d_{endo}^{2D}(u,v)\right).
\]

更一般、更稳妥的实现：先构建真实心肌区域及其完整边界集合，再用“到所有有效边界线段的最短距离 + 心肌内外判定”计算距离。不要在非嵌套、开口、缺标注时机械使用上式。

内膜处心肌外法向指向心腔，外膜处指向外部背景；这两个方向不能都使用相同的 polygon 外法向。

### 2.2 缺失内膜不等于内膜不存在

必须区分：

1. `known_absent`：完整标注明确该切面没有心腔，例如真实 apex 端部；可以使用外膜内的实心心肌区域。
2. `unannotated/invalid`：内膜未知；只能监督可信边界和可信区域，不能把整个 epi 内部当作心肌。
3. `present`：有有效 endo，按 shell 处理。

同样，轮廓为空不能直接推断整张平面都是背景。只有完整标注、有效 FOV 明确为空时，才是可信背景平面。这样的真实空平面应该保留，不能沿用当前“只保留 contour_valid”过滤规则将它删除。

### 2.3 LAX 的开口与 basal rim

必须读取实际缓存语义：LAX 是闭合心肌区域、开口 epi/endo 曲线，还是带完整 segmentation mask 的平面。**不能简单删除 `contour_closed` 条件后，把开曲线首尾相连再当真实轮廓。**

统一的数据目标选择为：**在解剖 basal rim 截止的心肌组织体**。rim 闭合的是心肌壁厚的端面，不是横跨心腔的实心圆盘。

具体规则：

- 完整 LAX 心肌 mask 已包含标注约定下的 rim：按该 mask 的区域边界处理，保留 rim 标签。
- 有真实开口 epi/endo 曲线及可信 basal endpoint 标签：将同侧 epi/endo basal endpoint 连接成壁厚 rim，构建心肌区域。epi 与 endo 主曲线保持其真实几何，不横跨整个 LV cavity 封成心肌。
- epi/endo apex 不重合时，保留真实 apex 壁厚；不能把它们强制合并成一个尖点。
- 多视图 basal landmarks 足够且非共线时拟合共同 basal plane，供数据约定检查和几何描述使用；单一 LAX 不能唯一确定三维 basal plane。该情况下以 SAX stack axis 为法向、以可信 basal endpoints 的轴向位置作稳健估计，明确记录这是方向先验。
- 有开口曲线但没有区域 mask、端点语义或其他足以确定心肌内外的信息：保留真实曲线零值约束，不生成虚假的区域标签和 rim。该样本标记 `basal_definition_complete=false`。

当前方案默认交付闭合心肌组织体；数据缺少上述 base 信息时，agent 必须在转换报告中明确列出缺项，不能用提取盒边界补齐。瓣口相关尖角不接受光滑法向正则。另行输出 epi/endo 的临床开放曲面属于从完整组织边界作语义分离，不改变单标量预测头。

### 2.4 多视图相容性

同一三维位置只对应一个 \(F\)。在 SAX/LAX 交线上检查区域标签和边界交点是否一致。容差来自实际像素间距、轮廓量化误差和对齐精度，必须记录物理毫米值。

超出容差的冲突不靠“给两个视图两个 decoder”解决：修正坐标/对齐，或将确实不可靠的局部标注标为 unknown。不要删除整个 LAX，也不要让网络同时拟合相反标签。

不把 LAX 的 2D SDF 距离和 SAX 的 2D SDF 距离在交线上视为相等；只要求可信零值几何和符号相容。

## 3. 坐标、平面与 cache v2

### 3.1 物理坐标

保留既有 aligned world 坐标，统一固定尺度 \(L=100\,mm\)：

\[
q=\frac{x_{mm}-c}{L},\qquad f=\frac{s_{mm}}{L}.
\]

\(c\) 由当前可见观测按视图平衡计算：每张平面先作轮廓统计，SAX 与 LAX 两个视图组再平衡，避免 SAX 数量主导。masked 路径使用其可见观测得到的中心，并将 hidden targets 转到同一个学生坐标系；teacher 目标层也使用这个坐标系。不得用 hidden contour 决定中心、bbox、extent 或采样 proposals。

从旧缓存通过 `points * static.scale + static.center` 还原毫米，是解码存储坐标；之后仍重新建立本模型的可见观测归一化。不能把旧的每病例 `scale` 直接当网络尺度，消掉个体尺寸差异。

固定已有 aligned cache 的坐标变换作为本任务输入条件。留层监督应称为“固定预对齐坐标下隐藏轮廓重建”；若旧对齐本身利用全 stack，它并不等价于完全不见目标轮廓的原始图像对齐任务。本修订解决模型条件泄漏，不虚称重做了历史对齐过程。

### 3.2 平面表示

每个平面存 \(o,u,v,n\)，满足 \(u,v,n\) 正交单位长度，\(n=u\times v\)。

\[
a_j(q)=(q-o_j)^Tu_j,\quad b_j(q)=(q-o_j)^Tv_j,\quad d_j(q)=(q-o_j)^Tn_j.
\]

SAX 采用已验证的共同 stack axis \(a\)，\(h=q^Ta\)，层位 \(s_i=o_i^Ta\)。平行近似只对通过几何检查的 SAX 生效；LAX 永远使用自己的真实平面参数，绝不塞入 SAX 的高度排序。

SAX 非平行误差超出原数据的允许阈值时，应在 adapter 报告并处理，不能在 `load_case()` 中悄悄投影后假称完全一致。

### 3.3 v2 必须有的字段

建议存一个 case-phase 的结构化 record；平面数量和轮廓长度可变，不再假定固定 `[N,P,3]` 能覆盖所有视图。

| 字段 | 内容 |
|---|---|
| `schema_version` | `2` |
| `case_id / split / phase` | 保留既定 285/36/36 患者划分；ED/ES 共享患者 split |
| `anatomy / sign_convention` | `LV_myocardium` / `negative_inside_myocardium` |
| `world_frame / source_transform` | 原 aligned frame、毫米还原信息和来源 |
| `sax_axis` | SAX 单位 stack axis |
| `planes[*].view_kind / source_index` | SAX / LAX2CH / LAX4CH / OTHER_LAX；真实来源索引 |
| `origin_mm / u / v / normal` | 完整平面坐标系 |
| `epi / endo / rim` | 轮廓/曲线，闭合性，有效性，边界 role |
| `endo_state` | `present / known_absent / unknown` |
| `region_geometry` | 心肌/心腔/背景区域 polygon 或实际平面 mask 与 affine |
| `valid_fov / annotation_complete` | unknown 与背景必须分开 |
| `normal_valid / corner_mask` | 排除点状 apex、标注接缝尖角等不适用法向约束的位置 |
| `basal_definition_complete` | 是否具有按解剖约定定义 rim 的信息 |
| `source_label_mapping` | 明确绑定来源 channel/label 的解剖含义 |

**不能假设 `contours[...,1]` 必然就是 endo。** 当前 ZIP 没有完整原缓存定义；agent 必须从实际元数据或来源工程核验 label mapping，并在 manifest 写死经过核验的对应关系。

禁止从未观测三维 mask、dense mesh、原方法模板或 hidden plane 生成 encoder 输入、全局摘要、query bbox 或层间 SDF GT。平面上的已允许 segmentation 可以用于获得该平面的区域信息；这与使用完整三维 GT 不同。

## 4. 修改后的网络架构

### 4.1 结构总览

```text
真实 SAX/LAX 平面分层采样
    → 同一个 shared point-wise MLP + Max/Mean pooling
    → 每平面 128D latent

SAX：z_minus,z_plus + 物理高度 → 距离插值 z_I(h)
LAX：全部可见 LAX latent → 视图平衡摘要 g_L
    → z_I(h) + A(z_I(h),g_L,h,可靠性,平面轴,phase)
    → 连续 128D SAX-plane token z_S*(h)

LAX latent + query 的真实平面投影/距离
    → 共享位置条件 MLP + 平面距离加权
    → 128D query LAX 条件 y_L(q)

[z_S*(h), y_L(q),可靠性与可用性]
    → 条件 MLP（128D）
query xyz → 条件调制的共享 MLP decoder
    → 一个三维标量 F(q;O,phase)
    → level=0 mesh
```

LAX 和 SAX 使用相同编码器、同一个解码器、相同区域监督规则。它们的不同在于平面空间组织和条件构造：SAX 排序插值，LAX 保留独立平面并提供长轴信息。

这是现有 plane-latent 方法的条件构造修订，不引入另一个 global SDF 网络、双输出头、template deformation 或虚拟平面 encoder。

### 4.2 共享 Plane Encoder

每平面固定 1024 个编码输入点。使用与推理一致的 sampler；监督 query 使用独立随机样本，不与编码点绑定。

建议每点 21D 输入：

\[
[q_{xyz}(3),uv(2),d_{myo}^{2D}(1),region(3),n(3),phase(1),view(3),boundary\_role(4),distance\_valid(1)].
\]

- `region` 为 myo/cavity/background 的 one-hot；unknown 全零并以 valid mask 屏蔽，不能冒充已知背景。
- `view` 为 SAX/2CH/4CH；其他 LAX 的扩展映射必须显式配置。若新增 view，更新输入维度而非静默映射错类。
- `boundary_role` 为 epi/endo/rim/none；边界点的 region 可以全零。
- `d2` 来自真实平面几何，用作输入特征，不强迫 decoder 输出等于它。距离无定义时置零且 `distance_valid=0`。
- 距离截断范围为 ±15 mm 再除以 \(L\)，避免很远的背景距离压过局部几何特征。

逐点 MLP：`21→64→128→128→256`，SiLU。有效点 Max + Mean pooling：`512→256→128`。输出记为 \(z_i\)。padding 必须在 Max 和 Mean 中都正确屏蔽；Max 的无效输入为 `-inf`，Mean 按实际有效点数归一化。

不要在每张平面上单独缩放到单位圆/单位盒；真实壁厚与大小信息必须保留。名称为 `SharedPlaneMLPEncoder`，不是 PointNet。

### 4.3 SAX 距离插值

对 \(s_k\le h\le s_{k+1}\)：

\[
t=\frac{h-s_k}{s_{k+1}-s_k},\qquad
z_I(h)=(1-t)z_k+t z_{k+1}.
\]

这是原代码正确的部分。左层权重对应到右层的距离，右层权重对应到左层的距离。

对 stack 外：定义连续的端点常值延拓 \(z_I(h)=z_0\) 或 \(z_{N-1}\)，但同时给出衰减可靠性：

\[
d_{out}(h)=\max(s_0-h,0,h-s_{N-1}),\quad
r_S(h)=\exp[-(d_{out}/\ell_S)^2].
\]

初始 \(\ell_S=10\,mm\)。这不是线性外推的真实解剖 latent，只是最近观测摘要；外部几何由 LAX 修正项和 decoder 条件继续预测。

至少一张 SAX：\(N=1\) 时 \(z_I(h)=z_0\)，可靠性按到该平面距离衰减。没有上下层时不计算 gap/内部留层项，但 LAX 条件和三维场仍可查询。\(N=0\) 不属于本主方法的数据契约。

### 4.4 第三项 LAX：修正连续 plane token

将每个可见 LAX latent 先按 LAX view group 平衡 pooling，再作 Max+Mean 和 `256→256→128` MLP，得到 \(g_L\in\mathbb R^{128}\)。没有 LAX 时取零向量且 availability 为零；MnM2 主方法要求读取实际存在的 LAX。

\[
\boxed{
z_S^*(h)=z_I(h)+A_\phi\bigl([
\operatorname{LN}(z_I(h)),g_L,h,r_S(h),a,p,m_L
]\bigr)
}
\]

\(A_\phi\) 为 `263→256→128` MLP：128+128+1+1+3+1+1=263。实现必须按 schema 自动计算维度，并 assert。

残差输出不使用零初始化，不把整个 LAX 分支初始缩小到 `1e-4`；可使用正常 Xavier 初始化后将最后层 gain 设为 0.1。

这里正面落实“上下 SAX + 第三项 LAX”：\(z_I\) 保留上下层距离插值，\(A\) 学习 LAX 对该高度平面表征的补充。

**连续性要求：** \(A\) 的输入禁止包含在 knot 处突然重置的 `t`、gap index、单独的左/右距离或切换式 left/right token。即使 \(z_I\) 连续，这些输入仍会使最终场跳变。上述 \(z_I,h,r_S\) 都连续，因此连续 MLP 的输出连续；分段导数的接缝用弱 seam 项处理。

\(z_I(s_i)=z_i\) 是插值的性质；\(z_S^*(s_i)\) 经过 LAX 修正后不必等于 \(z_i\)。不得在论文或代码报告中混淆这两者。

### 4.5 查询点获得 LAX 空间条件

每个 LAX 平面的查询特征：

\[
\ell_j(q)=B_\psi([z_j^L,a_j(q),b_j(q),d_j(q),n_j,view_j])\in\mathbb R^{128}.
\]

`B` 共享于全部 LAX；上述输入为 128+2+1+3+3=137，采用 `137→256→128`。

使用纯几何权重，不称 attention：

\[
b_j(q)=\exp\left[-\frac{d_j(q)^2}{2\sigma_L^2}\right],\quad
w_j(q)=\frac{b_j(q)}{\sum_l b_l(q)},
\]
\[
r_L(q)=1-\prod_j(1-b_j(q)),\quad
y_L(q)=r_L(q)\sum_jw_j(q)\ell_j(q).
\]

初始 \(\sigma_L=30\,mm\)。`w` 用 log-softmax 实现以防 underflow；`r_L` 独立表达支持强度。只有一张 LAX 时，归一化权重恒为 1，**仍由 `r_L` 保留离平面的距离效应**。没有 LAX 时，直接返回零特征、零可靠性，不能 softmax 空数组。

本融合距离只表达条件位置，不把 LAX 平面外的投影 mask 当作三维标签。越出 annotation FOV 的 query 不获得该平面的符号监督。全局 \(g_L\) 保留形状上下文，查询条件只提供局部位置补充。

### 4.6 单头 Decoder 与初始化

构造：

\[
c(q)=C_\eta([z_S^*(h),y_L(q),r_S(h),r_L(q),m_L,p])\in\mathbb R^{128}.
\]

输入 128+128+4=260，`260→256→128`，SiLU。空间分支 `Linear(3,256)`；接 6 个 256D 条件调制块，Softplus(beta=20)，最终 `Linear(256,1)` 输出有符号值，不加 sigmoid/tanh。

每块使用实际 FiLM 形式：

\[
u_{l+1}=\operatorname{Softplus}(W_lu_l+b_l),
\quad u_{l+1}\leftarrow (1+\gamma_l(c))\odot u_{l+1}+\beta_l(c),
\]

其中 \(\gamma_l=0.1\tanh(G_lc)\)，\(\beta_l=0.1H_lc\)，条件权重普通非零随机初始化；在中间块拼接一次 `[u,c,q]` 后投回 256D。可使用残差连接，但各层形状必须由配置明确。这里确实是 feature-wise affine modulation，称 FiLM 是准确的。

不用病例 ID embedding、不优化每例专属 latent、不将坐标分支额外连到一个不受条件影响的独立输出头。坐标仍然需要输入；任何条件网络仍可能在训练中忽略条件，架构不能提供“不塌缩”的数学保证。

初始化替换 `initialize_decoder()`：

1. 普通 Xavier/适合激活的随机初始化全部线性层；条件路径保持非零。
2. 只用训练数据可见范围的随机坐标与普通初始编码，测量完整场 \(\|\nabla_qF\|\) 的中位数。
3. 必要时整体调整最终线性层的 weight/bias 尺度，使初始梯度为非零且接近单位量级；乘数范围 `[0.1,10]`，需要超界则报初始化问题，不无限放大。
4. 不令输出初始为固定球，不设置 latent 权重为 `1e-4`，不使用全零输出权重。

这只是改善优化入口，不是预训练、模板或几何监督。记录初始化空间梯度、条件 Jacobian 和随机 latent 替换影响；非零影响仅证明路径已接通，不证明影响具有解剖意义。

## 5. Loss1：删去反馈型距离监督，分别约束几何与尺度

### 5.1 为什么不能把 d2 当 d3

真实 3D signed distance 限制到某个平面，其绝对值通常小于该平面到截面轮廓的距离：最近三维表面点未必在平面内。

局部一阶关系 \(F\approx\|P\nabla F\|d^{2D}\) 需要表面附近、截面可微且非切触等前提；不是全平面精确 GT。旧代码的问题不仅是阈值，还在于预测梯度参与生成当前距离目标，使缩小整个场不被该距离项辨别。

对倾斜平面 \(F(x,y,z)=0.6x+0.8z\)，在 \(z=0\) 上有 \(d^{2D}=x\)。将 \(F\) 乘 \(\epsilon\)，自缩放距离残差仍为零，而平面梯度从 0.6 变为 \(0.6\epsilon\)；当 \(\epsilon=0.1\) 就退出旧阈值监督。

所以本方法**不再回归 `factor*d2`，也不改成全点 `F=d2`**。保留 d2 的数据用途：encoder 输入、近边界采样、构造可信符号。

### 5.2 真实边界零值

设 \(\delta_0=1\,mm/L\)，使用归一化 Huber/SmoothL1：

\[
L_0=\mathbb E_{q\in\Gamma_{epi}\cup\Gamma_{endo}\cup\Gamma_{rim}}
\operatorname{SmoothL1}\left(F(q)/\delta_0,0;\beta=1\right).
\]

按 epi/endo/rim 分组归一化后再平均，避免某一长轮廓淹没另一条。未定义的 rim 不进入该项；尖角能接受零值监督，但不接受光滑法向约束。

### 5.3 固定温度区域符号

设 \(y=-1\) 表示心肌、\(y=+1\) 表示心腔/背景。固定 \(T_s=1\,mm/L\)：

\[
L_{sign}=\mathbb E_{known}\operatorname{softplus}(-yF(q)/T_s).
\]

等价于对单一标量的区域分类监督；没有另一个分类头。采用 `softplus` 保持数值稳定。

- 心肌组占 1/2 权重，正值组占 1/2；正值组内 cavity/background 平衡。
- 按该平面实际已知区域重新归一化；某区域真不存在时不凭空生成样本。
- 位于标注不确定带的点不做符号监督，初始带宽 `0.5×实际最小像素间距`；精确边界点由 \(L_0\) 监督。
- unknown/FOV 外完全屏蔽。
- 温度固定物理毫米值，不能设成预测梯度、预测 F 标准差或自适应预测距离。

这里不把二维距离当三维距离下界，也不施加 `|F| >= |d2|` 的不成立约束。固定温度只是损失对 signed scalar 的优化尺度。

### 5.4 投影法向方向

对真实光滑边界，已知平面内心肌外法向 \(\nu^{2D}\)。完整场梯度为 \(g=\nabla_q F\)，平面投影：

\[
P_jg=g-(g^Tn_j)n_j.
\]

\[
L_{dir}=\mathbb E_{normal\_valid}
\left[1-\left\langle\frac{P_jg}{\sqrt{\|P_jg\|^2+\epsilon^2}},\nu^{2D}\right\rangle\right].
\]

\(\nu^{2D}\) 已由 plane basis 嵌入三维。初始 \(\epsilon=10^{-6}\)。只对已知光滑有效边界计算，不用模型预测 `factor>0.15` 决定是否保留。对点状/切触 apex、polygon 角点、rim 转角关闭此方向项。

该项限制投影方向；**不强迫三维法向等于二维法向，也不强迫投影长度为 1**。后者会把 surface 拉成与切片垂直。

最终观测损失：

\[
L_{obs}=L_0+0.5L_{sign}+0.1L_{dir}.
\]

这些系数是起始配置。每项先按实际有效数量平均，再按平面、视图组和患者平均，不对所有点粗暴求和。

### 5.5 场尺度由 Eikonal 固定

\[
L_{eik}=\mathbb E_q\left(\|\nabla_qF(q;\mathcal O,p)\|-1\right)^2.
\]

分开统计并共同训练：

- \(L_{eik,b}\)：真实光滑 epi/endo 边界及其极小邻域。
- \(L_{eik,v}\)：完整可查询体积，包括 gaps、LAX 支持的 stack 外端部和背景。

**从 step 0 启用，初始权重各为 1.0；不与 DINO/曲率共用 warmup 开关。** masked observation state 也要计算其对应完整场的 Eikonal，而不只约束 direct 场。

在 shell 的 medial axis、角点或边界最近点切换处，真实 SDF 可能不可微；理论要求是几乎处处，不是每个离散点都必须可微。均匀连续采样通常不会精确落在这些集合；已知 polygon/rim 角点不作为边界 Eikonal 的强锚点。

正确几何附近，若 \(F\) 满足单位梯度且真实边界为零，将其缩小为 \(\epsilon F\)：

\[
L_{eik}(\epsilon F)=(\epsilon-1)^2;
\quad\operatorname{softplus}(-y\epsilon F/T_s)\text{ 在正确符号区域随 }\epsilon\downarrow\text{ 增大。}
\]

因此缩小场不能同时“白拿”更小区域损失和更小 Eikonal。这个推导排除了旧目标的一条退化通道，**并不证明复杂网络的优化一定成功**；错误几何时损失还存在竞争，必须保留梯度健康诊断。

Eikonal 对完全为零的梯度可能不能单独给出可靠恢复方向，这也是不采用全零输出初始化、从第一步维持梯度尺度的理由。

## 6. 完整坐标梯度：SAX 和 LAX 都不能 detach

最终场为：

\[
F(q)=D(q,c(q)),
\quad c(q)=C(z_S^*(q^Ta),y_L(q),r_S(q),r_L(q),\ldots).
\]

\[
\boxed{
\nabla_qF=\partial_qD+J_c(q)^T\nabla_cD
}
\]

\(J_c\) 必须包含：SAX 插值高度、LAX token 修正中的高度、端部可靠性、LAX 平面投影坐标、平面距离、归一化融合权重、查询条件 MLP。

`searchsorted` 的离散区间选择可以使用 `h.detach()`；选定区间内的插值权重仍使用原始可导的 `h`。不能把插值得到的 token 或融合条件缓存成独立于 query 的常量后计算 Eikonal。

框架代码示意，完整 PyTorch 部署由 agent 实现：

```python
# TODO(AGENT): CPU NumPy 数学核验已完成；这段 PyTorch 示例未在当前环境执行。
def field_and_gradient(model, state, query):
    # state 的 observed codes 不 detach：几何正则也要更新 encoder。
    q = query.detach().clone().float().requires_grad_(True)
    with torch.autocast(device_type=q.device.type, enabled=False):
        f = model.query_sdf(q, state)  # 在内部用 THIS q 构造全部条件
        grad = torch.autograd.grad(
            f.sum(), q, create_graph=True, retain_graph=True
        )[0]
    return q, f, grad

def observed_region_loss(f, y, valid, temp_normalized):
    # 实际实现必须按 myo / cavity / background 分组归一化。
    loss_per_point = torch.nn.functional.softplus(-y * f / temp_normalized)
    return loss_per_point[valid].mean() if valid.any() else f.sum() * 0
```

模型训练参数和上述 gradient forward 保持 FP32。普通编码/普通 scalar query 可在后续显存优化时使用 AMP，但不要先把模型整体 `.half()` 再认为 `autocast(False)` 能恢复 FP32。

监督直接在真实平面位置查询；Eikonal 边界锚点在分段 SAX knot 两侧取小偏移平均，而不是把自动选择的一侧导数当作两侧都正确。初始偏移 0.1 mm，并保持不跨另一个平面/薄壁边界。

物理单位关系：\(F_{mm}(x)=LF((x-c)/L)\)，因此 \(\nabla_xF_{mm}=\nabla_qF\)。SDF 与坐标必须用同一个 \(L\)，否则 Eikonal 目标不能直接写成 1。

## 7. Loss2：恢复真实 masked-plane 的 EMA 蒸馏

### 7.1 正确配对

人为隐藏真实 SAX \(i\)，学生观测集：

\[
\mathcal O^{-i}=\mathcal O\setminus\{\text{目标 SAX 的全部几何输入}\}.
\]

teacher 用目标真实平面的独立点采样得到 \(z_i^T\)。student 从剩余 SAX 与可见 LAX 得到：

\[
\hat z_i^S=z_S^*(s_i;\mathcal O^{-i}).
\]

teacher 和 student 使用相同的学生归一化坐标系，目标 plane pose 为允许知道的 query metadata；目标 contour、面积、壁厚、bbox 不可进入学生条件。

\[
p_i^T=\operatorname{softmax}\frac{g_T(z_i^T)-c_T}{T_T},\quad
p_i^S=\operatorname{softmax}\frac{g_S(\hat z_i^S)}{T_S},
\]
\[
L_{distill}=-\mathbb E_i\sum_k\operatorname{sg}(p_{i,k}^T)\log p_{i,k}^S.
\]

同一个 \(\hat z_i^S\) 是 hidden 高度的实际重建条件组成部分；不能另造一个只供 DINO head 使用、不会进入 SDF 的预测 latent。

删除 `target_gap = (1-a)*probability[k] + a*probability[k+1]` 及无 GT gap 蒸馏。真正层间没有 teacher target，没有第二个 encoder，没有伪 SDF 标签。

### 7.2 头、温度与更新

采用官方 DINO head/centering/sharpening/EMA 的实现原语；通过局部适配改变输入为 plane latent 和配对规则，保留来源版本、文件和修改记录。命名为 `MaskedPlaneEMADistillation` 或 `DINO-style masked-plane distillation`，不能称完整标准 DINO。

起点：prototype 256；student temperature 0.1；teacher temperature 0.04；center momentum 0.9；teacher EMA 0.996 随成功 optimizer steps 向 1 变化。

- teacher 仅包含 encoder 和 projection head，不复制无用 decoder。
- target stop-gradient，teacher 不进 optimizer。
- center 用一个有效 optimizer minibatch 的真实 teacher raw logits 更新；按病例平衡，不能每处理一例就更新。
- DDP 时 all-reduce 真实 sum/count 或病例平衡后的统计。
- teacher EMA 只在 optimizer 确实成功更新后进行；AMP skipped step 和 nonfinite step 不能推进 EMA/schedule。
- 从训练进度 5% 开始，在随后 10% 进度内将蒸馏权重从 0 ramp 到 0.02；这是单次训练的 warmup，不是实验矩阵。
- center、EMA、warmup/global_step 均保存进 checkpoint。

centering 和 sharpening 不能数学上排除所有均匀 stationary point：所有 centered logits 都为零时，任何温度下都是均匀分布。因此本方案没有声称“修正配对就一定消除 DINO collapse”，也不加入单独熵最小化来人为造出低熵外观。

记录单 token 熵、跨病例概率差异、prototype 使用分布、编码 latent 的跨病例变化；即使熵降低，也不能单独证明表征有效。几何监督始终是主目标。

## 8. Loss3：弱的几何连续性，不把薄壁压平

### 8.1 主要保留 seam

分段线性 SAX 插值连续但导数可能跳变。对平面内可信光滑 epi/endo 边界点，计算完整场两侧单位法向：

\[
N_\pm(q)=\frac{\nabla F(q\pm\varepsilon a)}{\|\nabla F(q\pm\varepsilon a)\|},\quad
L_{seam}=\mathbb E[1-N_-^TN_+].
\]

初始 \(\varepsilon=0.1\,mm\)，不超过相邻最小 gap 的 1/20。包含有 LAX 支持的首末 SAX 接缝；不在真实解剖 rim 尖角上要求两侧光滑。

### 8.2 gap 法向变化只在可靠零值面附近

保留很弱的法向变化约束，删除旧的仅靠 `exp(-(F/b)^2)` 选近表面的机制。

可见场梯度达到有效尺度后，使用 detached 的一阶几何距离估计：

\[
d_{geom}(q)=\frac{|F(q)|}{\|\nabla F(q)\|+10^{-6}}.
\]

仅从 \(d_{geom}<1\,mm/L\) 且 `0.5 < ||grad F|| < 1.5` 的候选中取样。这个预测依赖筛选只作用于可选正则，**不关闭真实边界/符号/Eikonal 监督**。低梯度场不启用平滑，不将整个空间误判为 surface。

令 \(N=\nabla F/\|\nabla F\|\)，轴向表面切向 \(t=(I-NN^T)a\)，detach 采样方向；沿 \(q\pm\delta t\) 取法向差：

\[
L_{normal}=\mathbb E\operatorname{Huber}
\left(\frac{N(q+\delta t)-N(q-\delta t)}{2\delta},0\right).
\]

初始 \(\delta=0.25\,mm\)。距离 gap knot、basal corner 和已知内外壁最近点切换区过近的候选不使用；不使用固定 6 mm 步长，不把内外法向的相反方向当作待平滑的突变。候选不够则这次该正则为零，不扩大选择带硬凑数量。

各项独立统计，初始权重 `seam=0.01`、`normal=1e-4`。由于法向有限差分包含尺度放大，normal 权重必须低，不能照搬无除差分步长的权重。

启用规则：训练进度至少 10%，且过去有效步骤的 \(L_{eik,v}\) EMA <0.1；不满足时保持为零，核心监督继续。该门槛是平滑项启用策略，非宣布模型已经是精确 SDF。

不加入无差别最小表面积损失：在本任务中它可能鼓励薄心肌消失；也不拿相邻点的角度不等式当几何定理。

## 9. 采样修改：编码、监督和正则各用合适的数据

### 9.1 编码采样的统一

每平面固定 1024 点，默认配额：256 边界（epi/endo 平衡，rim 从边界预算中取少量）、256 心肌内部、128 心腔、128 背景、256 近边界带。

严格处理实际可用区域：没有心腔时将配额重分配到已知区域；未知区域不能接受配额。薄壁心肌内部用 polygon/mask 内部拒绝采样或分层像素采样，保证最低数量，不通过固定半径偏移猜 interior。

近边界带用 0.5/1/2 mm 多尺度扰动，之后重新计算实际区域标签，不能靠“向内偏移”假定没有穿过整层薄壁。法向可计算的光滑 segment 才提供法向标签。

训练、验证和 mesh 推理使用同一个 sampler、同一个配额和同一个点数。训练更新 seed；验证/推理固定 `case_id+phase+view+plane_index` 的稳定哈希 seed，不能用 Python 的进程随机 hash。

### 9.2 真实平面监督

每步从真实几何动态生成 query，初始每平面 1024 区域点 + 256 边界点。独立于 encoder 输入；固定缓存的训练点可作为 acceleration cache，但不能成为唯一永不变化的 query 集合。

完整区域 mask 的 pixel affine 和毫米 spacing 必须保留；unknown annotation padding 不属于背景。凸起和薄壁 interior 采用区域内保底配额，避免 area 小就永远采不到。

### 9.3 无 GT 三维正则点

每个 direct/selected masked state 初始体积正则点 2048：

- 40% 在当前可见 SAX+LAX 的三维 padded bbox 均匀采样，覆盖 SAX 外的端部。
- 30% 在 gap 中采样：按 \(\Delta s_k\) 分配 gap 概率，从两端 epi/endo 边界各自独立抽点，沿 axis 投影到随机高度，增加 0.5/1/2 mm 横向扰动；不使用第 j 点与第 j 点对应。
- 30% 在可见 LAX 的 epi/endo/apex/rim 邻域采样，沿 plane normal 以 1/3/5 mm 多尺度扰动。标签仍未知，只用于 Eikonal。

gap probability 初始 \(P(k)=\Delta s_k/\sum\Delta s\)，适用近似恒定 lateral 支持体积。若各 gap 横向面积变化大，按 \(\Delta s_k A_k\) 分配预算，\(A_k\) 只由可见邻层估计。大 gap 不应与小 gap 无条件获得同样预算。

无需 contour 起始索引配准，不按圆周点序假定对应关系。不因为候选来自临时轮廓/边界投影就生成虚拟区域标签。

### 9.4 查询域与背景外壳先验

由当前可见 SAX + LAX 的全部真实边界得到 bbox，每边初始加 15 mm。该 bbox 是查询/采样域，不是人工 mesh cap。

加入很弱的域外壳正值项：仅对完整 direct state，在六个 padded box 面随机采样，按 `outside` 使用同一固定温度符号 loss，权重 0.05。**这是“目标组织包含在足够宽的观测包围域中”的紧支撑先验，不是二维观测直接提供的三维 GT。** 数据存在明显 FOV 截断或 incomplete end coverage 时关闭该先验并标明几何 extent 未充分约束。

masked state 默认不使用该外壳符号先验：其可见 bbox 可能比真实目标小，尤其隐藏首末层时，不能把缺失组织区域提前标为背景。masked bbox 仍可用于提议无标签 Eikonal 采样；hidden target query 不受该 bbox 限制。

不要将盒外值硬写正、把模型 output 与 box distance 作 `max` 后仍称精确 SDF。训练中的 soft prior 与提取时强制封口不是同一件事。

## 10. 三条 forward 路径与总损失

### 10.1 路径 A：真实观测统一重建

从全部可见 SAX/LAX 构造 state \(\mathcal O\)，所有真实平面 query 都计算同一个 \(F(q;\mathcal O,p)\)。

按视图平衡：

\[
L_A=\tfrac12\operatorname{mean}_{SAX}L_{obs}
+\tfrac12\operatorname{mean}_{LAX\ groups}L_{obs}.
\]

组缺失时按实际存在的组重新归一化。不按数量让 12 张 SAX 淹没 1–2 张 LAX。

### 10.2 路径 B：隐藏真实 SAX

每病例每步选一个真实 target SAX，移除它所有学生几何输入，重新构造归一化和 state。已知 target plane pose 只用于查询。

- 有内部 SAX 时，75% 从内部选、25% 从首末选；少于三层时从可隐藏且仍至少保留一张 SAX 的层选。
- endpoint target 是为了直接训练“剩余 SAX + LAX 完成端部”；不是向未知虚拟层造 GT。
- 对被隐藏的实际心肌区域计算 \(L_B=L_{obs}^{hidden}\)。
- 同步计算 \(L_{distill}\)；layer token 与该 hidden 高度的实际几何条件一致。
- 对 masked state 的体积/Eikonal 也训练，避免只让 direct 场有尺度约束。
- masked bbox 和 extent 来自可见观测；hidden targets 是监督坐标，可以在 masked bbox 外被查询，不能因此扩张学生输入 bbox。网络本身可查询任意 q。

这一版不另造 LAX 的轴向插值器。LAX 直接监督与编码路径与 SAX 共享；“隐藏 SAX→预测 plane token”是该研究的层间训练任务，不意味着 LAX 必须拥有相同的排序结构。

### 10.3 路径 C：无 GT gaps 和 stack 外端部

所有三维候选调用同一个 field：Eikonal、弱 seam/normal、已声明的 soft compact-support prior；没有距离 GT、虚拟 plane encoder 或 teacher probability interpolation。

### 10.4 总损失起点

\[
L=L_A+L_B+L_{eik,b}+L_{eik,v}
+0.05L_{outer}
+\lambda_D(t)L_{distill}
+\lambda_S(t)L_{seam}
+\lambda_N(t)L_{normal}.
\]

边界/体积 Eikonal 分别在 direct 和 selected masked states 平均；每种状态各占 1/2，不随有效点数量改变相对权重。\(L_A,L_B\) 已包含第 5 节内部分项系数。没有合法 hidden target 时 \(L_B\) 与 distill 为零，不除以零。

\(\lambda_D\) 按第 7 节 ramp 到 0.02；\(\lambda_S\) 到 0.01；\(\lambda_N\) 到 `1e-4`，后两者遵循第 8 节健康门槛。

| 参数/路径 | 真实/hidden 几何 | 完整 Eikonal / seam / normal | masked 蒸馏 |
|---|---|---|---|
| Student encoder | 更新 | 更新 | 更新可见输入 encoder |
| LAX pooling、A/B/C 条件模块 | 更新 | 更新 | A 与其上游可见条件更新 |
| SDF decoder | 更新 | 更新 | 不更新 |
| Student projection head | 不更新 | 不更新 | 更新 |
| Teacher encoder/head | 不反传 | 不反传 | 仅目标，随后 EMA |
| SAX 距离插值/几何权重 | 无参数，但保留 query 导数 | 无参数，但保留 query 导数 | 无参数 |

## 11. 训练器与优化修改

单模型多病例训练；ED/ES 共享权重并保留 phase。沿用固定患者划分，不能把 ED/ES 分到不同 split。zero-shot 与 fine-tune 80e 的历史协议如继续使用，仍保持该定义。

起点配置：AdamW；encoder/condition/decoder 学习率 `1e-4`，projection head `1e-4`；weight decay `1e-5`，bias/normalization 不 decay；有效 minibatch 4 个不同患者，显存不足用梯度累积实现。

每个患者的 ED/ES 先平均，再跨患者平均。projection center 在实际有效 minibatch 后更新。几何高阶导数使用 FP32；开始部署时全模型 FP32，先确保数值路径正确，再优化 AMP。

梯度裁剪上限初始 5.0，先记录 unclipped norm 和裁剪系数；不能只记录裁剪后值。不要用极低上限长期把 Eikonal 更新压到近零。

每次成功 step：`zero_grad → 各患者完整 loss → backward/累积 → finite check → clip → optimizer.step → teacher EMA + center → global_step++`。失败不递增有效步数、不更新 EMA；保存失败上下文而非 `nan_to_num` 掩盖。

每个 epoch 内每患者均匀访问。记录 `optimizer_steps`、`patient_phase_exposures`、`hidden_target_counts` 和各视图预算，避免只比较 epoch 数误判每病例训练机会。

不根据“loss 继续下降”默认选择最后一轮。checkpoint 选择以真实心肌 slice occupancy 几何为主；\(|F|\times L\) 只有在距离尺度有效时才是距离代理。

这些是主方法运行规则，不要求实施消融矩阵或多组超参数搜索。

### 11.1 单病例好、多病例差在这版中的处理

单病例训练允许 decoder 参数记忆一个人的 ED/ES，也通常给该患者更多重复更新；因此单病例成绩不能单独证明 plane encoder、插值和跨病例重建能力。

本次修订针对可定位的原因：去掉压低 latent 影响的初始化；将条件反复注入 decoder；用真实区域及完整梯度阻止场幅度缩小；统一 encoder 点数/分布；按患者、phase 和视图归一化损失；记录实际曝光次数；从所有学生几何通道移除 hidden plane。

它没有假定单/多病例差异全由一个 bug 造成，也不承诺修复后两者成绩相同。多病例模型还必须学习泛化，而单病例可以专门拟合一个对象。新的条件诊断确认的是患者观测能否改变对应几何；不以一个患者高 Dice 当作该性质已成立。

## 12. Mesh 提取与 endpoint 完整性

### 12.1 推理接口

```python
state = model.encode_observations(case, sampling_seed="deterministic")
f = model.query_sdf(query_xyz, state)
vertices_mm, faces, report = extract_myocardium_mesh(model, state)
```

只用 student encoder、A/B/C 条件模块、student SDF decoder。teacher/projection head 不参与 mesh 推理。

统一 `query_sdf` 不接受 view-specific decoder bypass；同一个 q 在 SAX 评价、LAX 评价和 grid extraction 中得到相同 F。

### 12.2 查询网格

- bbox 使用全部可见 SAX+LAX 的边界范围 + 初始 15 mm padding。
- 用毫米 spacing 控制精度，默认 0.5 mm；不固定所有病例为 `160^3`。内外壁分辨不足时可局部细化到 0.25 mm。
- 保留真正 q；删除 extraction 中 `clamped[:,2]` 以及“边界外强设正值”的 cap 代码。
- Marching Cubes `level=0`，`step_size=1`，去掉退化三角形。
- 体素数组轴序、spacing、stack basis、origin 和 \(L,c\) 转换保持显式。世界 basis 行列式负时要处理 winding 反射。

不要只凭 `gradient_direction` 名字认定面朝向正确。对每个 shell 连通分量，检查面法向与 \(\nabla F\) 的内积；应朝向 F 增大的方向，即离开心肌。必要时翻转该分量 faces。inner surface 的法向朝 cavity，不是远离心脏中心；不能对全部 components 按径向方向统一翻面。

### 12.3 不删除 inner surface

提取单一心肌 SDF 的全部零值面。不得默认“保留最大连通分量”并删除较小的内膜面。若有 basal rim，内外面可能连接为一个分量；若是完整封闭腔壳，它们也可能是两个独立分量。因此“恰好一个 component”不是 shell 成功标准。

默认保存原始全部 components 和 component diagnostics；疑似小伪表面只能基于体积/支持关系另作标记，不能静默修剪成看起来更漂亮的结果。

### 12.4 边界接触的处理

若 F 的零值/负值区域接触提取盒边界，将相关方向扩大 10 mm 后重新查询，最多两次。禁止通过补正值 voxel 封口。扩域后仍接触则 report `domain_truncated=true`，不要宣称完整解剖 mesh。

LAX 帮助定义和监督 SAX 外的 apex/base，不能数学上从一两张长轴轮廓唯一决定所有未观测端部。这里通过真实 LAX 几何、endpoint masking 训练和共享数据先验进行完成。base 的组织闭合还依赖第 2 节一致的 rim 约定；这与在首末 SAX 强行截断不同。

### 12.5 输出 report

必须包括 `anatomy=LV_myocardium`、`sign_convention`、`views_used`、`observed_sax_extent_mm`、`lax_extent_mm`、实际 grid bbox/spacing、是否扩域、boundary/nonmanifold edges、各 component 面积/体积信息、face-gradient alignment、`basal_definition_complete`、`domain_truncated`、内外壁评价是否通过。

主输出命名 `myocardium_mesh.obj`；可以附 epi/endo/rim 语义视图，但不能把只含 epi 的文件继续叫完整心肌 mesh。展示需要 cutaway 或切面，避免外壳遮挡使内膜不可见。

## 13. 逐文件修改清单

### 13.1 `adapt_static_cache.py`

当前位置：仅 `view_ids == 0`、`contour_valid[...,0]`、channel 0；`anatomy='LV_epi_envelope'`。

必须：

1. 核验来源 anatomy/view label mapping，写进 adapter audit。
2. 读取 SAX + 实际 LAX，恢复 epi/endo；按完整性保留真实空平面。
3. 读取 closed/open/FOV/annotation 信息；开 LAX 不自动闭合。
4. 将原存储坐标还原毫米，保留各平面真正 basis。
5. 构建心肌区域、边界 role、内膜状态与 basal rim 完整性。
6. 校验嵌套、self-intersection、plane residual 和跨视图冲突。
7. 输出 v2 manifest，保留 570/72/72 case-phase 数量及 285/36/36 patient split；失败病例明确列出，不静默改变数据划分。

### 13.2 `core/data.py`

拆成可维护的 `geometry.py / sampling.py / data.py` 或保留同文件，功能契约必须一致。

- `polygon_sdf()` 不再声称只能填充单一 epi；新增 `myocardium_region()` 与 `distance_to_boundary_segments()`，开口/未知要有 valid mask。
- `load_case()` 读取 v2 可变数量视图，验证单标量心肌符号目标；旧 envelope 数据拒绝作为新主方法训练输入。
- `select_view()` 替换为统一分层 `sample_plane_encoder_points()`，train/val/infer 同点数配额。
- `sample_observed_queries()` 与 encoder points 独立；按真实 region 平衡。
- `gap_queries()` 删除同索引轮廓插值与固定 0.06 noise，改第 9 节 proposals。
- `build_observation_state()` 支持先隐藏目标再计算归一化、proposals 和 summaries。

### 13.3 `core/model.py`

- `PlaneEncoder` → `SharedPlaneMLPEncoder`，显式输入 schema，带 padding mask。
- 保留正确的 SAX distance interpolation，扩展端部常值摘要 + reliability，支持 N=1。
- 新增 LAX group pooling、连续 plane-token residual A、query LAX MLP B、condition MLP C、FiLM decoder。
- 新增 `encode_observations()` 和唯一 `query_sdf()`。
- `leave_one_out_latent()` 不再作为完整 masked forward 的替代；可以保留数学原语，但 masked state 必须真正移除目标层。
- `make_teacher()` 仅复制 encoder/head；修改 `ema_update()` 的模块映射，检查命名与 shape 一致。
- `DistributionHead` 接入官方 DINO head 原语并记录改动，不声称自定义全流程是标准 DINO。

### 13.4 `core/losses.py`

- 删除预测 factor 生成目标和 `factor>0.15` 筛选。
- `observed_loss(field_fn, query_batch, plane_metadata)`，通过 unified field 计算 L0/sign/dir。
- `eikonal_loss()` 内部重新用 requires-grad query 生成所有 SAX/LAX 条件；FP32，分别记录 boundary/volume。
- 蒸馏仅 hidden-real pairing；删除 distance-weighted probability target。
- `gap_geometry_loss()` 替换尺度可靠的候选机制与小步长，权重低。
- `seam_normal_loss()` 覆盖合适的 interior/end knots，排除 anatomical corners。
- 所有 empty mask 返回 graph-connected zero，避免空 `.mean()` 和 NaN。

### 13.5 `core/initialization.py`

删除 paired-direction sphere 初始化和 latent `1e-4`。实现普通随机初始化 + 有限输出尺度校准，校准只使用训练观测范围，不访问 val/test 标签。保留初始化 audit。

### 13.6 `core/training.py`

重写 `case_objective()` 为：direct state → 统一多视图监督；selected target → masked state → hidden几何和真实 teacher配对；两种 state 的完整 Eikonal；最后弱连续项。

`z_gap` 不再配任何 teacher probability。teacher target 与 hidden student 共用坐标系；目标层不进入学生 encoder/g_L/proposal/bbox。每项按病例/视图/有效数量归一化。

把 DINO、Eikonal 和 geometry 的 schedule 分开，避免旧的单一 `ramp` 一起延迟尺度约束。

### 13.7 `train.py`

- 训练/验证统一采样 policy；优化器参数组显式。
- 患者级有效 batch 与 ED/ES 平均；保存 exposure counts。
- 梯度/数值检查、成功 step 后 EMA/center；FP32 初始部署。
- checkpoint 记录 schema、anatomy、配置、条件模块、teacher/head/center、RNG 和 sampler seeds。
- 所有恢复检查包含 architecture/schema；旧权重不得静默加载进新模型。
- 提供真实 heart-mask 几何评价，不能靠低 boundary F 自动选“最佳”。

### 13.8 `core/evaluation.py`

- 从 `F<0` 评价真实心肌区域；分别报告 epi/endo section geometry 与 cavity preservation。
- Direct 与 hidden 在相同 target 集合比较；缺失、失败、没有零值面必须计入分母，不能只对成功提取病例平均。
- 增加 LAX query、首末层范围之外的 LAX 端部诊断。
- 报 boundary/volume Eikonal 及梯度 p10/p50/p90，不只输出一个平均 loss。
- latent 指标按不同患者、相同 phase、相近相对高度比较；原 `z.std(dim=0)` 只是一个病例内部层间差异，不能叫跨病例多样性。
- 真实输入、预测心肌的切面面积/厚度/occupancy 多样性在共同毫米 frame 上比较；不对任意 mesh vertex 序号求方差。

### 13.9 `extract_mesh.py / core/mesh.py / check_mesh_pair.py`

- 全部使用统一 sampler/state/field。
- bbox 融合 SAX+LAX；物理 spacing；去掉 axial clamp、人工 cap 与默认最大分量过滤。
- 检查毫米转换、basis handedness、face-gradient orientation；保留 endo/rim。
- `check_mesh_pair.py` 若原先假定一个 epi mesh 或两相位外包络，更新解剖语义；不能把 ED/ES 文件对误当 epi/endo 对。

### 13.10 `config.json / verify_core.py / scripts/verify_dataset.py`

ZIP 当前缺少实际 `config.json`；agent 必须新增完整新配置，并将 source/audit/config 与代码一起交付。

旧 `verify_core.py` 不能直接当新架构验收。替换为 schema检查、shell 符号检查、SAX权重、连续性、完整 SAX+LAX 梯度有限差分、teacher冻结、masked条件隔离、同点统一field和mesh单位/朝向检查。

禁止验收脚本从不存在的 `vendor/static_metrics.py` 静默 fallback 得出“全部通过”；提供实际依赖或准确报 missing。

## 14. 初始配置汇总

| 配置 | 主方法起点 |
|---|---|
| Target | 单一 LV myocardium signed scalar |
| Views | 所有有效 SAX + 真实 LAX |
| Fixed physical scale | 100 mm |
| Encoder | 21→64→128→128→256；Max+Mean；512→256→128 |
| Encoder points | 1024/plane，同配额 train/val/infer |
| Plane token | 距离插值 + LAX residual A；263→256→128 |
| Query LAX MLP B | 137→256→128 |
| Condition MLP C | 260→256→128 |
| Decoder | 6 个 256D 条件块；Softplus beta=20；单一线性输出 |
| Region temperature / zero scale | 1 mm / 1 mm |
| Eikonal boundary/volume weight | 1.0 / 1.0，step 0 启用 |
| Direct / masked observed weight | 1.0 / 1.0 |
| DINO-style weight | ramp 到 0.02；仅 hidden-real |
| Teacher/student temperature | 0.04 / 0.1 |
| Prototype count / EMA start | 256 / 0.996 |
| Seam / normal weight | 0.01 / 1e-4，尺度健康后启用 |
| Volume regularization points | 2048/state/phase |
| SAX exterior reliability length | 10 mm |
| LAX distance kernel sigma | 30 mm |
| Initial bbox padding | 15 mm，各方向 |
| Mesh spacing | 0.5 mm，必要处 0.25 mm |
| Optimizer / initial LR | AdamW / 1e-4 |
| Effective batch / grad clip | 4 patients / 5.0 |
| Precision | 首次部署 FP32 |

`view` one-hot 等 schema 若改变，MLP 输入维度应同步自动计算；不能仍固定抄表中的维度。

## 15. 实现验收：只验收本方法，不要求实验矩阵

### 15.1 数学与接口

1. SAX 非均匀层距插值权重方向正确、端点相等、值连续；N=1 不除零。
2. 加入 A 后 knot 左右 F 极限相等；不得因 gap metadata reset 产生值跳变。
3. 单 LAX 时 y_L 仍随离平面距离改变；多 LAX weight 有限、无 LAX 返回合法零。
4. 同一个 observation state、同一个 q，SAX/LAX评价和grid提取返回同值。
5. query coordinate autograd 与完整条件有限差分在远离 knot/corner 的点一致；分别覆盖 stack 内/外。
6. 区域符号心腔为正、心肌为负、背景为正；真实 epi/endo 均有零值边界。
7. teacher 没有梯度；EMA方向正确；无 GT gap 没有 distillation target。
8. 目标层从学生输入删除后，改变 hidden contour 不改变该 state 的 latent/中心/bbox/proposals；监督 target 和 teacher target可以改变。固定预对齐 transform 前提须保持一致。
9. 模型/optimizer/teacher/center/RNG 完整恢复；empty masks 不产生 NaN。

### 15.2 几何与条件有效性

在实际 checkpoint 上，必须记录相同 q/固定 normalization/同一观察框架下的原始条件、零条件和交换条件输出。交换不仅看 F 数值，还看心肌零值面和真实区域误差是否发生符合病例差异的变化。它是实现诊断，不是要求另训练三组模型。

只交换向量会破坏 plane position/normal 配对时，该结果仅是敏感性诊断；解剖差异验证应交换完整可比观测 state 并使用共同物理 frame。若只交换某个近似对应的 SAX token，记录实际高度、phase 和对齐假设。

必须能在切面上看到 cavity-positive/myo-negative/background-positive 的结构；外观只有平滑外壳不能算验收。inner/outer 的 component 数量并不固定。

梯度范数与 field magnitude一起报告。若仍在约 0.01 量级，属于尺度失败，不能因为 Dice 尚可就把 |F| 解释成毫米距离。初始化接通条件、蒸馏熵不均匀也都不替代真实条件几何检查。

端部验收分开看：是否超过首末 SAX、是否覆盖 LAX apex/base、是否仍接触提取盒、basal rim 是否符合数据定义；这些不是同一个指标。

### 15.3 本次已完成的验证及其边界

使用 NumPy float64，解析 Jacobian 与中心有限差分比较，检查的是本修订的结构原理，未执行完整学习网络：

| 验证 | 结果 |
|---|---|
| 单标量 spherical-shell 的三域符号与两处零值 | 通过；cavity + / myo − / background + |
| 固定温度 sign + Eikonal 对全场缩小的反应 | 正确几何样本下，scale 1→0.3→0.1→0.01 总损失递增 |
| SAX 插值、端部可靠性、LAX投影/权重和残差token的完整 Jacobian | 最大绝对误差 `3.67e-10` |
| 忽略所有条件 coordinate derivative 的对照 | 梯度 L2 误差 `15.80`，证实遗漏条件导数不可接受 |
| 连续 token 的 knot 值连续性 | 两侧相隔 `2e-7` 时最大 F 差 `1.76e-6`，随步长趋零 |
| 同一物理尺度的坐标/SDF归一化梯度 | 最大绝对误差 `2.41e-10` |
| 旧倾斜面自缩放距离目标 | 场缩小后残差仍零，旧阈值关闭，重现退化通道 |
| 256维均匀 DINO 分布 | 熵 `5.54517744`；centering/sharpening 不消除完全均匀 fixed point |

不将以上结果写成“PyTorch backward 已通过”“真实模型必然收敛”或“MnM2 shell/端部已验证”。当前运行环境没有 PyTorch/skimage，完整 backprop 和 mesh backend 检查由部署 agent 在实际环境完成。文中 PyTorch接口片段是修改规范，不是已训练主网络。

## 16. 来源与与已有方法的关系

- [CrossSDF，CVPR 2025 正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Walker_CrossSDF_3D_Reconstruction_of_Thin_Structures_From_Cross-Sections_CVPR_2025_paper.html)：与“平面轮廓监督三维 SDF”任务高度相似；与本方法的 plane latent encoder、跨病例条件学习、SAX/LAX token融合仅部分相似。本修订没有把其 symmetric-difference loss 改名据为己有，也没有照搬它的 per-object hash-grid 架构。
- [CrossSDF 作者论文全文](https://arxiv.org/html/2412.04120v3)：支持 2D/3D SDF 距离不等价、符号区域与零值面监督、薄结构内部采样的重要性。本文件的固定温度 sign + 边界梯度方案是针对当前退化问题的设计，不宣称等于 CrossSDF 原始损失。
- [DINO 官方训练与 loss 源码](https://github.com/facebookresearch/dino/blob/main/main_dino.py)、[官方 head](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)：EMA、centering、teacher sharpening 原语高度相似；几何 masked-plane 配对属于本任务适配，非标准图像多裁剪训练。
- [PyTorch autograd.grad 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)：`create_graph=True` 构建导数图，用于几何正则对网络参数反传。
- [scikit-image marching_cubes 官方文档](https://scikit-image.org/docs/0.22.x/api/skimage.measure.html#skimage.measure.marching_cubes)：spacing 和返回坐标轴序按 volume 维度对应；实际面朝向仍需本项目解析 shell/field-gradient 核验。

## 17. 给 agent 的执行要求

依次完成数据契约与 anatomy 修正、统一条件场、核心监督与尺度修复、真实 hidden-plane 蒸馏、弱连续性、统一评价与提取。这个顺序是实现依赖顺序，不是分批设计或多轮实验要求。

最终交付同一份完整方法代码、v2 adapter/schema、明确配置、数学/反向传播/mesh接口验收结果。保留旧实验来源与结果，不覆盖历史报告；新结果必须标注 `LV_myocardium + SAX/LAX`，不得与旧 `LV_epi_envelope + SAX-only` Dice 直接混成同一种解剖指标。

不能以“增加训练轮数”“调大 DINO”“扩大 marching cubes 盒子”“给外膜后处理挤出固定厚度”替代本文件规定的修复。核心目标是：条件真正控制几何；真实区域与两类边界一致；完整场具有有效距离尺度；LAX 参与未被 SAX 覆盖的端部完成。
