# SAX Plane Latent → 连续三维隐式场：点对点部署文档

版本：2026-10-09。工作目录名 `PlaneLatentSDF_Deploy` 只描述实现，不代表已有标准算法名。

## 0. 交付目标与边界

将 M&Ms-2 **已有对齐缓存中的 SAX 轮廓**作为稀疏观测：每层用共享逐点 MLP 编码成 latent；上下真实观测层按物理位置插值 latent；共享 MLP 将三维查询坐标和该位置的 latent 解码为单个连续标量场；通过零值面提取 mesh。训练包括真实层及留一层重建、基于 EMA 的距离加权分布自蒸馏、弱二阶几何约束、接缝法向约束和三维 Eikonal。

本包提供可执行参考核心、数据转换入口、训练入口、mesh 入口、数学及梯度检查。**没有读取你的真实对齐缓存源码，也没有运行 MnM2 临床训练。** agent 必须完成缓存字段映射，并在真实数据上验证效果。历史对话最后的网络结构回复未能取回，因此本文件将可见对话中的方法落实为明确实现；维度、权重、采样数和以下数学细化是本次部署选择，不能声称与未取回回复逐字一致。

以下是两项必须明示的数学修正，不能在实现时偷偷改回：

1. EMA 教师不接受梯度；真实层 SDF 监督更新的是在线学生。在线模型同时处理真实层和无 GT 的层间查询，而不是建立两个独立、只分别看 GT/无 GT 的训练网络。
2. 三维坐标输入不意味着二维距离标签自动变成三维 SDF。保留真实轮廓上的零值、内外符号以及局部校准的二维距离监督，再以三维 Eikonal 约束完整查询场。禁止把所有二维距离原样当作精确三维距离硬拟合。

主方法不包含图像、LAX、模板变形、GAN、VAE、BYOL、global heart latent、额外占据头。先只部署一个填充区域的单标量场：建议 LV epicardial envelope。若目标同时有 epi 和 cavity，分别使用相同核心训练两份单场模型；不要把两个距离相减当作心肌的真实 SDF。联合壳层拓扑并非当前包已完成的功能。

## 1. agent 的执行顺序

| 顺序 | 文件/动作 | 必须完成的验收 |
|---|---|---|
| 1 | 阅读本文及 `core/model.py`、`core/losses.py` | 理解插值反距离权重、EMA方向、完整坐标梯度 |
| 2 | `python verify_core.py` | 所有检查 PASS；出现失败先修实现 |
| 3 | 在现有项目添加缓存 → manifest 适配器 | 坐标、毫米、plane基底、病例/相位对应正确 |
| 4 | `prepare_cache.py` 转换 | 点在相应平面内；SDF负内正外；病例不跨划分 |
| 5 | 单病例可视化缓存，检查2D标签 | 真轮廓上距离为0，内外符号正确；无轴翻转 |
| 6 | 合成数据完整损失短训练，再跑真实数据短训练 | loss/梯度有限，在线encoder/decoder/head更新，教师无梯度 |
| 7 | 真实病例小规模拟合及真实层留出检查 | 看切面零轮廓、sign和梯度，不能只看低loss |
| 8 | 按固定划分训练、从验证集选checkpoint | test不参与选模；ED/ES同病例保持同一split |
| 9 | `extract_mesh.py` | mesh位置和单位正确；报告开边界和截断 |
| 10 | 返回修改清单、命令、真实指标与失败病例 | 区分实现验收与重建效果，不承诺本方法必然成功 |

代码里的 `TODO(AGENT)`、`TODO(AGENT-CACHE)` 是部署核对点。核心函数已经实现，不是待补空函数；真正未完成的是与现有缓存的字段对接、真实数据可视化及临床评估。用户提到的 TODU 在代码中统一拼写为标准 `TODO`。

## 2. 坐标、平面与单位契约

### 2.1 使用已有对齐结果

对第 i 张 SAX，使用缓存中的物理原点 o_i、单位正交基底 u_i、v_i，将平面内毫米坐标映射为：

\[
q_{mm}=o_i+u_i x_{mm}+v_i y_{mm}.
\]

如果缓存给的是像素坐标，要先通过已验证 affine / spacing 转为物理毫米。禁止重复乘 spacing、重新凭经验对齐、将 `(row,col)` 直接当 `(x,y)`、将 slice index 当 z 毫米值。

当前核心使用共同的单位 stack axis n，以 s_i=nᵀq_i 表示平面位置，按 s_i 严格升序排序。同一心脏各平面近似平行是这里的显式简化，不是通用非平行切片解法。

`prepare_cache.py` 默认仅容忍数值级不平行误差（最大位移0.01 mm）；明显不平行会失败。若明确采用共同平面近似，可传 `--project-parallel`：沿共同轴投影，并重新计算投影后轮廓的平面距离，同时保存最大投影位移。agent 必须报告该误差，不能静默把真正非平行的数据当精确平行。投影后顶点仍在连续分段线性轮廓上，点到轮廓距离使用线段距离。

### 2.2 固定尺度

\[
q=(q_{mm}-c_{mm})/L,\quad L=100\,mm;\qquad d_{2D}=d_{2D,mm}/L.
\]

c 只由该病例可用观测轮廓计算，L对所有病例固定。**不要逐病例按bbox边长缩放到同样大小**，否则病例间真实体积差异会被擦除。这里点的单位和输出场的目标距离单位一致，Eikonal目标梯度范数为1；坐标和距离同时除以同一个L时，该目标无需再乘100。

反变换：`vertices_mm = vertices_normalized * scale_mm + center_mm`。输出为已对齐缓存坐标中的毫米；若需回到原始扫描坐标，还需使用现有缓存保存的对齐逆变换，本包不猜测该矩阵。

### 2.3 真实层采样

每层只在其平面内采样：从连续轮廓均匀弧长取种子，在平面内加高斯噪声。默认σ为1/3/6 mm，比例40/40/20%；768个近轮廓点加256个平面bbox均匀点。另存256个有序轮廓零值点。不要在法向上给二维监督点加噪声后仍复用该二维距离。

`polygon_sdf` 使用到全部轮廓线段的最近距离，ray crossing决定内外，负内正外。它实现的是单个简单填充轮廓；若数据有洞、多连通域、严重自交，agent应显式扩展表示和距离计算，不能直接拼成一个多边形。

建议可视化至少：ED与ES各1例、近基底/中间/近心尖层、3D轮廓stack、随机点的SDF符号及距离。确认epi标签对应其包围的整个填充区域，而不是仅心肌环。

## 3. 缓存适配格式

适配器唯一必须把现有缓存导出的中性 manifest 形如：

```json
{
  "records": [
    {
      "case_id": "patient_001",
      "phase": "ED",
      "split": "train",
      "planes": [
        {
          "origin_mm": [0, 0, -20],
          "u": [1, 0, 0],
          "v": [0, 1, 0],
          "contour_uv_mm": [[20,0],[0,20],[-20,0],[0,-20]]
        }
      ]
    }
  ]
}
```

此片段只展示字段，实际至少需要2个不同平面；完整的留一层训练至少需要3层。`case_id` 必须为患者标识，不能把ED和ES改成不同患者。`phase`为ED或ES，`split`使用现有固定划分，不能重新随机切分。M&Ms-2当前固定病例数285/36/36；若每例都有ED/ES，文件数分别可为570/72/72，但**病例数不能按文件数报告**。

每个phase对应一个npz：

| 字段 | 形状 | 含义 |
|---|---|---|
| `q` | `[N,P,3]` | 已归一化的真实层采样三维坐标 |
| `sdf2d` | `[N,P]` | 到该层连续轮廓的归一化二维有符号距离 |
| `boundary` | `[N,B,3]` | 真轮廓上的有序等弧长点；标签隐含为0 |
| `s` | `[N]` | `q @ axis`；严格升序 |
| `axis` | `[3]` | 共同单位stack轴 |
| `center_mm` | `[3]` | 归一化中心 |
| `scale_mm` | 标量 | 默认100 mm |
| `phase` | 标量 | ED=0、ES=1 |
| `case_id` | unicode标量 | 原始患者标识；禁止object array |

同一文件内N可以变，不同文件无需padding。当前单文件内P/B相同，转换器固定生成。`load_case` 检查形状、finite、排序、平面残差、单位轴；`validate_split` 检查患者交叉泄漏。

轮廓有序点的起点/方向建议在适配器统一，如保持CCW并从共同平面正x方向附近开始。中间层的粗轮廓只是采样proposal；起点错位不会产生伪SDF标签，但会浪费采样，所以仍需核对。

## 4. 网络结构与张量流

### 4.1 Plane encoder

已知轮廓能计算2D SDF，所以采用每点 `[x,y,z,d2D,phase]`，5维输入。将d2D作为输入属于本次明确的实现选择：它让随机平面点携带形状的内外/距离信息，而不是让均匀坐标云本身承担形状识别。推理时只需要输入的稀疏2D轮廓即可算出这些值；不需要完整三维GT。

```text
[N,P,5]
→ shared point-wise MLP: 5→64→128→128，SiLU
→ 每平面 mean pooling + max pooling
→ [N,256]
→ MLP: 256→256→128，SiLU
→ [N,128]，每层z_i
```

没有T-Net；名称为共享逐点MLP编码器，不叫标准PointNet。无跨病例索引embedding、无可学习单病例latent表、无全心global pooling。相位标志是一个可见条件；ED/ES共享网络但不混成同一个plane序列。

### 4.2 距离插值

真实相邻层a、b，s_a<s_b；查询轴向位置s：

\[
\alpha(s)=\frac{s-s_a}{s_b-s_a},\qquad
z(s)=(1-\alpha)z_a+\alpha z_b.
\]

即左latent的权重为“到右端的距离/总距离”，右latent反之。不要把左latent乘以到左端的距离。D就是这两个端点的总间距，而不是混用其它邻居的间距。

真实中间层i做留一层监督：

\[
\alpha_i=\frac{s_i-s_{i-1}}{s_{i+1}-s_{i-1}},\qquad
z'_i=(1-\alpha_i)z_{i-1}+\alpha_i z_{i+1}.
\]

z'_i构造时**不使用z_i**；但真层i的2D标签用于监督其解码结果。这是用户修订后的loss1，不再要求在latent上硬做 `z'_i=z_i`。

没有GT的m只使用相邻真实观测层i、i+1构造z_m。禁止让预测的m再作为下一步插值端点，禁止递归生成伪观测。N=2时可做间隙查询、分布蒸馏、Eikonal和direct监督，但没有留一层项；N=1本核心拒绝，因为不能定义层间插值。

### 4.3 Scalar decoder

输入 `[q_xyz,z(s),phase]` 共132维；输出未经过sigmoid/tanh的单个实数F：

```text
132 → 256 → 256
拼接最初的132维输入形成skip
388 → 256 → 256 → 1
隐藏激活 Softplus(beta=10)，末层线性
```

Softplus使坐标导数更平滑；不是仅靠激活就保证真实SDF。所有位置调用同一个decoder，不给每个slab单独decoder。

给定一个病例的观测集合C，完整场是：

\[
F_\theta(q\mid C)=D_\theta(q,z(n^Tq),phase).
\]

这是在观测轴向范围内定义的连续三维隐式场。z在真实层处连续但其轴向导数可跳变，因此场默认C0连续，不能声称自动C1；接缝项用于减小这些导数突变。

### 4.4 分布投影头与EMA

投影头只服务表征损失：`128→256→64`、GELU、64维L2 normalize、归一化prototype矩阵`64→256`，生成256维logits。重建仍使用128维z，而不是256维概率向量。

在线模型有encoder、decoder、head，全部由optimizer更新。教师由在线模型完整deepcopy初始化，参数`requires_grad=False`、`eval()`；训练时只调用教师encoder/head生成目标。复制decoder是为了保存一致状态，不代表教师decoder接受SDF梯度。

每次成功优化：`teacher = m*teacher + (1-m)*student`。loss NaN或梯度非finite时停止并保留先前有效checkpoint，禁止`nan_to_num`后继续EMA，禁止跳过告警并声称训练正常。

## 5. Loss1：真实层与留一层的兼容监督

### 5.1 为什么不能直接全范围拟合F=d2D

举一个斜面：三维曲面 `x-kz=0`，其真实三维SDF为：

\[
F_{3D}=\frac{x-kz}{\sqrt{1+k^2}}.
\]

固定z切片内的二维SDF却是 `d2D=x-kz`。两者零值面相同、符号相同，数值一般不同。原样拟合二维距离会要求三维梯度范数为 `sqrt(1+k²)`，与Eikonal=1冲突。单纯让输入坐标有xyz不会解除冲突。

### 5.2 局部校准

令P=I−nnᵀ为切片切向投影。对光滑横截面，真轮廓附近的一阶近似是：

\[
F(q)\approx \|P\nabla_qF(q)\|_2\,d_{2D}(q).
\]

因此采用一个标量输出，但将二维距离作为平面几何信息使用：

\[
L_{obs}=L_{zero}+0.2L_{sign}+0.2L_{plane}.
\]

- `L_zero = mean(|F(boundary)|)/tau`，boundary来自真实连续轮廓。
- `L_sign = mean(ReLU(margin − sign(d2D)*F))/tau`，排除距轮廓小于0.2 mm的点；margin为 `min(|d2D|,0.5mm)`。
- `L_plane = SmoothL1((F−stopgrad(kappa)*d2D)/tau, 0)`，`kappa=||P grad F||`。仅在0<|d2D|<2 mm、kappa>0.15处启用；Huber beta=0.2。

默认tau=0.02归一化单位，即2 mm。局部校准是**一阶近似和本方案设计选择**，不是论文中的精确全局SDF监督定理。接近切面与曲面相切的情况，kappa接近0，校准项会失效；零值与符号项仍保留。kappa stop-gradient避免模型利用这一步中的目标尺度反向改动；Eikonal负责抑制场尺度退化。训练初期校准mask可能很少，应记录有效比例，不能误以为距离项始终提供强监督。

实现上，真实层direct和LOO在同一平面上使用固定z_i/z'_i计算切向导数即可：沿平面移动q不改变s，也不改变插值latent。**三维Eikonal则必须重算z(nᵀq)，不能复用这个固定latent梯度写法。**

最终：

\[
L_1=0.5\,mean_i L_{obs}(z_i)+mean_{i\in interior}L_{obs}(z'_i).
\]

两个子项按各自有效层数归一化；病例之间平均，避免切片多的患者权重更大。端点只接受direct重建，不能假造它缺失的一侧邻居。

## 6. Loss2：明确配对的距离加权分布自蒸馏

采用DINO的EMA目标、teacher centering、teacher sharpening、student softmax交叉熵机制，但这里的配对是自定义几何配对。因此命名为 **DINO-style distance-weighted self-distillation**，不能宣称它就是原始DINO或已验证的标准DINO变体。

每层独立重采样A/B两组**同一真实轮廓**上的点+2D SDF：student看A，teacher看B。不要用跨病例、跨ED/ES的表征当同一个目标。

\[
p_i^T=softmax((H_T(z_i^T)-c)/T_T),\quad
p^S(z)=softmax(H_S(z)/T_S).
\]

默认T_S=0.1，T_T=0.07，center momentum=0.9。target全部detach。

两类明确配对：

1. **有真实中间层的留一层配对**：student `H_S(z'_i)` 匹配该真层teacher `p_i^T`。teacher有真实层形状输入，student只有上下邻居；它是最清楚的同一目标层配对。
2. **没有真实GT的层间配对**：student `H_S(z_m)` 匹配 `p_m^T=(1−alpha)*p_i^T+alpha*p_{i+1}^T`。这是局部平滑分布先验，不是未知中间层的真实表征；先各自center+softmax，再混合概率。禁止把全数据集表征随机匹配，也不要将softmax前混logits和混概率说成完全一样。

\[
L_2=\tfrac12 CE(p_i^T,p^S(z'_i))+\tfrac12 CE(p_m^T,p^S(z_m)).
\]

只有2层时使用gap项。center在一次optimizer成功更新后使用该batch的真实teacher logits更新一次；本包单进程。改DDP时必须全局汇总logits sum与样本数，否则各卡center不一致。

此项无法凭空创造真实缺失层GT，也不保证防塌缩或一定提升重建；相邻层属于不同几何位置，强行全部表征一致可能擦除变化。默认它的总权重仅0.05，监测teacher entropy、latent_std、病例间shape差异。表征分布接近不等于重建准确。若研究其收益，后续才做去掉Loss2对照，当前交付以主方法部署为准。

## 7. Loss3：由角度意图转为弱二阶几何正则

### 7.1 放弃补角不等式

如果theta与theta'互为补角，`theta>theta'`仅表示某种角度取法下theta>90°；换另一边/延长线取法可改变判断，它不能不变地表达光滑性。真实心脏也允许较大斜率，甚至在端部出现接近水平的切向。**不写theta大小阈值，不将所有中间零轮廓强制放在任意匹配的上下轮廓直线上。**

要惩罚的是突然折弯：沿零值面走，相邻位置法向不要突变。它限制二阶变化，而不是直接禁止一阶斜率。线性过渡允许任意固定斜率；高曲率被弱惩罚但仍可由真实观测保留。

### 7.2 推导与实现

对F=0附近，定义单位法向N=gradF/||gradF||，表面切向投影P_S=I−NNᵀ。将stack方向投影到表面切向：

\[
t=P_Sn=n-(N^Tn)N.
\]

沿t移动时，法向导数：

\[
\partial_tN=\frac{(I-NN^T)H_Ft}{\|\nabla F\|}.
\]

H_F是坐标Hessian，所以它是二阶几何量。若将t单位化，它对应该方向的法曲率/法向变化；本实现不单位化t，以避免曲面端部 `||t||≈0` 的除零放大，也使接近端部时该方向正则自然变弱。其代价是约束强度带有 `||t||` 因子，应准确描述。

采用中心有限差分，避免显式构造每点3×3 Hessian：

\[
K_t(q)\approx\frac{N(q+\epsilon t)-N(q-\epsilon t)}{2\epsilon}.
\]

\[
L_{geo}=mean_q\left[stopgrad\left(e^{-(F(q)/b)^2}\right)
\,\tfrac13\sum_j Huber(K_{t,j},\beta=1)\right].
\]

默认epsilon=0.003，即0.3 mm；b=0.03，即3 mm。N是由**完整场**求导得到；有限差分两个位置也要重新插值latent。t和近零值带权重detach，把它们视为当前步的几何探测位置/权重。反传仍通过两个位置的N进入encoder/decoder。该式依赖场已接近SDF的行为，初期要warmup；它不能代替真实边界监督。

这不是对提取出的mesh做可微Marching Cubes；训练时直接约束连续场。不会为了求法向而将预测点投影到不存在的真中间轮廓。

### 7.3 接缝法向项

分段线性z只保证C0；在内部真实层轮廓附近，用单侧探针 `q−epsilon*n`、`q+epsilon*n`：

\[
L_{seam}=mean(1-N_-^TN_+).
\]

在严格内部真实层计算，epsilon小于相邻gap的十分之一，端点不计算。它减少接缝法向突变，并不保证数学上严格C1，也不保证零值面拓扑正确。

最终Loss3用两个弱项：默认 `0.002 L_geo + 0.05 L_seam`。Huber属于稳健二阶变化惩罚，不是所谓“二阶L1距离loss”；与Eikonal的一阶梯度范数约束区分开。

## 8. Loss5：真正的三维Eikonal

\[
L_5=E_q(\|\nabla_qF_\theta(q\mid C)\|_2-1)^2.
\]

在无GT的gap空间采样query；不是只在2D真实层上求 `(dF/dx,dF/dy)`。共同平面情况下：

\[
\nabla_qF=\partial_qD+\left(\partial_zD\right)
\frac{z_{i+1}-z_i}{s_{i+1}-s_i}\,n.
\]

如果在生成query之前先固定z_m，或对alpha、latent插值结果detach，那么自动微分漏掉第二项。即使返回3个梯度分量，也不是该病例完整场的三维梯度。

正确代码路径见 `eikonal_loss` 和 `PlaneField.field`：先让新的qr需要坐标梯度，再在field函数内部用qr计算轴向位置、间隔和alpha，然后解码。`searchsorted`只决定分段索引；在每个开区间中alpha必须保留梯度。数据位置s固定，encoder输入观测点固定，不随query移动；encoder参数仍通过z影响Eikonal并得到梯度。

`verify_core.py` 检查F=x+z(s)、z(s)=2s时应有grad=(1,0,2)，且与有限差分一致；错误detach会得到(1,0,0)。这项验收不可删除。

Eikonal鼓励F成为距离场，但稀疏边界问题仍不适定：它不唯一确定缺失层解剖，不保证全局精确距离、闭合、正确拓扑或正常体积。表述应为“3D SDF-like implicit field，经Eikonal正则”，效果通过真实评估确认。

## 9. 无GT间隙采样的隔离规则

在真实相邻端点之间随机选alpha，确保两端各留至少1.5epsilon的间隔。用上下有序轮廓相同索引的线性位置作为粗proposal，叠加6 mm平面内高斯噪声；25%位置替换为更宽的平面bbox采样。

此粗proposal仅为了覆盖可能的表面附近；上、下相同索引不被宣称是准确的解剖对应。不保存它作为pseudo contour，不计算其SDF标签，不进入真实层encoder。gap代码返回 `(q,k,alpha)`，没有sdf字段。m只接受Loss2、Loss3和Loss5；Loss1只在真实层/真实留一层监督。

如gap<=4epsilon，不能安全使用中心差分探针，当前采样器跳过该gap；全部gap都过小时会报错。应调小epsilon并同步记录，而不是在真层外查场或默默clamp到端点。

## 10. 训练设置与梯度更新顺序

主配置见 `config.json`。200 epochs、AdamW lr=2e-4、weight_decay=1e-5、每batch4个case-phase、clip_grad=5、float32；每步每层encoder抽512点，监督抽256点、边界抽128点，gap正则128点。这些是可运行起点，未经过MnM2调参验证。

前5 epochs只优化Loss1；随后10 epochs线性加入Loss2/3/5。EMA照常更新；teacher head在warmup期间尚未受分布loss训练，启用蒸馏后需监测其熵。总loss：

\[
L=0.5L_{self}+L_{LOO}+r(e)[0.05L_2+0.002L_{geo}+0.05L_{seam}+0.1L_5].
\]

顺序严格如下：

1. 学生对真实层A视图编码；教师no_grad对真实层B视图编码。
2. 学生direct与LOO预测接受真实轮廓/2D几何监督。
3. 仅从真实层插值构造gap latent及teacher概率目标。
4. 查询完整field，计算三维梯度及几何项，合并loss。
5. `zero_grad → loss.backward → clip_grad(error_if_nonfinite=True) → optimizer.step`。
6. 该batch真实teacher logits更新center；学生新参数EMA更新教师。
7. 保存日志与checkpoint；失败时不更新teacher/center。

当前参考训练器是单进程；手动按case循环并累积梯度以支持可变N，病例权重均等。不要直接对多层、多病例flatten后全mean。代码不启用AMP；若agent要加AMP，坐标导数、法向及Eikonal至少维持float32并重新跑全部梯度测试。

checkpoint包含student/teacher/distill center/optimizer/epoch/global_step/RNG/config。恢复必须使用完全相同的配置，本入口不支持随意改配置后声称精确resume。训练末保存last和best，best仅依验证集。

验证采用固定真实采样点、不随机生成标签；参考selection score为：

`val_boundary_field_mm + val_loo_boundary_field_mm + 10*val_eikonal + 10*(1-val_sign_balanced_accuracy)`。

后两项用于避免把F≈0的低边界loss误认为好模型。这里field_mm在Eikonal未收敛时只是场值尺度的误差代理，不是已验证的surface distance。真实项目应接入零值轮廓距离、mask Dice等几何验证用于选模；不能将参考代理直接汇报为MCD。

## 11. 从连续场到mesh

`extract_mesh.py` 加载监督在线student，使用全部输入真实层轮廓的采样编码（部署后可固定随机种子和点集），在共同stack基底(u,v,n)形成规则三维网格，query转换到归一化3D坐标；逐块查询F。体数据轴顺序为[u,v,s]，`meshgrid/indexing=ij`；Marching Cubes在0处提取，再加入grid origin、旋转回对齐世界坐标、乘L加c。

默认grid分辨率160³，chunk=32768。先128³排错，最终可256³复查；分辨率和spacing必须写入结果，不能将更细网格当成模型本身性能提升。代码记录零值跨越、finite、开边界edge数、非流形edge数、domain边界接触情况，发现没有正负值跨越则报错，不返回假mesh。

**轴向范围默认只覆盖最下到最上观测层，不外推未知心尖/基底。** 若端点仍有非零面积轮廓，默认结果可以是开放mesh。此时不能从开放mesh直接算可信EDV/ESV/EF。

`--cap-stack` 是显式的人工端面截断：几何上取 `max(F,s_min-s,s-s_max)` 的零面，网格向两端扩一格。这只是和观测slab相交形成的cap，不是恢复真实心尖/基底；会在`.report.json`写 `artificial_axial_caps=true`。有开边界并不自动启用cap。lateral bbox也应检查是否足够覆盖形状，禁止只凭一个最大连通块操作隐藏严重错误。

完整心脏体积需另外验证数据本身是否包含完整端部，或明确新增端部观测/假设；本包不把未观测解剖当成已解决的问题。既有mesh评估若依赖闭合壳层，也必须对这一限制作相同处理。

## 12. 运行命令

在本目录运行；Python建议3.10+。PyTorch CPU/CUDA wheel应根据部署机器从官方安装，以下requirements不是严格锁文件。本包在Python3.12的CPU上验证：首轮PyTorch2.5.1+cpu，最终复核PyTorch2.14.1+cu130（仍使用CPU）。最终版本号见 `verification.json`，这不代表完成CUDA训练验证。

```bash
python -m pip install -r requirements.txt
python verify_core.py --output verification_local.json
python make_synthetic.py --output synthetic
python train.py --data synthetic --output runs/synthetic --device cpu --epochs 1 --max-steps 1
```

最后一条只跑默认warmup smoke，不检验全部loss开启后的路径；`verify_core.py` 已单独验证全部项。要验证真实训练入口全部项，将另存的smoke config设置 `warmup_epochs=0,ramp_epochs=1`，减少采样量和网络宽度，运行2 epochs，随后恢复主config，不能把smoke超参数带入主实验。

真实缓存接入：

```bash
# TODO(AGENT-CACHE): 先使用现有缓存生成 aligned_sax_manifest.json
python prepare_cache.py --manifest aligned_sax_manifest.json --output data/plane_sdf
python train.py --config config.json --data data/plane_sdf --output runs/main --device cuda
python extract_mesh.py --checkpoint runs/main/best.pt --case data/plane_sdf/test/patient_001_ED.npz --output outputs/patient_001_ED.obj --resolution 160 --device cuda
```

若确实采用共同平面投影：

```bash
python prepare_cache.py --manifest aligned_sax_manifest.json --output data/plane_sdf_projected --project-parallel
```

需要人工端面时才显式运行：

```bash
python extract_mesh.py --checkpoint runs/main/best.pt --case data/plane_sdf/test/patient_001_ED.npz --output outputs/patient_001_ED_capped.obj --cap-stack
```

## 13. 点对点修改指令与禁止的误实现

| 核心位置 | agent应做 | 绝对不要做 |
|---|---|---|
| 现有data loader | 从已对齐SAX缓存映射manifest/NPZ | 另跑未经验证对齐；混入LAX或3D GT |
| `prepare_cache.py` | 检查毫米、基底、SDF符号、投影误差 | 复制2D标签到slab厚度里冒充3D距离 |
| `PlaneEncoder` | 独立编码每层点+2D SDF并pool | 命名PointNet；引入全心latent替代plane latent |
| `interpolate_latent` | 使用实际s和反距离权重 | 混用slice index；detach alpha；预测层作锚点 |
| `leave_one_out_latent` | 仅从i−1/i+1构造z'_i | 拼回z_i；只做latent MSE替代解码监督 |
| `SDFDecoder` | 同一个query+latent decoder、单线性输出 | 末层sigmoid；将点编码器误叫decoder |
| `observed_loss` | 零值+符号+局部校准的二维几何 | 全域硬做 `abs(F-d2D)` 同时强Eikonal |
| `DistanceWeightedDistillation` | 真实层配对及gap概率混合 | 随机病例匹配；教师反传；声称完整标准DINO |
| `eikonal_loss` | 在qr建立梯度后调用完整field | 预先固定z_m；只求xy梯度；`no_grad`包住field |
| `gap_geometry_loss` | 弱二阶法向变化、单位一致 | 补角阈值；强制轮廓在线段上；无效除零 |
| `seam_normal_loss` | 内部真实层两侧单向探针 | 声称线性latent插值自动C1 |
| optimizer/EMA | 成功更新学生后EMA | 单独优化教师；非finite仍更新EMA |
| `extract_mesh.py` | 0等值面、正确spacing/基底/毫米反变换 | threshold=0.5；mesh坐标乘错轴；静默cap |
| 评估 | 真实mask/mesh只用于固定验证/测试 | 用GT中间层给训练伪标签；test选超参 |

## 14. 真实部署诊断与验收

本包验证覆盖：非等间距权重、LOO不使用当前层、查询轴向越界拒绝、条件场C0连续、连续多边形距离符号、二维/三维距离斜面例子、完整Eikonal与有限差分一致、EMA方向及teacher冻结、gap仅在真实端点之间、梯度到encoder/decoder/head、Eikonal独立梯度到encoder、二阶几何项、Marching Cubes轴/spacing/winding/毫米反变换。

真实部署仍必须交付：

- 缓存适配器实际修改文件/字段对应、是否进行了共同平面近似及其误差。
- 完整train命令、环境版本、每个phase文件数与患者split计数。
- 真实层direct及LOO的零轮廓叠图，观测层内外符号正确率，gap gradient norm分布。
- 有效局部距离监督比例、teacher entropy、latent_std、非finite次数及对应病例。
- 至少数个不同体积/病理病例的ED/ES mesh，病例间体积与形状差异，排查“只是等比放大”。
- 输出mesh是否开放、是否人工cap、是否截断；完整范围未验证时不汇报可信EF。
- 对可用完整GT的固定test集计算Dice、MCD、HD95；有可信封闭体积时再报EDV/ESV MAE和EF误差。指标实现沿用既有项目并核对毫米单位。

诊断要点：

| 现象 | 首先检查 |
|---|---|
| 边界loss极小但F几乎全0 | sign balanced accuracy、Eikonal、field正负范围；不能只以边界项选模 |
| 每个病例几乎同一形状/只变尺度 | 输入是否确实有case-specific轮廓及d2D；是否抹去物理尺寸；latent/query交换诊断 |
| Eikonal不错但切面完全错 | 真边界/符号监督、两种单位、是否读取错误anatomy |
| 真层好但中间层有尖折 | LOO、完整链式导数、seam项；不是直接增大DINO权重 |
| 加蒸馏后变圆/层间差异消失 | 熵/latent变化、配对和权重；邻层分布先验过强 |
| normal或grad非finite | 小gap除法、平面重复、normalize eps、低精度、坐标异常 |
| OBJ翻转/位置错 | MC轴顺序、basis、grid origin、scale/center及缓存逆对齐 |
| mesh开放或体积异常 | 观测端部缺失、domain裁切、是否误把人工cap当解剖 |

## 15. 源依据与设计归属

- DINO官方论文：Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021：https://arxiv.org/abs/2104.14294。
- DINO官方实现，teacher EMA、center/temperature及交叉熵机制：https://github.com/facebookresearch/dino/blob/main/main_dino.py。
- IGR论文：Gropp et al., *Implicit Geometric Regularization for Learning Shapes*, ICML 2020：https://arxiv.org/abs/2002.10099；官方实现：https://github.com/amosgropp/IGR。

原始DINO的视图设置、头和训练recipe并没有被原样移植。本包的geometry配对、概率距离混合、局部二维距离校准、stack切向法向差分、各loss组合及超参数为本方案具体设计，不应借引用声称它们是上述工作的标准做法，也不声称文献检索证明了新颖性。
