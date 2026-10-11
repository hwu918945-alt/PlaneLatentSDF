# Plane Auto-Decoder + Binary Myocardium Occupancy：设计与部署

版本：2026-10-11。交付范围：新增完整实验代码、配置、缓存转换、测试、部署说明和 Git 增量提交。真实 M&Ms-2 缓存不在本包中，需要 SSH 端提供。

## 1. 已确定的研究问题

保留“真实 SAX 的位置表征 → 相邻表征按物理距离插值 → 三维查询 → LOO 监督层间预测”的主线。改动是：表征由 Auto-Decoder 的病例变量生成，输出由 SDF 变成心肌二分类 logit。

正式模型跨病例共享 `generator` 和 `decoder`，每病例每相位有独立的 128D 可优化向量。先在一个病例 ED 上联合优化共享网络及一个向量，检查表示是否能拟合心肌环；该实验本身不构成跨病例形状先验学习的证据。

| 部分 | 新实现 | 对应文件 |
|---|---|---|
| Case latent | 可训练变量 `codes[c,p]`，128D | `train.py` |
| Plane latent | 位置条件 MLP；不读取 Mask | `model.py::AutoOccupancy.plane_latents` |
| Query latent | 相邻观测平面按物理距离线性插值 | `model.py::interpolate` |
| LOO latent | 隐藏内部平面自己的 latent，以两侧插值代替 | `model.py::loo` |
| 占据分类 | 一个连续 logit；心肌为正类 | `model.py::OccupancyDecoder` |
| 观测损失 | Direct BCE + LOO BCE | `objective.py` |
| 额外约束 | Case latent L2；可选的轴向 Seam | `objective.py` |
| 最终几何 | logit=0，即 sigmoid(logit)=0.5 | `extract_mesh.py` |

不再使用 Encoder、两套 Epi/Endo Decoder、DINO/EMA、Zero、Sign、Distance、Eikonal、Orientation 或 Pair。这里只输出一个心肌 Occupancy 场，心腔和心外背景均为负类；用于分层采样的 `region` 不是三分类输出。

## 2. 网络和张量逐点定义

### 2.1 每病例每相位变量

`z[c,p] ∈ R^128`，初始化 `Normal(0,0.01)`。ED 与 ES 独立优化，网络共享，Decoder 同时接收相位 `p ∈ {0,1}`。当前不引入 ED/ES 耦合损失。

### 2.2 位置条件 Plane latent

`u_i = G_phi(concat(z[c,p], s_i, p)) ∈ R^128`。

`G_phi` 为 `Linear(130,128) → SiLU → Linear(128,128) → SiLU → Linear(128,128)`。这里 `s_i` 是已有缓存归一化空间里的轴向位置：`s_i = q_i · axis`。输入无需再乘毫米尺度，否则会把坐标和位置的尺度混用。

这是 latent generator，不是 Mask Encoder。没有独立可学习的每平面残差或 embedding；新病例只优化一个 128D 变量，不能额外优化每张切片的表示。

### 2.3 保留原 Decoder 深度、宽度、激活和输入连接

以提供的 `PlaneLatentSDF(1).zip/core/model.py::SDFDecoder` 为结构依据。默认输入 132D：`[q(3),u(128),p(1)]`。

1. `132 → 256 → Softplus(beta=10)`。
2. `256 → 256 → Softplus(beta=10)`。
3. 将第二层输出与原始 132D 输入拼接，`388 → 256 → Softplus(beta=10)`。
4. `256 → 256 → Softplus(beta=10)`。
5. `256 → 1`，输出 logit。

总计 5 个 Linear、4 个 Softplus。末层保持源实现 `weight Normal(0,0.01), bias=0`。类名改为 `OccupancyDecoder`，避免把非距离输出称为 SDF。训练入口不会先调用 sigmoid，直接使用 PyTorch `binary_cross_entropy_with_logits`；sigmoid 仅用于解释概率分数。

所有 MLP、插值和聚合按实际算子命名，不称 PointNet、Transformer 或 attention。

### 2.4 插值必须使用物理间距

对 `s_i ≤ h=q·axis ≤ s_(i+1)`：

`t=(h-s_i)/(s_(i+1)-s_i)`，`u(q)=(1-t)u_i+t u_(i+1)`。

例：位置 `0,10,30 mm`，在 `20 mm` 使用两侧各 0.5；在 `10 mm` 的 LOO 使用 `2/3*u_0 + 1/3*u_30`。不能按切片序号求均值。

真实平面 Direct 用其 `u_i`。LOO 对内部观测平面 `i` 使用：

`t_i=(s_i-s_(i-1))/(s_(i+1)-s_(i-1))`，`u_loo_i=(1-t_i)u_(i-1)+t_i*u_(i+1)`。

未观测平面查询时，只在 **被选择为观测的平面集合** 中搜索相邻层。隐藏平面不能悄悄成为插值节点；至少两层才能定义插值，至少三层才能训练 LOO。任何超出观测轴向范围的 query 报错，禁止静默外推。

## 3. Loss 替换与尺度

### 3.1 Binary 标签

真实 SAX 的心肌内部 `y=1`，心腔与心外背景 `y=0`。从现有心肌 Shell `sdf2d<0` 转为标签，只使用符号，不回归距离幅度。若输入只有 Epi 的 filled-region SDF，不能用来训练心肌 Occupancy；转换器要求 `boundary_endo` 并核验符号。

每个真实平面先平均 BCE，再在平面间平均，避免点数和平面数改变总损失尺度。LOO 对内部真实平面的 BCE 独立求均值。

`L = 0.5*L_direct + 1.0*L_loo + 1e-4*mean(z[c,p]^2)`。

若配置 `seam_weight>0`，再加 `seam_weight*L_seam`，到 `seam_start` 后启用。首版 `seam_weight=0`，用 Direct/LOO/L2 先验证分类几何，不承担任何单位距离梯度约束。

均衡抽样后的 sigmoid 是分类分数，不能未经校准就声称是原始空间分布下的真实后验概率。默认阈值固定为 0.5；不要按测试集结果挑阈值。

### 3.2 可选 Occupancy Seam

在内部真实 SAX 的 Epi/Endo 轮廓取 `q`，沿 unit axis 取 `q±eps*axis`。三次查询都经完整的插值场，不把两个偏移点都送入固定 `u_i`。

`delta=[ell(q+eps*a)-2*ell(q)+ell(q-eps*a)] / eps_mm`。

这等于左右轴向一阶斜率差，单位为 logit/mm。它不是除以 `eps²` 的二阶导数，也不是 Eikonal。对 `delta` 使用 `SmoothL1(beta=1)`。

`eps_mm` 初值 0.1 mm，实际 `eps` 在归一化空间内取该值除以 `scale_mm`，并限制为左右层间距最小值的 10%。两侧不使用伪 Occupancy GT。Seam 可在环已形成后单独试 `seam_weight=0.001`，这只是未验证初值。

## 4. 缓存与采样接口

转换读取原 Shell NPZ：

| 字段 | 形状/含义 |
|---|---|
| `q` | `[N,P,3]`，已有归一化三维坐标 |
| `sdf2d` | `[N,P]`，心肌负、心腔和背景正 |
| `boundary` | `[N,B,3]`，有序 Epi 轮廓 |
| `boundary_endo` | `[N,B2,3]`，有序 Endo 轮廓 |
| `s` | `[N]`，严格递增的平面位置 |
| `axis` | `[3]`，单位 SAX 轴 |
| `center_mm`, `scale_mm` | `q_mm=q*scale_mm+center_mm` |
| `phase`, `case_id` | ED=0/ES=1、病例身份 |
| `sampling_source` | 可选；原边界 source=0/2 明确排除 |

输出保留 q、轮廓、平面及归一化元数据，新增 `occ`、`region`、`near`、`schema_version`、源文件 SHA256。源 SDF 不再出现在训练接口中。`region=0` 为心肌、1 为心腔、2 为背景，`-1` 表示边界模糊样本，不用于 BCE。

转换时从轮廓填充关系审计心肌区域、区分两个负类。从已有点池中排除距任一边界 ≤0.05 mm 的点及原精确零值点；`near=True` 指距边界 ≤3 mm。二维距离只用于抽样/审计，不作为三维距离 GT。轮廓是已缓存的重采样多边形，因此这是该缓存的几何版本，存在原 raster Mask 与轮廓离散化差异。

每张 SAX 每步抽 1024 点：心肌512、心腔256、背景256；每类尽量一半 near、一半 far。如果某类无 near/far 子池，用该类全池补足并记录 `pool_fallbacks`，不改变标签。若任何语义类全池缺失则报错。可在小点池内有放回抽样，离线候选池容量不等于每步点数。

整个实验不创建层间三维 Occupancy GT，不把 2D Mask 复制为厚 slab。监督点仍在真实 SAX 平面上。

## 5. SSH 端部署与 Git 推送

外层交付 ZIP 含 `source/experiments/plane_auto_occ` 全部代码、`PlaneAutoOccupancy.patch`、`PlaneAutoOccupancy.bundle` 和本说明。ZIP 不含真实数据或真实训练 checkpoint。Git bundle 是此新增实验目录的独立提交历史，**不是现有 GitHub main 的镜像或完整仓库备份**；推荐使用 patch 追加到当前 SSH 仓库。

设 `PKG` 为解压后的交付目录、`REPO` 为 SSH 端已有 PlaneLatentSDF 仓库：

```bash
cd "$REPO"
git status --short
git switch -c experiment/plane-auto-occupancy
git apply --check "$PKG/PlaneAutoOccupancy.patch"
git am "$PKG/PlaneAutoOccupancy.patch"
python -m experiments.plane_auto_occ.tests
git log -1 --oneline
git push -u origin experiment/plane-auto-occupancy
```

应用前保持已有改动已提交或已妥善保存，且 `experiments/plane_auto_occ` 目录尚不存在。这个 commit 只增加该目录，不覆盖现有 core、train.py 或双 SDF 文件。代码实际可以执行，原仓库仍可按自己的旧入口运行。

如果选择 bundle，替代上面的 `git am`：

```bash
git bundle verify "$PKG/PlaneAutoOccupancy.bundle"
git fetch "$PKG/PlaneAutoOccupancy.bundle" experiment/auto-occ
git cherry-pick FETCH_HEAD
```

二者择一，不重复安装。bundle 的独立根提交通过 cherry-pick 追加到当前分支，不能将独立根直接强推覆盖远端 main。我们没有访问你的 SSH 端，也没有在 GitHub 上提交或推送。

运行依赖见 `requirements.txt`。通常复用已能训练 PlaneLatentSDF 的环境；若缺包，在该环境中安装对应包。`torch` 使用 SSH GPU 对应的官方 CUDA wheel，不要为了本实验替换正在使用的 CUDA 环境。本包的本地验证用 CPU。

## 6. 单病例先跑通：实际命令

先在仓库根设置路径。`CASE` 必须指向转换后的 **一个真实病例 ED** 文件。索引是排序后缓存平面的 0-based 索引，不是 DICOM 原编号。

```bash
RAW=/absolute/path/current_shell_endpoint_cache
OCC=/absolute/path/new_occupancy_cache
python -m experiments.plane_auto_occ.convert_cache --input-root "$RAW" --output-root "$OCC"
CASE="$OCC/train/example_ED.npz"
```

示例取 0、3、6 三层。应根据该病例实际平面数量修改，优先保留首末有效层，以使其他真实层处于可评估范围。一个病例若真的只有三层，`0,1,2` 没有严格隐藏 SAX 的 GT，只能检查未观测层的几何外观。

### 6.1 两步 smoke 与正式短程拟合

```bash
python -m experiments.plane_auto_occ.train --case "$CASE" --observed 0,3,6 \
  --output runs/auto_occ_one_ed_smoke --device cuda --steps 2
python -m experiments.plane_auto_occ.train --case "$CASE" --observed 0,3,6 \
  --output runs/auto_occ_one_ed --device cuda --steps 2000
```

两次均从头训练，共享网络与 case code 联合优化。禁止恢复旧 SDF checkpoint。脚本要求新输出目录，避免覆盖旧实验；当前不实现 optimizer resume。若要 Direct-only 对照，复制 config JSON，将 `loo_weight` 设为0，以另一新输出目录启动。

默认配置并非经过临床验证的最佳参数。观察 0/50/100/200/500/1000/2000 step 的独立 Direct、LOO、梯度和几何结果；不要只凭 total loss 决定继续长程训练。早期低梯度或恒定分数不等于训练已成功，也不能仅由 BCE 形式推出保证成功。

### 6.2 真实 SAX、隐藏 SAX 和层间可视化

```bash
python -m experiments.plane_auto_occ.evaluate --checkpoint runs/auto_occ_one_ed/last.pt \
  --case "$CASE" --output runs/auto_occ_one_ed_eval --device cuda
python -m experiments.plane_auto_occ.extract_mesh --checkpoint runs/auto_occ_one_ed/last.pt \
  --case "$CASE" --output runs/auto_occ_one_ed_myo.obj --device cuda --resolution 128
```

`metrics.json` 将结果分为 observed、strict_heldout、unlabelled_gap。LOO 的指标命名为 `training_loo_proxy_dice`；它仍使用训练期被监督过的内部 SAX，不能声称为未知切片泛化。

严格隐藏的平面不在 `fit_manifest.json` 的 observed 列表里，Mask/轮廓不参与训练目标、Seam、边界范围生成或插值节点；它们仅在训练完成后的 evaluate 中使用。既有病例对齐与归一化 metadata 沿用输入缓存，因此“strict”指 **在既定对齐坐标系统中排除隐藏平面的形状监督**，不声称重新从原始 MRI 完成全流程盲测。

评估的 Dice 为 **均匀平面网格上的缓存轮廓填充代理 Dice**，不是原始 raster Mask 的临床 Dice。脚本输出独立 PNG：蓝色预测，橙色真实缓存轮廓。如果 `ground_truth_touches_border=true`，说明评估域可能截断真实心肌，应扩大 `--margin-mm` 后重新评估。对于真正未观测的中间位置没有 Dice GT。当前也不报告 EF、EDV/ESV、MCD/HD95；这些需要后续对接真实体积/曲面评估协议。

Mesh 从三维 grid 的零 logit 面提取，只覆盖观测轴向范围，不添加外侧空背景、端点 cap 或假闭合。若观测首末 SAX 有心肌，网格可能在轴向端点开口；这是 slab 截断，不能通过填帽伪装成完整心脏。报告 boundary edges、nonmanifold edges 和轴向/侧向边界接触。OBJ 顶点还原到 aligned world mm。

## 7. 多病例共享先验与新病例拟合

保持 M&Ms-2 既定患者划分 285/36/36，ED/ES 每患者两文件。不要把相位拆进不同集合。

```bash
python -m experiments.plane_auto_occ.audit_splits --root "$OCC" --expected-patients 285,36,36
python -m experiments.plane_auto_occ.train --data-root "$OCC/train" \
  --output runs/auto_occ_multi --device cuda --steps 114000
```

`steps` 是优化器更新次数：每步处理一个病例相位文件，并无 batch-DDP。570 个文件时 114000 steps 约为200遍历；这里只说明计数，不要求正式实验固定200 epoch。默认2000 steps 是短程诊断预算，不能直接视为正式多病例训练完成。多病例候选点池保留在 CPU，逐例搬到 GPU；该最小实现未提供 DDP 或大规模性能调优。

正式多病例可使用全部观测 SAX，LOO 在其中的内部层训练。若希望训练时每病例只见固定三张 SAX，提供 manifest：

```json
{"records": [{"path": "/absolute/path/occ/train/example_ED.npz", "observed": [0,3,6]}]}
```

```bash
python -m experiments.plane_auto_occ.train --manifest configs/auto_occ_train_manifest.json \
  --output runs/auto_occ_multi_sparse --device cuda --steps 114000
```

路径为绝对路径或相对 manifest 的目录。manifest 必须只列 train 病例，每个 case/phase 最多一次。多病例 checkpoint 保存 shared 网络、code 表、训练身份与每例 observed 集合。

新病例测试：冻结 `G_phi,D_theta`，重新随机初始化新病例 code，只用该病例实际保留的 SAX 优化 code：

```bash
NEWCASE="$OCC/test/new_example_ED.npz"
python -m experiments.plane_auto_occ.train --pretrained runs/auto_occ_multi/last.pt \
  --case "$NEWCASE" --observed 0,3,6 --output runs/auto_occ_new_case --device cuda --steps 1000
python -m experiments.plane_auto_occ.evaluate --checkpoint runs/auto_occ_new_case/last.pt \
  --case "$NEWCASE" --output runs/auto_occ_new_case_eval --device cuda
```

`--pretrained` 只接收本方法的 checkpoint，自动冻结共享参数，并拒绝训练患者重新作为“新病例”测试，包括同一患者另一相位。1000 steps 是 latent 优化预算示例，不是统一 fine-tune 80e：这里没有 fine-tune 共享权重。不要把训练病例自己的 code 作为新病例初始化，也不要将隐藏切片用于选择拟合步数、权重或阈值。val 用于定超参数，test 仅作最终评估。

## 8. 必须观察的实验结论边界

1. 真实 SAX：是否出现一个心肌连通环、内外两个闭合边界；代码提供闭合曲线数量与边界接触指标，仍需看 PNG，两个闭合曲线也可能是两个错误孤岛。
2. LOO：训练内部层的插值输出是否重建正确环，单独看 LOO 指标。
3. 真正隐藏 SAX：Mask 从未参与目标后是否仍重建合理，区别于训练期 LOO。
4. 三维：层间是否有断裂/侧向截断，Mesh 是否存在异常碎片；轴向开口按 slab 范围解释。
5. latent：记录相邻 `||u_i-u_(i+1)||`，并检查 mean/reversed/zero plane-latent 三种查询扰动引起的预测变化。若扰动不改变结果，Decoder 可能主要依赖 q，不能宣称已验证层间表征贡献。

重要限制：`u_i` 是全局 code 和位置的函数，可能趋于相同或沿位置近似线性，使 LOO 容易满足。Decoder 也可能绕过 latent，直接从 q 拟合单病例。没有为了防止这些情况偷偷加入 latent diversity 或对比损失；保持主方法可解释，先记录证据。如果单病例拟合成功但 latent 扰动无效，结论只能是 Occupancy 场能拟合该对象，无法证明核心层间 latent 机制有效。

源依据与本次验证见 `PROVENANCE.md` 和 `VALIDATION.md`。损失 API 使用官方 PyTorch： https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 。
