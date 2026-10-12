# PlaneAutoOccupancy 新采样、BCE+ROI Dice 与残差 Decoder：完整部署及五组实验

## 1. 交付范围和冻结定义

搭配代码包：`PlaneAutoOcc_Integrated_Code.zip`。这是对现有仓库的代码 overlay，包含新采样、loss、Decoder、单例批量入口、评分入口与接口测试。**不包含 example 数据、训练缓存、点池或训练 checkpoint。** SSH 端直接复用你之前实验的 occupancy-schema NPZ；不会重新处理原始 MRI 或重新对齐。

参考仓库：<https://github.com/hwu918945-alt/PlaneLatentSDF>。

本次核对的基准 commit：`9367dadef3c4e9a10c904590a01a24b39378dee4`。安装脚本按文件 SHA256 验证替换对象，保留原文件备份；已有改动不会被无条件覆盖。

附件采样核心 `paired_geometry.py` 直接来自本次提供的 `PlaneAutoOcc_Sampling_008_ED_ES(2).zip` 中的 `sampling_trial/paired_sampling.py`，其原始字节 SHA256 为 `e5567b511cd6a32de835cb0428bc70aff7e2ee8d030d66502335b527bc1ac315`。**几何策略没有再次修改**；新 `paired_pools.py` 负责物理坐标、缓存与原训练接口的接入。

## 2. 五个实验标签，及 D/FULL 的准确含义

| 标签 | 采样 | Shape loss | 最后隐藏激活 | 特征残差 |
|---|---|---|---|---|
| A | 新成对多尺度 + ROI 均匀 | BCE_all | 原 Softplus | 无 |
| B | 同 A | BCE_all + 1×Dice_ROI | 原 Softplus | 无 |
| C | 同 A | 同 B | LeakyReLU(0.01) | 无 |
| D | 同 A | 同 B | LeakyReLU(0.01) | h2→h4，相加后激活 |
| FULL | 同 A | 同 B | LeakyReLU(0.01) | 同 D |

**按你给出的 D 公式，D 已是完全版。** 因此本包保留五个运行入口，FULL 是 D 的独立从零复跑，配置和 seed 相同；不能把 D→FULL 的差异解释为第五个方法改动。没有擅自把 D 改成 Softplus 残差或引入其他模块。CPU 全预算冒烟测试中，D/FULL 的最终模型和 latent 逐位相同；GPU 上仍需考虑常规数值非确定性。

比较关系：原已完成的“全 SAX 从头单例”结果→A 检查采样；A→B 检查 ROI Dice；B→C 检查最后激活；C→D 检查残差。D/FULL 检查复跑一致性。若后续要第五个不同的模型配置，应另行明确干预因素；本轮按当前定义执行。

## 3. 对象和原实验协议

已从原“最差10位心脏患者三组汇总”核对患者顺序：

`008, 055, 159, 206, 021, 358, 029, 343, 352, 138`。

每位患者分别训练 ED、ES，共 **20 个阶段**。每个实验标签均对这 20 个阶段单独初始化 G、D 和 case code，单独建立 Adam，共 **5×20=100 个阶段模型**。不同阶段/患者/实验组之间不共享训练权重。

原实验协议保留：

- 训练监督包含该阶段全部缓存 SAX，不是只用三层。
- 每阶段 2000 次完整 optimizer 更新；每次覆盖全部观测 SAX。
- 每层 32,768 点；每层池 262,144 点；8 次完整覆盖；每 80 次本阶段更新换代；2000 次共 25 代。
- seed 19；128D case/plane latent；256 宽度 Decoder；原 Generator 不变。
- Adam 学习率 shared/latent 均沿用原配置（报告值 0.001）；整体 clip 5；无 scheduler、早停或动态权重。
- 内部 SAX：Direct/LOO=0.75/0.25；首末 SAX：Direct=1；逐层等权平均；原 data_weight=1.5、latent L2=1e-4、Seam=0。
- 采样及评分的 XY ROI 均固定为该阶段原首/中/末三层 bbox +20 mm。训练/mesh 的轴向范围是当前全部 SAX 的首末范围。
- 正式评分固定最终 step2000、160²、logit≥0；原生 mesh 96³，不平滑、不补盖、不修复。

prepare 从原 20 个单例 fit 的 config/records 读取实际设置并核验一致性。旧 `fit()` 可能使用 `budget_steps=2000` 而 config.steps 仍写着多病例训练总预算；因此本入口明确固定单例 steps=2000，避免误跑 114000 次。`--steps` 只用于你主动指定的冒烟检查。

**此前的 90 张“原隐藏 SAX”在本轮已参与全 SAX 训练。** 可保留该子集做和原报告对照的拟合评分，但不能再称它们为独立 heldout 泛化结果。

## 4. 新采样的最终设置与来源标记

| 来源 source | 含义 | 单次点数 | 池点数 |
|---|---|---:|---:|
| 0 | Epi 心肌侧 | 3072 | 24576 |
| 1 | Epi 背景侧 | 3072 | 24576 |
| 2 | Endo 心肌侧 | 3072 | 24576 |
| 3 | Endo 心腔侧 | 3072 | 24576 |
| 4 | 整个 ROI 面积均匀，类别自然分布 | 20480 | 163840 |
| 合计 |  | 32768 | 262144 |

Epi/Endo 各抽 3072 **对**，每对两个点；按 pair 单位打乱、抽取，两侧不会被拆到不同的更新。ROI 部分不设 50/25/25 类别配额。

轮廓按弧长分层选锚点；名义偏移 0.25/0.75/2 mm，预算比例 50%/30%/20%，连续扰动 ±20%。在同一锚点两侧采用相同偏移，delta=min(requested,0.4×到另一条轮廓最近距离)。两点各自核验真实区域、ROI 内有效性及到任一边界 >0.05 mm。必要时在同一锚点调整距离/方向；失败明确报错，不把薄壁锚点默默丢掉。法向不直接决定最终标签。

核心仍在局部二维物理 mm 下采样。adapter 根据原 `basis`、`s`、`scale_mm` 映射到原 Decoder 使用的归一化三维坐标，并对最终 float32 坐标重新判标签、检查平面误差和唯一性。没有从层间三维 GT 取标签。

缓存包含 q、region、source、scale_id、delta_mm、pair_id。source==4 才是 Dice 的 ROI 均匀来源；“位于 ROI 内”不是来源判据。标签、source 随同一索引抽取，不能打乱后丢失对应关系。

每个缓存目录按数据 SHA、observed、ROI 参考层、seed、容量、尺度、排除带等采样契约区分；旧三类区域池不会被当成新池。A–FULL 共享同一物理采样池，因此相同阶段、相同 visit 的坐标/标签/source 相同。不同尺度在单次抽样中允许随机计数波动，整池尺度配额固定。

## 5. 新 loss 及精确反传

对第 i 张 SAX 的全部样本 Q_i，以及来源为4的 U_i：

\[
L_{BCE,i}^{all}=\operatorname{mean}_{j\in Q_i}\operatorname{BCEWithLogits}(a_j,y_j),
\qquad p_j=\sigma(a_j).
\]

\[
L_{Dice,i}^{ROI}=1-\frac{2\operatorname{mean}_{j\in U_i}(p_jy_j)+\epsilon}
{\operatorname{mean}_{j\in U_i}(p_j)+\operatorname{mean}_{j\in U_i}(y_j)+\epsilon},
\quad \epsilon=10^{-6}.
\]

\[
L_{shape,i}=L_{BCE,i}^{all}+\lambda_D L_{Dice,i}^{ROI}.
\]

A 的 lambda_D=0；B/C/D/FULL 固定 lambda_D=1，作为本轮明确的起始超参数，不做 schedule，也未声称最优。每张 SAX 独立计算，随后层间平均。BCE 吃原始 logits，Dice 吃 sigmoid 概率，不做硬阈值。

内部层的 Direct 和 LOO 都计算完整 shape loss，并用 **同一 q/y/source**：

\[
L_i=0.75L_{shape,i}^{Direct}+0.25L_{shape,i}^{LOO}.
\]

端层仅 Direct；总损失沿用 data_weight×mean_i(L_i)+latent_weight×mean(code²)。没有增加 Eikonal、SDF、Focal、Boundary、点对 rank loss 或新正则。

原 fast_chunked.py 只能累加逐点 BCE，不能直接用来平均 microbatch Dice。新 backend 分两遍：第一遍 no_grad 得到整张 SAX 的 logits，用一个小的 [N] leaf 图计算完整 loss 的 dL/dlogit；第二遍按原 query microbatch 重算 Decoder，施加这些导数；最后一次性把累积 latent 梯度传回 Generator，L2 计一次。caller 再整体 clip、Adam step 一次。

这个方法保留完整层 Dice 的精确梯度，显存不需要保留全部 SAX 的 Decoder 图。代价是 Decoder 前向计算两遍；A 也使用同一 backend，避免不同组的执行算法混杂。没有 AMP、Dropout 或 BatchNorm；以后若加入随机层或改变运行状态，必须重新验证两遍一致性。

BCE_all 包含边界增强分布，不能称纯面积无偏 BCE。ROI Dice 是有效 ROI 域（扣除0.05mm排除带）的均匀采样估计；固定网格最终评分不扣除这个采样带。

## 6. 新 Decoder 的准确位置

\[
x=[q,z,phase],\quad h_1=Softplus(W_1x),\quad h_2=Softplus(W_2h_1),
\]
\[
h_3=Softplus(W_3[h_2,x]),\qquad h_4=LeakyReLU(h_2+W_4h_3),\qquad a=W_5h_4.
\]

W_k 包含 bias。D/FULL 使用以上公式；C 仅把 h4 改为 LeakyReLU(W4h3)；A/B 保留 Softplus(W4h3)。前三个 Softplus(beta=10)、5个 Linear、256宽度、原始输入拼接、head N(0,0.01²)/零bias均保留。h2 与 W4h3 同为256维，无投影层。**输出 head 是线性 logit，不附加 LeakyReLU 或 sigmoid。**

state_dict 的 a.0/a.2/b.0/b.2/b.4 等参数名保留，组间初始权重相同；激活和残差由 checkpoint.config 恢复。评分时不能只加载新权重到原 Softplus Decoder，所以 inference.restore 也已接入 model_from_config。

## 7. 逐文件部署对应

| 文件 | 操作 | 职责与连接点 |
|---|---|---|
| plane_auto_occ/paired_geometry.py | 新增 | 附件中的二维几何核心，保持原策略 |
| plane_auto_occ/paired_pools.py | 新增 | 原 cache→mm→新池→归一化xyz；get/prefetch/close及draw接口；source独立保留 |
| plane_auto_occ/paired_loss.py | 新增 | shape_loss、整层Dice精确分块反传、只读常数场监测 |
| plane_auto_occ/model.py | 替换 | 加可配置最后激活和h2→h4；默认配置仍与原模型一致 |
| plane_auto_occ/population.py | 替换 | 新 backend 分派到新 store/backward；原 fit 的 Adam/clip/visit/checkpoint流程保留 |
| plane_auto_occ/inference.py | 替换 | 按checkpoint.config恢复Decoder，按原三层计算固定XY ROI |
| plane_auto_occ/train.py | 替换 | 拒绝把新配置误送入不支持ROI来源的legacy --area-pool入口；原配置不受影响 |
| plane_auto_occ/paired_evaluate.py | 新增 | 原评分/mesh入口上增加固定网格Precision/Recall/面积比和原隐藏子集拟合统计 |
| plane_auto_occ/paired_suite.py | 新增 | 读旧单例metadata，冻结新运行，预缓存，100任务执行，汇总 |
| plane_auto_occ/paired_tests.py | 新增 | Dice来源、分块梯度、点对池、checkpoint恢复与默认模型测试 |
| scripts/paired_worst10_array.sh | 新增 | 100个独立GPU任务；默认并发4，可按现有配额改sbatch参数 |
| scripts/paired_worst10_precache.sh | 新增 | 优化前CPU离线准备全部新池 |

原 train.py --area-pool、旧AreaSampler/OfflineSampler的三类布局不用于本轮新配置。**使用 paired_suite 入口**，它实际调用修改后的原 population.fit；不把新池伪装成旧schema，也不修改全量 train/val/test 或 solid 实验入口的协议。

## 8. SSH 解包、核对和安装

以下变量按现有路径填写；新运行目录必须为空。文件链接仅交付下载，SSH端需先将ZIP与本MD传过去。

```bash
TASK_REPO=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneAutoOccupancy
TASK_BUNDLE=/absolute/path/PlaneAutoOcc_Integrated_Code
TASK_PREVIOUS=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneAutoOccupancy/runs/heart_test_worst10_full_sax_20261012_v1
TASK_SUITE=/gpfs/work/aac/haoyuwu24/Heart_Reconstruction/PlaneAutoOccupancy/runs/paired_worst10_20261012_v1
TASK_PY=python

unzip PlaneAutoOcc_Integrated_Code.zip -d "$TASK_BUNDLE"
"$TASK_PY" "$TASK_BUNDLE/install_overlay.py" --repo-root "$TASK_REPO" --check-only
"$TASK_PY" "$TASK_BUNDLE/install_overlay.py" --repo-root "$TASK_REPO"
cd "$TASK_REPO"
"$TASK_PY" -m unittest plane_auto_occ.paired_tests -v
```

复用已有SSH环境的官方CUDA PyTorch，不要为这个包重新安装CPU版。仓库requirements已包含Shapely>=2；新采样不需要新增神经网络包。若包未安装，按仓库requirements补齐即可。

若出现 Base mismatch：说明 SSH 对应文件不是本包核对的版本。先保存现有修改；可以建立基准commit的独立worktree，再安装，不要强行覆盖：

```bash
git -C "$TASK_REPO" worktree add /absolute/path/PlaneAutoOccupancy_paired_revision 9367dadef3c4e9a10c904590a01a24b39378dee4
TASK_REPO=/absolute/path/PlaneAutoOccupancy_paired_revision
"$TASK_PY" "$TASK_BUNDLE/install_overlay.py" --repo-root "$TASK_REPO"
```

如果本地Git尚未包含该commit，先按你现有Git更新方式获取。旧实验的数据/checkpoint绝对路径仍可读取，不需要复制旧训练权重到新网络。

## 9. 冻结 100 个新运行

```bash
cd "$TASK_REPO"
"$TASK_PY" -B -m plane_auto_occ.paired_suite prepare \
  --previous-experiment "$TASK_PREVIOUS" \
  --output "$TASK_SUITE"
```

prepare 读取每个 `outputs/single_case/<patient>_<ED|ES>/fit/fit_manifest.json`；若缺失，回退只读取原last.pt中的records/config。**不加载原权重或optimizer用于训练。** 缓存源文件按SHA核对后复制进新suite/data；冻结当前软件到suite/runtime、生成每阶段每组config和100任务plan。

核对 plan.json：20阶段，100任务；all cached SAX observed；2000更新；seed19；ROI参考 [0,n//2,n-1]；D/FULL配置相同。task_index 0–19=A，20–39=B，40–59=C，60–79=D，80–99=FULL；每组按上述患者顺序排列，每患者ED然后ES。

如果旧输出布局不同，用 `--records-json /absolute/path/records.json` 替代 previous-experiment。JSON提供20条真实NPZ的绝对path与原config（可每条record.config，也可顶层config），以及可选source_sha256、roi_reference_observed。程序核验真实case_id/phase及十位患者集合；不从不明目录模糊猜测患者。不要填模板/example文件代替原缓存。

之后运行命令都在 **suite/runtime** 执行，避免误导入其他PYTHONPATH下的旧包。配置/代码冻结后不要直接编辑；若要改参数，重新prepare新suite。

## 10. 先完成 CPU 离线点池，再运行 GPU

```bash
cd "$TASK_SUITE/runtime"
"$TASK_PY" -B -m plane_auto_occ.paired_suite precache \
  --suite "$TASK_SUITE" --workers 3
```

共20阶段×25代=500个新池；五组共用，不是2500个池。无需每个step存一份；同一代在10个8次覆盖周期内重排，80次后换代。新的审计数组使每点约23字节，比原仅xyz/near布局大；磁盘预算约为 `262144×23×所有20阶段的SAX层数之和×25`，另有少量metadata。不要把它与32,768的单次GPU查询预算混淆。

CPU几何任务可提交到你现有可用CPU队列，脚本为runtime外checkout中的 `scripts/paired_worst10_precache.sh`。使用原集群已验证的account/CPU partition/time选项；导出TASK_SUITE、TASK_PY和可选TASK_CACHE_WORKERS。precache成功后才提交GPU数组，或沿用现有afterok依赖。

构建中断可重新运行同一precache：已完成且契约/文件hash正确的代会复用，未发布的partial不会被当成有效池。所有失败保留在PRECACHE_STATUS.json。几何异常不自动放宽排除带、不换ROI、不丢病例。

## 11. 执行 A/B/C/D/FULL

串行方式：

```bash
cd "$TASK_SUITE/runtime"
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --device cuda
```

它按100任务顺序执行。也可以逐组执行：

```bash
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --arm A --device cuda
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --arm B --device cuda
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --arm C --device cuda
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --arm D --device cuda
"$TASK_PY" -B -m plane_auto_occ.paired_suite run --suite "$TASK_SUITE" --arm FULL --device cuda
```

Slurm数组方式（沿用原可用GPU队列参数，默认100任务、并发4）：

```bash
mkdir -p "$TASK_SUITE/logs"
sbatch --account=xiaowusun_comp \
  --export=ALL,TASK_SUITE="$TASK_SUITE",TASK_PY="$TASK_PY" \
  --output="$TASK_SUITE/logs/paired_%A_%a.out" \
  --error="$TASK_SUITE/logs/paired_%A_%a.err" \
  "$TASK_REPO/scripts/paired_worst10_array.sh"
```

首轮默认不resume。每个任务重新初始化独立网络/latent/Adam；FULL不从D加载参数。若这次新运行被中断，可以明确对对应task加 `--resume` 恢复它自己的last.pt；这是继续本次从0启动的训练，不是继承旧实验或其他组模型。原fit_contract会检查配置、源数据及观测范围。

```bash
"$TASK_PY" -B -m plane_auto_occ.paired_suite run \
  --suite "$TASK_SUITE" --task-index 60 --device cuda --resume
```

如果训练已完成而评分被中断，重新评分前移走该阶段不完整evaluation目录；fit可显式resume，避免旧surface文件被误当成本次评分结果。默认入口会拒绝非空旧fit目录，不会悄悄覆盖或跳过实验。

## 12. 常数场监测与输出

history.jsonl逐层、分别记录Direct和LOO：ROI概率均值/std、logit std、预测/GT正类比例、ROI hard Dice、Precision/Recall、面积比、empty/full prediction、constant_candidate。同时保留BCE和ROI Dice分项、全局梯度norm、pool generation和source计数。

constant_candidate 仅按std<1e-3提示近常数；初始小head下也可能为true。结合训练过程、空/满预测、最终固定网格及零值面一起判断。**不触发重启、早停、loss调整、学习率修改或结果剔除。** 非有限loss/gradient仍按原规则报错并保存失败记录。

每任务输出：

```text
outputs/<A|B|C|D|FULL>/<patient>_<ED|ES>/
  fit/fit_manifest.json
  fit/effective_config.json
  fit/history.jsonl
  fit/last.pt
  fit/TRAIN_COMPLETE.json
  evaluation/metrics.json
  evaluation/plane_*.png
  evaluation/surface.obj        # 仅有原生零面时存在
  RUN_COMPLETE.json            # 或RUN_FAILED.json
```

NO_ZERO_CROSSING 阶段保留metrics与评分，无OBJ是明确状态，不是漏文件。失败不能删除后只比较“成功病例”。原checkpoint/Adam/缓存/原结果不修改。

## 13. 评分与汇总

```bash
cd "$TASK_SUITE/runtime"
"$TASK_PY" -B -m plane_auto_occ.paired_suite summarize --suite "$TASK_SUITE"
```

生成 SUMMARY.json / SUMMARY.md，逐组列20阶段完成/失败/pending、十位患者的ED/ES等权平均、患者等权All-SAX Dice、原隐藏子集拟合Dice、有零面阶段数、空预测平面数。全部20阶段都有结果才给主要患者均值；不完整时主均值为null，失败/缺失名单保留，不生成隐蔽的成功子集主成绩。空预测但正常完成评分的阶段照常计入低分。

主要看：A→B 是否降低面积膨胀并改善整层Dice；B→C 是否减轻常数场/无零面；C→D 是否进一步改善稳定性和拟合。固定最终step2000，不在这些事后选出的最差test患者上调threshold、挑最佳历史checkpoint或宣称独立泛化。

## 14. 交付前验证及边界

包内 VALIDATION_UNIT.json：六项测试通过。A–D 的整层一次反传与新分块两遍反传，在float64下loss一致到11位小数，所有参数/code梯度通过rtol=1e-9、atol=1e-11比较；还检查ROI来源过滤、8次点池覆盖和点对完整性、Decoder恢复、默认原模型前向、D/FULL配置相同。

VALIDATION_FULL_BUDGET.json：使用上传008_ED提取的9张真实SAX构建测试cache，宽256、latent128、每层32768点、完整262144池；A/B/C/D/FULL各执行2次真实CPU Adam更新并恢复checkpoint，loss/gradients有限，source配额正确；D/FULL的模型与code逐位相同。另用生成的接口fixture准备100任务、20阶段池，完成冻结runtime CLI的训练→评分链路。**所有测试数据仅在验证环境生成，不包含在交付ZIP。**

这些验证证明接口和数学计算可跑，不是SSH端20阶段×2000次正式实验，也不能证明新方案已解决几何膨胀或塌缩。正式结果由本说明定义的100个新任务产生。
