# PlaneAutoOccupancy V2：Case-only 与 Plane-latent 插值消融实验部署指导

> **交付性质：**面向 Coding Agent 的实施规范与逐项验证清单；本文件本身**不代表代码已经修改、实验已经运行或 GitHub 已经推送**。
>
> **研究目标：**先验证 Plane latent 整套表征是否优于简单的 Case-only Auto-Decoder，再验证在已有 Plane latent 的条件下，**线性插值**是否优于**最近邻选择**。只有这两项获得积极证据，再单独讨论 LOO Loss 的增量贡献。
>
> **基线仓库：** [`hwu918945-alt/PlaneLatentSDF`](https://github.com/hwu918945-alt/PlaneLatentSDF)，分支 `experiment/plane-auto-occupancy-repair`，原始固定提交 `989601aabc0083388ec88230d3ab0eee117e5e68`。
>
> **范围：**先只研究 `002_ED` 单病例、真实观测 SAX 索引 `[0,3,6]`，独立评价 `[1,2,4,5]`；**不扩展多病例、不增添新的架构、Loss 或正则**。所有实验在真实 SAX 轴向范围内重建，不外推、不人工封盖。

---

## 0. 先读结论：本轮只有三个组，A 需要重新训练

| 实验 | 符号 | 模型条件变量 | 无 GT 层间如何取 latent | 是否需重新训练 | 主要问题 |
|---|---|---|---|---|---|
| Case-only Auto-Decoder | **A** | 直接使用每病例 128D `z_case` | 始终用同一个 `z_case`；无 Plane latent | **是，2000 完整 update** | Plane latent 整套设计是否带来增量收益？ |
| Plane latent + 最近邻 | **B** | 当前 V2 的 `u_i=G(z_case,s_i,phase)` | 取轴向距离最近的**已观测** Plane latent | **否，固定 C 的最终 checkpoint** | 线性插值是否优于最近邻选择？ |
| Plane latent + 线性插值 | **C** | 当前 V2 的 `u_i=G(z_case,s_i,phase)` | 按实际 SAX 距离加权相邻 latent | **否，复用已有 V2 完整结果** | 原方法对照 |

核心比较：**C vs A** 和 **C vs B**。**目前不做第四组 LOO 消融**。

**严谨解释：**
- `C vs A` 检验带有 Generator、Plane latent、线性插值及配套 LOO 训练机制的**整体 Plane-latent 方法**相对于 Case-only 的增量；不能把差距单独归因于 Generator。A 的 LOO 在数学上与 Direct 相同，故化简掉且保持相同的数据总权重。
- `C vs B` 是**同一已训练模型的推理算子消融**，能隔离当前 checkpoint 上“线性插值 vs 最近邻选择”的效果；因为 C 的权重曾在插值/LOO 条件下训练，结果不是“独立重训的最近邻网络”的最优能力比较。严禁据此直接宣称线性插值在所有训练协议下都必然优于最近邻。
- 后续只有在有积极增量证据时，才考虑 `C(LOO=0.25)` vs `C(LOO=0)` 的重新训练对照。

## 1. 固定现有方法和实验事实

### 1.1 C（V2）网络和公式

真实观测轴向坐标（原缓存归一化单位，`scale_mm=100`）：

\[
(s_0,s_3,s_6)\approx(0,0.3,0.6),\qquad \mathrm{phase}=0\ (ED).
\]

每例可训练病例码（128D）：

\[
 z_c\in\mathbb R^{128}.
\]

Generator（`130 → 128 → 128 → 128`，两个 SiLU）：

\[
 u_i=G_\phi([z_c,s_i,\mathrm{phase}]),\quad u_i\in\mathbb R^{128}.
\]

三维查询 `q` 沿共同 SAX 轴 `a` 的高度：`h=q·a`。若 `s_i ≤ h ≤ s_{i+1}`：

\[
 \alpha=\frac{h-s_i}{s_{i+1}-s_i},\qquad
 u_C(q)=(1-\alpha)u_i+\alpha u_{i+1}.
\]

共享 Decoder 沿用原 `OccupancyDecoder`：输入 **`[q_xyz, u_C(q), phase]` 共 132D**，经过 `Linear(132,256) → Softplus → Linear(256,256) → Softplus → concat(原132D) → Linear(388,256) → Softplus → Linear(256,256) → Softplus → Linear(256,1)`；总计 5 个 Linear、4 个 `Softplus(beta=10)`，输出**单个 logit**（非 SDF）：

\[
 \ell_C(q)=D_\theta(q,u_C(q),p),\quad P_{myo}(q)=\sigma(\ell_C(q)).
\]

使用 `logit≥0` 作为心肌预测；三维规则网格上使用 `Marching Cubes(level=0)` 提取 mesh。当前 `extract_mesh.py` 默认 `128^3`，无 padding、无封盖。

**注意：** `u_i` 是 Generator 的**输出**，不是每张切片的独立 `nn.Parameter`；真实训练只优化 `z_case`、共享 Generator 和 Decoder 的参数。

### 1.2 V2 的 Loss（本轮应复用，不要发明新损失）

每个真实观测 plane 的区域平衡 BCE：

\[
 D_i=0.5\,\operatorname{BCE}_{myo}
 +0.25\,\operatorname{BCE}_{cavity}
 +0.25\,\operatorname{BCE}_{background}.
\]

内部真实观测 `s_3` 的 LOO 条件：

\[
 \tilde u_3=\frac{s_6-s_3}{s_6-s_0}u_0+\frac{s_3-s_0}{s_6-s_0}u_6,
 \quad O_3=\operatorname{BCE}(D_\theta(q_3,\tilde u_3,p),y_3).
\]

三平面总体：

\[
 \boxed{L_C=0.5D_0+0.375D_3+0.125O_3+0.5D_6
 +10^{-4}\operatorname{mean}(z_c^2)}.
\]

Seam=0；无 Eikonal、SDF Zero/Sign/Distance、DINO、层间伪 GT。

### 1.3 现有正式运行事实，用作 C 的固定参照

来自已完成的 `002_ED` V2 2000-update 报告，不是新实验结果：

| 平面索引 | 类别 | C 统一网格 Dice（`160²`） |
|---|---|---:|
| 0 | 观测 | 0.8695908 |
| 1 | 严格隐藏 | 0.7801921 |
| 2 | 严格隐藏 | 0.7706892 |
| 3 | 观测 | 0.8194783 |
| 4 | 严格隐藏 | 0.8682804 |
| 5 | 严格隐藏 | 0.7880372 |
| 6 | 观测 | 0.7163121 |

观测平均约 0.8018，隐藏平均约 0.8018。7/7 SAX 都是单连通心肌环、两条闭合边界。C 最终 128³ mesh：97,755 顶点 / 194,362 面 / 1148 轴向开放边 / **零侧向开放边**。这只是基线记录，不能预填 A/B 的结果。

### 1.4 锁定协议：不可改变

- 数据：相同 `002_ED` occupancy NPZ，`observed=[0,3,6]`，其余 `[1,2,4,5]` 仅事后评价。**先选择 observed，后建立 ROI 和训练点池**。
- 训练抽样：每张已观测 plane、每完整更新 **32768** 独立三维坐标；myocardium 16384（标签1）、cavity 8192（0）、background 8192（0）。边界 0.05 mm 排除带，不把 `sampling_source` 当真值。
- 离线池：每张 131072/65536/65536；更新内无放回，8 updates 消耗完整一轮，80 updates（10轮）按原方法刷新。池基于**仅观测**轮廓、+20 mm 采样 ROI。
- 完整 update：三层总独立输入 98304，4096/块，仅在全病例累计完成后 **一次** Adam step、一次统一 clip=5；每层 loss 权重一致。
- 训练预算：**2000 次完整 optimizer 更新**，`lr_shared=lr_latent=1e-3`，Adam 默认参数，seed 19，`log_every=50`，`save_every=250`。不加 scheduler、warmup、动态权重、早停或新的 loss。
- 评价：同一 `160×160` 网格；ROI 为相同**仅观测**轮廓 bbox + 5 mm；另外 bbox+20 mm 鲁棒性审计；相同 `logit=0` 阈值；GT 为缓存 Epi/Endo 填充多边形代理，**不是**原始 MRI raster。
- Mesh：`128³`、同一轴向范围、相同 `center_mm/scale_mm/axis`、相同 bbox 与 marching-cubes 调用；不得人为封口、扩大端部、只保留最大连通块或删掉失败部件。
- C 统一使用**已保存的第2000步最终在线模型**，不根据隐藏指标回退选择 checkpoint。A 同样只用第2000步 checkpoint；B 与 C 共享一个 checkpoint，禁止微调。

## 2. 实验 A：最简 Case-only Auto-Decoder（唯一新增训练）

### 2.1 目标与形式

去掉整个 Plane latent Generator 和层间条件插值，不改变 Decoder 内部结构，也不改变 128D latent 的维度。

\[
\boxed{\ell_A(q)=D_{\theta_A}(q,z_c,p)}.
\]

**输入仍然为 132D** `[xyz(3), case_code(128), phase(1)]`。这里不是把所有 Plane latent 各自作为独立变量，也不是把 2D Mask 体素插值成 3D GT。

单个 128D `z_case` 直接通过 `.expand(Q,128)` 广播到一批不同的 3D 查询点。不同高度来自 `q` 自身的 `xyz`，而不是 Plane latent。

**保留原 `OccupancyDecoder` 类而不重新手写网络**，以保证 Linear 数量、通道数、skip、Softplus beta、末层初始化完全一致。C 比 A 多出的 Generator 约有 49,792 个参数，此差异是被删除模块的自然结果，应报告总可训练参数量，**不要为了匹配数量另加一个无意义 MLP**。

### 2.2 推荐的最小模型适配（可直接供 Agent 参考）

新增 `experiments/plane_auto_occ/ablations/model_case_only.py`，不要覆盖 `experiments/plane_auto_occ/model.py`：

```python
import torch
from torch import nn
from experiments.plane_auto_occ.model import OccupancyDecoder

class CaseOnlyAutoOccupancy(nn.Module):
    def __init__(self, latent_dim=128, width=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = OccupancyDecoder(latent_dim, width)

    def plane_latents(self, code, s, phase):
        # Compatibility only: [N,128] copies of the SAME case code.
        # They are NOT independently optimized plane latents.
        if code.numel() != self.latent_dim:
            raise ValueError("Invalid case-code dimension")
        return code.reshape(1, -1).expand(len(s), -1)

    def field(self, q, s, u, axis, phase):
        # No interpolation. The decoder sees xyz + SAME case code + phase.
        return self.decoder(q, u[:1].expand(len(q), -1), phase)
```

本兼容接口允许复用现有 `chunked.backward_case()` 的一部分逻辑，**但 A 的配置必须令 `loo_mix=0`**。此时 `loo()` 路径根本不应执行，避免无效的重复 Decoder 调用。推理时无需 `.field()` 中的 `interpolate()`。

注意 `z_c` 仍由原 `train.py` 同样的 `nn.Parameter(torch.randn(...)*0.01)` 初始化；不得无意改成固定 latent 或添加 Decoder-only 的新训练模式。

### 2.3 A 的 Loss：与 C 同等数据系数，但 LOO 被数学化简

对于 A，因为所有切片使用同一个 `z_c`：

\[
\tilde z_c=z_c\quad\Rightarrow\quad O_i=D_i.
\]

因此 C 形式中的 `0.75D_i + 0.25O_i` 对 A 完全等于 `D_i`，不构成独立 LOO 监督。

A 使用：

\[
\boxed{L_A=0.5D_0+0.5D_3+0.5D_6
+10^{-4}\operatorname{mean}(z_c^2)}.
\]

这与 `data_weight=1.5 × mean(D0,D3,D6)` 等价。**不要**简单地把旧 `0.5*mean(Direct)+1.0*mean(LOO)` 当成 A 的目标，不要令数据总系数由 1.5 意外变为 0.5 或 1.0。

A 配置（以原 `config.json` 复制为 `ablations/config_A_case_only.json`；这里只列出差异，其他所有值必须完全相同）：

```json
{
  "model_variant": "case_only",
  "loo_mix": 0.0,
  "loo_weight": 0.0,
  "data_weight": 1.5,
  "seam_weight": 0.0
}
```

`model_variant` 是新增的实验元数据字段；当前原始 `train.py` **并不支持它**。Coding Agent 应提供一个新的 A 训练入口（见 §5），不要把上面的 JSON 当成可以直接交给原 `train.py` 就会自动生效的命令。

### 2.4 A 的初始化：公平但不借用 C 的已训练权重

优先检查 C 的 `step_00000/checkpoint.pt` 是否存在，若存在：

1. 读取 **C 第0步** checkpoint，只复制相同 `OccupancyDecoder` 的**初始权重**到 A；
2. 同时复制第0步的 **同一 128D 初始 case code**；
3. **禁止复制 C 第2000步权重、优化器动量、训练后的 Plane latent 或 Generator 的任何参数**；
4. A 使用一个全新的 Adam 优化器、同种参数组学习率，独立完成 2000 updates；
5. 把初始化来源及哈希写入 A 的 `fit_manifest.json` 和实验报告。

如果第0步 checkpoint 不可获得：采用原 seed=19、原 Decoder 初始化规则重训 A，但应在报告标明：**由于去掉 Generator 改变随机数消耗顺序，A/C 的 Decoder 初始权重未必逐项相同**。不要声称逐权重配对。

### 2.5 A 的训练实现要求

建议新增 `experiments/plane_auto_occ/ablations/train_case_only.py`：

- 使用现成的 `data.load_case()`、`area_pool.AreaSampler`、`area_pool.build_pool()`、`chunked.backward_case()`，不要重新实现区域采样、分块梯度累积、日志和标签语义。
- 保留原训练脚本的病例选择、source SHA256、observed 检查、池重建、seed、检查点保存、梯度裁剪和错误上报。
- 仅将 `AutoOccupancy(...)` 替换为 `CaseOnlyAutoOccupancy(...)`，并把 `loo_mix/loo_weight` 设为0、`data_weight` 保持1.5。
- 若直接复用 `chunked.backward_case()`，它会调用 `.plane_latents(code,...)`；上面的兼容接口返回相同 case code，符合 A 定义。
- 原始 `train.py` 的 `--area-pool` 分支限制单病例，**继续保留**；不自行扩展到多病例。
- 注意每个 chunk 中只需共享 code；反传正确累积到同一 code 和 Decoder，`latent_weight*mean(code**2)` **只 backward 一次**。
- 检查点必须声明 `model_variant=case_only`，以免在推理时错误实例化 `AutoOccupancy`。**A 的 checkpoint 不得伪装成 C checkpoint。**
- 即使二者的日志字段名称类似，A 的 `loo` 应标记 `not_applicable` 或 0 且附解释，**不能把 Direct=LOO 的数值报告为独立 LOO 成果**。

## 3. 实验 B：同一已训练模型的“最近邻 Plane latent”推理

### 3.1 目标与公式

**B 不重新训练，不重新初始化 case code，不优化任何参数，也不改训练 Loss。**

仍然读取 C 最终 checkpoint，并使用相同的：

\[
 u_i=G_{\phi^*}(z_c^*,s_i,p).
\]

唯一改变：将原本连续线性插值

\[
 u_C(q)=(1-\alpha)u_i+\alpha u_{i+1}
\]

替换成最近邻真实观测 Plane latent：

\[
\boxed{u_B(q)=u_{j^*},\quad j^*=\arg\min_{j\in\{0,3,6\}}|q\cdot a-s_j|.}
\]

所有查询点仍输入相同的 `xyz`、phase 和当前 `u_B(q)`：

\[
\ell_B(q)=D_{\theta^*}(q,u_B(q),p).
\]

最近邻规则在两观测 plane 的中点出现切换；**相等距离时一律选择左侧较低高度**，并写入方法元数据，保持可复现。

### 3.2 最小实现代码（供 Agent 直接复用）

新增 `experiments/plane_auto_occ/ablations/nearest.py`，不修改原 `model.interpolate()`：

```python
import torch

def nearest_plane_latent(q, s, u, axis):
    """q [Q,3], s [N] ascending, u [N,128], unit axis [3]."""
    if len(s) < 2 or not bool(torch.all(s[1:] > s[:-1])):
        raise ValueError("Need >=2 increasing observed heights")
    h = q @ axis
    if bool(((h.detach() < s[0]-1e-5) |
             (h.detach() > s[-1]+1e-5)).any()):
        raise ValueError("No axial extrapolation")
    r = torch.searchsorted(
        s.contiguous(), h.detach().contiguous(), right=False
    ).clamp(0, len(s)-1)
    l = (r - 1).clamp(0, len(s)-1)
    choose_l = (h-s[l]).abs() <= (s[r]-h).abs()  # tie -> left
    index = torch.where(choose_l, l, r)
    return u[index]


def query_nearest(model, q, s, u, axis, phase):
    latent = nearest_plane_latent(q, s, u, axis)
    return model.decoder(q, latent, phase)
```

此处 `q`、`s` 必须统一采用原仓库缓存的归一化坐标。不能用毫米坐标去计算最近邻却保留归一化 Decoder 输入。对于与观测平面高度一致的点（含端点），应严格选中对应 `u_i`。

B 的“中间 Plane”**不是由插值产生新 latent**，而是从最近观测切片直接复制已有 latent，仍由 Decoder 的连续 `xyz` 输入决定空间分类。

### 3.3 B 必须保持什么不变？

- 调用 `inference.restore(C_final_ckpt, case)` 获得**完全相同**的 model、code、`u_i`、obs、full。
- 使用与 C 完全相同的 `inference.bounds()`、`inference.plane_grid()`、`geometry.contour_labels()`、`geometry.contour_report()` 以及 `extract_mesh.mesh_arrays()`。
- 只为 B 建立新的 `query_variant(..., method='nearest')`；原始 C 的 `query()` 保持不变。
- 查询块大小、dtype、CPU/GPU、随机性、phase、阈值、空间 ROI 与输出顺序都需相同。
- 三维 `128³` 采样网格大小及布局不变，`volume[:,:,i]` 的构造和 `marching_cubes(level=0)` 保持原样。
- 绝不允许用隐藏真实 SAX Mask 计算 `u_i`，也不允许因 B 产生裂面就后处理补面。

### 3.4 B 特有的 discontinuity 诊断（必须有）

B 的最近邻条件在 `s_mid=(s_i+s_{i+1})/2` 切换，原则上 logits 可能跳跃，进而产生层间伪面。至少审计两处：**15 mm 与 45 mm**。

推荐同一 XY 网格，记录两种性质：

**A. 条件引起的跳跃：**固定 `q=(x,y,s_mid)`，只替换同一坐标所用的 `u_left`、`u_right`，计算：

\[
J_{logit}=\operatorname{mean}_{xy}|D(q,u_{right},p)-D(q,u_{left},p)|,
\]

以及两侧分类 `logit≥0` 不一致的 XY 比例。

**B. 真正三维场的近邻跳变：**同一 XY 网格比较 `q^-=(x,y,s_mid-0.1mm)` 与 `q^+=(x,y,s_mid+0.1mm)`，记录 logits 的平均差值与分类翻转比率。`0.1mm` 为仅用于诊断的偏移，不改变训练或重建。

同时将 C 在同样位置、同样网格的变化作为参照，并在 mesh 中检查新增侧向/内部异常边、片状分量、伪曲面。**最近邻的离散切换是该对照算子的性质，不应通过额外正则或 Mesh 修复消除。**

## 4. 实验 C：原 V2 作为不可改变的固定基线

- 固定训练来源为 `989601aabc0083388ec88230d3ab0eee117e5e68`。
- 使用已有 `002_ED` **2000-step final** checkpoint，及其 `effective_config.json`、输入 NPZ、原训练报告，全部记录文件 SHA256。
- 原始 `AutoOccupancy.generator`、`OccupancyDecoder`、`interpolate()`、`loo()`、`objective.py`/`chunked.py`、`train.py` 均不可覆盖或重新定义。
- C 的评估用已确认的原函数 `inference.query()`；B 才使用新 `query_nearest()`。
- 使用新统一 evaluator 再跑一遍 C，要求与原 `metrics.json` 的每层 Dice、闭合曲线数量、面积、拓扑在计算误差内一致。**不一致就先修复 evaluator，再比较 A/B**。
- 不得在 A/B 评价时根据隐藏层 Dice 改 C 的阈值、ROI、checkpoint 或轴向范围。

## 5. 文件级实施表与入口协议

推荐仅**新增独立的消融子目录**，保持已通过 V2 报告审计的源文件不变：

```text
experiments/plane_auto_occ/
├── model.py                       # 原 C，保持不变
├── objective.py                   # 原 C，保持不变
├── chunked.py                     # 原 C，保持不变
├── area_pool.py                   # 原 V2 池，复用
├── data.py                        # 原数据合同，复用
├── inference.py                   # 原 C restore/query，保持不变
├── evaluate.py                    # 原 C evaluator，保持不变
├── extract_mesh.py                # 原 C marching cubes，保持不变
└── ablations/                     # 本轮新增，不覆盖旧结果
    ├── __init__.py
    ├── model_case_only.py         # A 网络兼容类
    ├── nearest.py                 # B 最近邻选择和查询
    ├── train_case_only.py         # A 训练（复用V2采样+分块训练）
    ├── restore_variants.py        # A/C安全checkpoint加载；B与C相同
    ├── evaluate_three.py         # A/B/C统一逐层与Mesh对照
    ├── test_ablations.py          # 断言/反向传播/协议测试
    ├── config_A_case_only.json    # 原config全量复制，仅必要字段不同
    └── README.md                  # 命令、source lock与结果声明
```

这是一份**建议创建的文件清单**，不是当前 GitHub 仓库已经存在的文件列表。Agent 可以合理减少胶水模块，但**不允许用新近似实现替换原采样器、原核心 Decoder 或统一 evaluator**。

### 5.1 路径和环境变量（示例；Agent 需替换真实远端路径）

从固定 Git commit 创建独立工作分支，不应直接修改已发布的 repair 分支：

```bash
REPO=/path/to/PlaneLatentSDF
cd "$REPO"
git fetch origin
# 核对：目标分支确实包含原始指定commit。
git rev-parse origin/experiment/plane-auto-occupancy-repair
# 建议按固定提交开隔离分支；不得无意切到 main。
git switch -c experiment/plane-auto-occupancy-ablation \
  989601aabc0083388ec88230d3ab0eee117e5e68

python -m experiments.plane_auto_occ.tests
python -m experiments.plane_auto_occ.repair_tests
```

原始数据位置由远端有效数据和 `fit_manifest.json` 决定；不要自行猜测远端 GPU、Slurm 目录或 NPZ 文件名。下列环境变量只是**执行模板**：

```bash
CASE=/absolute/path/to/002_ED.npz
POOL=/absolute/path/to/observed_0_3_6_generation_000.npz
C_FINAL=/absolute/path/to/V2/last.pt
C_STEP0=/absolute/path/to/V2/step_00000/checkpoint.pt   # 如果有
OUT=/absolute/path/to/ablation_results
```

开始前必须以 `sha256sum` 核对输入 NPZ、C checkpoint 及其 `records/source_sha256`；验证其 `case_id=002`、`phase=0`、`observed=[0,3,6]`、`step=2000`、模型格式和实际 `scale_mm`。

### 5.2 Agent 实现完成后应提供的命令（**以下是待实现目标 CLI，并非原仓库现成可运行命令**）

```bash
# A：全新2000-step联合拟合。可选--init-from-c-step0用于匹配初始Decoder和case code。
python -m experiments.plane_auto_occ.ablations.train_case_only \
  --case "$CASE" --observed 0,3,6 \
  --area-pool "$POOL" \
  --config experiments/plane_auto_occ/ablations/config_A_case_only.json \
  --output "$OUT/A_train" --device cuda --steps 2000 \
  --init-from-c-step0 "$C_STEP0"

# 使用A checkpoint与同一个C checkpoint，统一做三组评估和mesh。
python -m experiments.plane_auto_occ.ablations.evaluate_three \
  --case "$CASE" \
  --a-checkpoint "$OUT/A_train/last.pt" \
  --c-checkpoint "$C_FINAL" \
  --output "$OUT/compare_final" \
  --device cuda --resolution 160 --mesh-resolution 128 \
  --margin-mm 5 --robust-margin-mm 20

# 单测入口
python -m experiments.plane_auto_occ.ablations.test_ablations
```

**CLI 实现要求：**如果 `--init-from-c-step0` 不可用，训练入口应明确接受省略它，走同 seed 随机初始化并记录限制，而非在文件缺失时静默改用第2000步权重；禁止隐式 resume、覆盖 C 的输出、利用隐藏层选最优 checkpoint。

### 5.3 A 与 C 的共同计算 budget

每个 A update 仍抽取 `32768×3=98304` 个独立点。因 A 不使用 LOO，多一次中间层 Decoder 的查询不再存在：
- A：约 98304 次 Decoder 点查询/完整更新；
- C：约 131072 次（其中中间层 32768 点被 Direct 和 LOO 查询两遍）；
- B：无额外训练，只有评估查询。

**不要把“同 2000 update”误称为“相同总算力”：C 比 A 多一个 Generator、额外中间层 LOO Decoder 查询。**这是模型消融固有的计算差异，应记录真实时间、训练 query 数和峰值显存。对照首先保持**相同训练数据量、完整更新次数与 loss 的尺度**，并透明报告计算成本。

### 5.4 远端执行要点

- 先跑 A 的 synthetic test、1～2 次完整 update 的 CUDA smoke（小步数仅验证可运行性），smoke checkpoint **不得继续作为正式2000步结果**。
- smoke 与 formal 使用不同输出目录；formal 重新初始化 code、Decoder、Adam、随机状态和采样池读取游标。
- 正式 A 只跑 2000 updates，固定最终 checkpoint；不设依 Dice 自动停止/选择模型。
- B/C 推理无需梯度、无需 optimizer；加载最终模型 `eval()`，对 C checkpoint 保持只读。
- C 的原始 checkpoint、评估、源锁和 V1/V2 结果全部保留，不做重命名覆盖。

## 6. 必须通过的单元测试与协议断言

Agent 至少提供这些可自动验证的测试。

### 6.1 模型构造与参数依赖

1. **Decoder 同构**：A 与 C 的 `decoder` 具备逐层同名、同形状的全部 Linear 权重；测试 `5 Linear + 4 Softplus(beta=10)`、`width=256`、输入 132D、输出 `[Q]`。
2. **Case-only 恒定条件**：对于任意 `s=[0,.3,.6]`，A 的 `.plane_latents()` 三行逐项完全相等且均等于同一个 case code；梯度由所有查询点累计到这**唯一**的 code。
3. **没有隐含 Generator**：A 的 `named_parameters()` 不允许出现 `generator.*` 或任何 per-plane learnable embeddings；没有 `nn.Parameter(N,128)` 的 plane codes。
4. **B/C 不共享可变状态**：分别推理时原始 C checkpoint 及模型权重 SHA/数值均不改变。B 的最近邻不可偷偷重新优化 case code。

### 6.2 插值与最近邻的解析单测

使用高度 `s=[0,.3,.6]`，手工设置可区分的 `u0,u3,u6`，并验证：

| 查询高度 | C 的线性插值 | B 的最近邻 |
|---|---|---|
| 0.00 | `u0` | `u0` |
| 0.10 | `(2/3)u0+(1/3)u3` | `u0` |
| 0.15 | `0.5u0+0.5u3` | **`u0`（平局向左）** |
| 0.20 | `(1/3)u0+(2/3)u3` | `u3` |
| 0.30 | `u3` | `u3` |
| 0.40 | `(2/3)u3+(1/3)u6` | `u3` |
| 0.45 | `0.5u3+0.5u6` | **`u3`（平局向左）** |
| 0.50 | `(1/3)u3+(2/3)u6` | `u6` |
| 0.60 | `u6` | `u6` |

**禁止轴向外推**：低于 `s0-1e-5` 和高于 `s6+1e-5` 的查询必须显式报错；保留 V2 的数值容忍阈值。

### 6.3 A 的 loss/梯度等价性

在 synthetic 三层输入上：

\[
L_A=1.5(D_0+D_3+D_6)/3+10^{-4}\operatorname{mean}(z_c^2).
\]

验证：
- 若强制 `u0=u3=u6=z_c`，则 `O3 == D3`，C 型层内混合在**数值和梯度**上与 A 的 `Direct-only` 同等（同 Decoder、同样本、同权重条件下）。
- 分块 `4096` 梯度累计与全量前向/反向在浮点误差内一致；每 complete update 仅一次 `Adam.step()` 和一次 clip。
- `latent` 正则仅 backward 一次，不被乘以层数/块数；分支 C/A 各组权重总数据系数都是 1.5。
- 训练/测试标签只用 `observed=[0,3,6]`；向隐藏 plane 的轮廓施加大幅度改动不应影响已构建 pool、A 训练输入及边界 ROI。

### 6.4 同 checkpoint 公平性及 evaluator 回归

1. **B 在所有已观测平面与 C 一致**：在同一 `160²` 网格上，分别于 0/30/60 mm 检查 logits `max_abs_diff <=1e-5`，分类像素逐项完全一致，Dice 与拓扑完全相等。不一致即实验无效。
2. **C 原生结果回归**：统一 evaluator 重新计算 C 七层 Dice，逐项与 §1.3 原结果在例如 `1e-4` 绝对误差内一致。若有不同的执行设备/库版本，必须调查并报告差异，不得放宽阈值掩盖代码改变。
3. **同一评估查询点**：A/B/C 的 7 层、2 个无标签 gap、3D Mesh 都使用同样的点阵和相同 ROI/阈值，不允许按各模型预测 bbox 分别裁剪。
4. **不使用隐藏标签进行模型操作**：隐藏 `[1,2,4,5]` 仅在完整训练结束后用于打分；不能参与 A 的取样域、初始化、loss、checkpoint 选择、阈值或后处理。
5. **NN 中点单测**：固定 `q` 的 `u_left/u_right` 对照可以检测条件跳跃，确认与选左 tie 规则一致。

## 7. 统一评价与输出文件规范

### 7.1 必须报告的逐层指标

- 每张 SAX 的 `index`、`position_mm`、`observed / heldout / unlabelled_gap`；对于无 GT gap，Dice 必须为 `null/N/A`，不能填0。
- 统一网格 `contour_proxy_dice`，心肌 Precision、Recall，预测/真值心肌面积比；心腔与背景的 false-positive rate。
- 预测轮廓 `curves`、`closed_curves`、`open_curves`、`positive_components`、`positive_touches_border`，必要时 Euler characteristic / 单环判断。
- **Epi、Endo 分别到 GT 的平均有符号偏移（mm）**和无符号距离；说明距离方向和匹配方法，避免将单向距离误称对称 ASSD。
- `logit_min/max`、各类预测概率/分类统计（可选但推荐用于解释厚壁问题）。
- 对 B 和 C，观测层必须一致；对 A，除隐藏层外也必须报告三张观测层拟合差异，避免“某方法因为观测层没拟合好才显得层间差”。

### 7.2 汇总指标（建议 JSON + CSV 同时保存）

```text
observed_mean_dice       = mean(Dice[0], Dice[3], Dice[6])
heldout_mean_dice        = mean(Dice[1], Dice[2], Dice[4], Dice[5])
heldout_minus_observed   = heldout_mean_dice - observed_mean_dice
observed_ring_coverage   = valid_ring_count(observed) / 3
heldout_ring_coverage    = valid_ring_count(heldout) / 4
observed_epi_offset_mm   = mean(per-plane observed epi signed offset)
observed_endo_offset_mm  = mean(per-plane observed endo signed offset)
```

`heldout_minus_observed` 是辅助诊断，不宜单独排名：如果一个模型观测 Dice 更低，它可能得到看似更小的差值。**主结果必须联合查看观测绝对精度与隐藏绝对精度。**

### 7.3 Mesh 与层间不连续性

每组输出标准 `128³` mesh + 报告：

- 顶点、面、连通分量、非流形边、边界边（区分 axial/lateral/other）、有无侧向触边、网格空间尺度。
- 检查四条轴向边界环的预期；因无封盖，`watertight=false` **不自动代表失败**。
- 对 B，重点检查15mm/45mm切换面附近额外的平面片、轮廓跳跃、局部裂缝、内部异常边和连通分量改变。
- 同一三维网格可输出 `logit` volume/抽样 `npz`（可选）。三组在该体积上的距离统计或二值差异可以补充几何解释，不能代替已有 GT 的 SAX Dice。
- 主视觉：**A/B/C 三列，7 张有 GT SAX 为行**，相同坐标范围、相同窗宽；统一显示预测心肌区域、真实 Epi/Endo 轮廓。另加无标签 15/45mm 层间比较，以及三组同坐标视角的 Mesh。

### 7.4 建议输出树

```text
ablation_results/
├── protocol.json                     # 固定source commit/随机种子/输入哈希/全部设置
├── A_train/
│   ├── effective_config.json
│   ├── fit_manifest.json
│   ├── history.jsonl
│   └── last.pt                      # 完整第2000步
├── compare_final/
│   ├── A_case_only/
│   │   ├── metrics.json
│   │   ├── myocardium.obj
│   │   └── myocardium.json
│   ├── B_plane_nearest/
│   │   ├── metrics.json
│   │   ├── myocardium.obj
│   │   └── myocardium.json
│   ├── C_plane_linear/
│   │   ├── metrics.json
│   │   ├── myocardium.obj
│   │   └── myocardium.json
│   ├── metrics_all_planes.csv
│   ├── comparison_summary.json
│   ├── plane_triptych.png
│   ├── mesh_triptych.png
│   ├── nearest_discontinuity.json
│   └── validation_checks.json
└── REPORT.md
```

所有执行产物保留在远端输出目录；**不要把真实病例数据、checkpoint、OBJ 或大量中间 cache 自动提交 GitHub**。按现有仓库 policy 只提交代码、配置、文档和不含隐私/大文件的验证结论。

## 8. 结果怎样解释：预注册的判定矩阵

**一级判断：对比绝对隐藏层精度，同时核对观测拟合和拓扑。** 本次没有事先确定“超过多少个百分点才有统计显著性”，不可在看完结果后自造一个偏好的阈值来宣称成功。先作为单病例探索性证据，必要时固定协议做跨病例重复。

| 观测结果 | 有效结论 | 不允许作出的结论 |
|---|---|---|
| **C 优于 A**，观测拟合相近且隐藏层/拓扑更好 | 完整 Plane-latent 方法相对于简单 Auto-Decoder 有增量价值 | 不能直接证明 Generator 单独有效，更不能证明 LOO 独立有效 |
| **C 与 A 接近** | 当前没有证据表明更复杂的 Plane-latent 组合值得额外成本 | 不能由一个病例证明所有病例无收益 |
| **A 优于 C** | 当前 Plane-latent 组合可能限制拟合或不适合该病例 | 不代表所有层间插值设计都无效 |
| **C 优于 B**，B 与 C 观测预测一致 | 在当前已训练 C checkpoint 上，连续插值较最近邻带来更好隐藏截面/连续性 | 不能证明最近邻从头训练后必然更差 |
| **C 与 B 接近** | 当前证据不足以支持线性插值相对最近邻的实际增量 | 不说明线性插值代码没有参与计算 |
| **B 出现明显切换面伪曲面** | 最近邻的非连续条件不适合直接作为此模型的连续场查询算子 | 单靠 Mesh 难看不能推断插值改善了定量 Dice |
| **C>A 且 C>B** | 支持保留 Plane-latent 设计与连续插值，并进入下一轮 LOO 消融 | 仍未证实 Case latent 跨病例先验或 LOO 的独立收益 |

**第二级判断（未来，不属于本轮部署）**：在 `C>A`、`C>B` 的前提下，同协议重新训练 `C_LOO=0`，比较同一线性插值网络有无 LOO 的性能。否则优先重新审视整体条件表征，而不是修改 LOO 细节。

**外推限制：**这一轮只有 `002_ED` 单病例，不证明 ED/ES 稳健性、跨病例普适性或新病例仅优化 Case latent 的形状先验性能。即便 A/C 单病例出现差异，之后仍需要独立患者级验证。

## 9. 对 Agent 的交付验收清单

- [ ] 固定 `989601a...` 源提交，新建隔离分支；附 `git diff --stat` / `git status` / 修改文件清单。
- [ ] C 已发布的 `model.py`、`objective.py`、`chunked.py`、`area_pool.py`、`inference.py`、`evaluate.py`、`extract_mesh.py` 逐文件保持不变；若确实必须修改，逐项说明必要原因并提供回归证据。
- [ ] A 使用同一个真实 `OccupancyDecoder` 实现，直接以 case code 为 128D 条件，**无 Generator、无独立 plane code**，Loss = `1.5 mean(D_i) + 1e-4 mean(code²)`。
- [ ] B 只替换查询阶段的 `interpolate` 为 `nearest`，同一个 C checkpoint，禁止微调或重新训练。
- [ ] A 2000 次完整 update，点预算和采样协议与 C 一致，所有 C / A 的计算成本差异明确报告。
- [ ] 原数据、C checkpoint、A checkpoint/源码、采样池、阶段输出均建立 SHA256/source-lock，能复现输入选择。
- [ ] 全部 synthetic tests + V2 原 `tests`/`repair_tests` 通过，原 C 七层评分再现；B/C 观测层 logits 与分类一致。
- [ ] 隐藏索引1/2/4/5只用于后验评估；不在训练、采样域、阈值/模型选择时读取其标签。
- [ ] A/B/C `160²` 统一网格/ROI/阈值、`128³` Mesh、真实毫米坐标一致；+20mm 额外稳健性分析完成。
- [ ] B 15/45mm 最近邻跳变审计，另报告条件固定 `q` 的 logit jump 和分类变化。
- [ ] 输出逐层 CSV、汇总 JSON、三组可视化与完整 `REPORT.md`；所有差异带具体数值，不只写“效果更好”。
- [ ] 明确单病例、综合机制 vs 算子级推理消融的因果边界；不预先宣称哪组必胜。
- [ ] 仅在用户授权要求下才 push 新实验代码到 GitHub；不得覆盖原 repair 分支或原实验输出。

## 10. 源文件定位与资料

现有 V2（本文件依据，Coding Agent 需重新固定 SHA256）：

- [`experiments/plane_auto_occ/model.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/model.py)：`AutoOccupancy`、`OccupancyDecoder`、`interpolate()`、`loo()`。
- [`experiments/plane_auto_occ/chunked.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/chunked.py)：V2 每层 32768 点 / 4096 分块、loss 权重和梯度累积。
- [`experiments/plane_auto_occ/objective.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/objective.py)：`mix_plane_losses()` 数学合同。
- [`experiments/plane_auto_occ/area_pool.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/area_pool.py)：observed-only 区域均匀池和 `AreaSampler`。
- [`experiments/plane_auto_occ/train.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/train.py)：Case code 参数、优化组、训练/保存格式。
- [`experiments/plane_auto_occ/inference.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/inference.py)：checkpoint restore、统一3D查询。
- [`experiments/plane_auto_occ/evaluate.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/evaluate.py)：`160²` 网格、严格隐藏层和拓扑评价。
- [`experiments/plane_auto_occ/extract_mesh.py`](https://github.com/hwu918945-alt/PlaneLatentSDF/blob/989601aabc0083388ec88230d3ab0eee117e5e68/experiments/plane_auto_occ/extract_mesh.py)：`128³`、`logit=0`、无 padding/cap Mesh。
- 用户已提供的《002_ED 面积均匀采样与层权重修复：2000 步结果分析》（REPORT(1).md）：基线 Dice、壁厚偏移、LOO 梯度余弦和 Case code 置零结果；用作解释基准，不当作消融完成证据。

---

**本轮唯一需要新增的正式训练是 A。** B 用原 C checkpoint 做最近邻推理，C 是原线性插值基线。保留相同 128D Decoder 条件、相同数据/训练预算与评价协议，将两个核心问题明确拆开。测试完成后再决定是否值得开展 LOO Loss 独立消融。
