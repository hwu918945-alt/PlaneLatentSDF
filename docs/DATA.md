# 预处理缓存与来源

## 心脏输入

`data/aligned_sax_manifest.json` 保存 714 个患者时相的闭合 epi 轮廓、平面原点与正交基、病例 ID、ED/ES 和固定划分。它来自 v11_5static 已对齐缓存，适配器用 `contours * scale + center` 恢复已对齐毫米坐标，没有在这里重新进行影像配准。适配与来源记录在 `data/alignment_adapter_audit.json`、`provenance/source_files.json` 和 `provenance/static_endpoint_manifest.json`。

`data/plane_sdf/` 包含 Train 285 人 / 570 时相，Val 36 人 / 72 时相，Test 36 人 / 72 时相。3 个 stress 病例不进入主实验。模型输入不包含原图像、原分割 mask、LAX、运动或模板。

每个 NPZ 可用 `numpy.load(path, allow_pickle=False)` 读取，加载入口是 `core/data.py:load_case()`：

| 键 | 形状 / 定义 |
| --- | --- |
| `q` | `[N,1024,3]`，每个真实平面的归一化三维查询坐标 |
| `sdf2d` | `[N,1024]`，真实平面填充轮廓的有符号距离；内部负、外部正 |
| `boundary` | `[N,256,3]`，等弧长边界点 |
| `s` | `[N]`，沿堆栈轴严格递增的归一化平面位置 |
| `axis` | `[3]`，堆栈单位轴 |
| `center_mm` / `scale_mm` | `[3]` / 标量；固定 `scale_mm=100` |
| `phase` / `case_id` | ED=0、ES=1；患者身份 |
| `anatomy` / `provenance_json` | 解剖目标与输入来源元数据 |

换回已对齐毫米坐标：`world_mm = q * scale_mm + center_mm`；距离也乘同一个 `scale_mm`。没有按病例 bbox 单独缩放。`sdf2d` 是真实平面的 **2D SDF**，不是完整三维 GT；未知中间层没有伪造 SDF 标签。

## 采样与重新生成

`prepare_cache.py:convert()` 每层采样 768 个近边界点（1/3/6 mm 高斯扰动）和 256 个扩展 bbox 均匀点；边界为 256 个等弧长点。`core/data.py:polygon_sdf()` 计算到真实边界线段的距离和射线奇偶符号。训练的 `select_view()` 从缓存抽点；`gap_queries()` 提供无标签间隙查询 proposal，供几何与 Eikonal 项使用。

已有缓存可直接使用。若要从自包含 manifest 重新生成，必须使用新的输出目录，转换器会拒绝覆盖：

```bash
python prepare_cache.py --manifest data/aligned_sax_manifest.json --output data/plane_sdf_rebuilt
```

无需运行 `adapt_static_cache.py`，也无需拥有原 v11_5static 仓库。只有改用其他已对齐缓存时才需要适配器；元数据中的旧 GPFS 路径属于来源记录，不是已发布缓存的读取依赖。

`python scripts/verify_release.py` 检查文件 SHA-256、NPZ 契约、ED/ES 配对、患者划分隔离、固定尺度和实验输入锁。更详细的原始标签/轮廓检查仍在 `scripts/verify_dataset.py`，从根目录运行会生成本地 `preflight/` 检查数据。

## 通用模型输入

每套实验的 `source/` 保留参考模型与出处，`data/` 保留共享未见切面与评价范围，`slices_NNN/data/` 保留可见切面 NPZ 与真实轮廓。完整参考模型不进入拟合 loss。

- Bunny 来源：[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)。原 archive、PLY、原说明及明确修补五个底部孔洞的参考几何均保留；沿用 `1 源单位 = 1000 benchmark mm`。
- Balloon Dog 来源：[OReX](https://github.com/haimsaw/OReX)，固定源 commit `089db1f757022606399d5f070d65bd172fe69362`；使用 `balloondog_scaled.stl`。原 CSL 仅保留来源，本实验重新生成平行切面，未修补参考模型；沿用 `1 源单位 = 100 benchmark mm`。下载记录在 `source/download_provenance.json`。

这些 benchmark mm 不代表已核实的实物毫米。Bunny v1 使用 45 个未见切面；两套密度对照各使用预先冻结、组间共享的 121 个未见切面。多环切面的 SDF 使用真实边界线段与奇偶符号，不把独立环人为连成一个多边形。

原数据保留各自的来源、归属与引用说明；本次整理没有为第三方数据重新授予统一许可证。
