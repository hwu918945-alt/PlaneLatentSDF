# 当前心肌 shell 数据契约

完整准备方法见 [SHELL_DEPLOYMENT.md](SHELL_DEPLOYMENT.md)。原 `data/plane_sdf` 保留为历史 Epi 缓存；新主训练要求 `data/plane_sdf_shell`。

| NPZ key | 定义 |
|---|---|
| `schema`, `anatomy` | `sax_myo_shell_v1`, `LV_myocardium_shell` |
| `q`, `sdf2d` | `[N,8192,3]`, `[N,8192]`；连续真实二维心肌 SDF |
| `sampling_source` | `[N,8192]`；0..6 对应 epi_zero/epi_gaussian/endo_zero/endo_gaussian/myocardium/cavity/background |
| `boundary`, `boundary_endo` | Epi/Endo 各 `[N,512,3]`，连续轮廓等弧长点 |
| `s`, `axis` | `[N]` 严格递增物理高度 / `[3]` 单位轴 |
| `center_mm`, `scale_mm` | 首层有效平面原点 / 固定 100 mm |
| `case_id`, `phase` | 原患者 ID；ED=0、ES=1 |
| `source_slice_indices`, `provenance_json` | 原切片索引及来源 |

世界坐标 `world_mm=q*100+center_mm`；距离 `distance_mm=sdf2d*100`。所有采样来源的真值都通过 `max(d_epi,-d_endo)` 计算；心肌负、心腔/外部正，真实边界显式赋 0。候选数固定为 `[512,1536,512,1536,2048,1024,1024]`。不是给 gap 标注三维 SDF。

默认验证/推理仅从池中确定性抽 2048 点；Train Student/Teacher 与 Decoder 各独立分层随机抽样。源/标签语义和完整配置见 [确认规格](SAX_Myocardium_Spec.md)。旧 Bunny/balloon-dog 使用各自独立 runtime 和原输入协议。
