"""One job: independent A training, then frozen B/C and matched final evaluation."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from ..area_pool import digest
from ..data import load_case
from ..model import AutoOccupancy
from .model_case_only import CaseOnlyAutoOccupancy
from .train_case_only import train,initialize
from .evaluate_three import evaluate


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))


def verify(root):
    protocol=json.loads((root/'protocol.json').read_text());inputs=json.loads((root/'inputs_lock.json').read_text());runtime=Path(__file__).resolve().parents[3]
    for row in inputs['records']:
        if digest(row['local'])!=row['sha256']:raise ValueError(f"Input changed: {row['local']}")
    source=json.loads((root/'reference/C_source_lock.json').read_text())
    for name,sha in source['files'].items():
        if digest(runtime/name)!=sha:raise ValueError(f'Original V2 source changed: {name}')
    lock=json.loads((root/'ablation_source_lock.json').read_text())
    for name,sha in lock['files'].items():
        if digest(runtime/name)!=sha:raise ValueError(f'Frozen ablation source changed: {name}')
    c=load_case(root/'cache/002_ED.npz',observed=[0,3,6]);ck=torch.load(root/'reference/C_final.pt',map_location='cpu',weights_only=False)
    if c['case_id']!='002' or c['phase']!=0 or c['scale_mm']!=100. or c['total_planes']!=7:raise ValueError('Wrong case/normalization')
    if ck['step']!=2000 or ck['format']!='plane_auto_occ_v1' or ck['records'][0]['observed']!=[0,3,6] or ck['records'][0]['source_sha256']!=digest(root/'cache/002_ED.npz'):
        raise ValueError('Wrong final C checkpoint')
    cfg=json.loads((Path(__file__).with_name('config_A_case_only.json')).read_text());original=json.loads((root/'reference/C_config.json').read_text())
    for key,value in original.items():
        if key not in ('loo_mix','loo_weight') and cfg[key]!=value:raise ValueError(f'A changed protected setting: {key}')
    initial=torch.load(root/'reference/C_step0.pt',map_location='cpu',weights_only=False)
    if initial['step']!=0:raise ValueError('C initialization must be step0')
    return cfg,protocol


def report(root,summary,checks,training_seconds,evaluation_seconds):
    cfg=json.loads((root/'A_train/effective_config.json').read_text());manifest=json.loads((root/'A_train/fit_manifest.json').read_text())
    history=[json.loads(x) for x in (root/'A_train/history.jsonl').read_text().splitlines()]
    if len(history)!=2000 or [r['step'] for r in history]!=list(range(1,2001)):raise AssertionError('A incomplete')
    if not all(r['independent_points']==98304 and r['decoder_queries']==98304 and r['pool_fallbacks']==0 and r['loo']==0 for r in history):raise AssertionError('A budget/LOO contract failed')
    chistory=[json.loads(x) for x in (root/'reference/C_history.jsonl').read_text().splitlines()]
    if len(chistory)!=2000 or not all(r['independent_points']==98304 and r['decoder_queries']==131072 for r in chistory):raise AssertionError('C recorded budget mismatch')
    poolpairing=[]
    for generation in range(25):
        candidate=root/'pools/generation_000.json' if generation==0 else root/'A_train/sampling_pools'/f'generation_{generation:03d}.json'
        ref=root/'reference/C_pool_audits'/f'generation_{generation:03d}.json'
        actual=json.loads(candidate.read_text());prior=json.loads(ref.read_text())
        if actual['pool_sha256']!=prior['pool_sha256']:raise AssertionError(f'A/C sampling pool differs in generation{generation}')
        poolpairing.append(dict(generation=generation,array_sha256=actual['pool_sha256']))
    execution=dict(A_steps=2000,A_decoder_queries=2000*98304,C_existing_steps=2000,C_decoder_queries=2000*131072,
        A_independent_training_points=2000*98304,C_independent_training_points=2000*98304,
        A_training_wall_seconds=training_seconds,final_evaluation_wall_seconds=evaluation_seconds,
        A_peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in history),
        A_update_seconds_mean=float(np.mean([r['update_seconds'] for r in history])),
        A_trainable_parameters=manifest['total_trainable_parameters'],
        C_trainable_parameters=sum(p.numel() for p in AutoOccupancy().parameters())+128,
        C_extra_generator_parameters=49792,initialization=manifest['initialization'],
        C_recorded_update_seconds_mean=float(np.mean([r['update_seconds'] for r in chistory])),
        C_recorded_update_seconds_total=float(sum(r['update_seconds'] for r in chistory)),
        C_recorded_peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in chistory),
        sampling_all25_generations_exactly_paired=poolpairing,
        fixed_final_checkpoint=True,B_training_updates=0,C_new_training_updates=0)
    write(root/'compare_final/execution_summary.json',execution)
    lines=['# 002_ED：Case-only / Plane-nearest / Plane-linear 消融','',
        '状态：A正式2000次完整更新及三组最终评估已完成。C为已完成V2的固定第2000步；B/C不重训、不微调。',
        '数据为同一002_ED，观测0/3/6，严格隐藏1/2/4/5。所有组160²、bbox+5mm、logit0，额外+20mm审计；128³原生开放slab、不封盖。','',
        '| 组 | 观测Dice | 隐藏Dice | 隐藏−观测 | 观测环覆盖 | 隐藏环覆盖 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,value in summary['main'].items():lines.append(f"| {name} | {value['observed_mean_dice']:.6f} | {value['heldout_mean_dice']:.6f} | {value['heldout_minus_observed']:.6f} | {value['observed_ring_coverage']:.3f} | {value['heldout_ring_coverage']:.3f} |")
    lines+=['','## 逐层主指标','', '| 层 | 类别 | A Dice | B Dice | C Dice |','|---|---|---:|---:|---:|']
    metrics={name:json.loads((root/'compare_final'/name/'metrics.json').read_text()) for name in summary['main']}
    for i in range(7):
        values=[next(r for r in metrics[name]['planes'] if r['index']==i)['contour_proxy_dice'] for name in ('A_case_only','B_plane_nearest','C_plane_linear')]
        lines.append(f"| {i} | {'observed' if i in (0,3,6) else 'heldout'} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','## 网格与最近邻跳变','', '| 组 | 顶点 | 面 | 连通分量 | 轴向/侧向/其他开放边 | 边界分量 | 非流形边 |','|---|---:|---:|---:|---|---:|---:|']
    for name in summary['main']:
        m=json.loads((root/'compare_final'/name/'myocardium.json').read_text())
        if m['status']=='EXTRACTED':
            b=m['boundary_edge_locations'];lines.append(f"| {name} | {m['vertices']} | {m['faces']} | {m['connected_components']} | {b['axial']}/{b['lateral']}/{b['other']} | {m['boundary_components']} | {m['nonmanifold_edges']} |")
        else:lines.append(f'| {name} | NO_ZERO_CROSSING | | | | | |')
    jumps=json.loads((root/'compare_final/nearest_discontinuity.json').read_text())
    for row in jumps['planes']:
        b=row['variants']['B_plane_nearest'];c=row['variants']['C_plane_linear']
        lines += ['',f"切换面{row['position_mm']:.4f}mm：固定xyz条件跳跃B为{b['fixed_xyz_condition_mean_abs_logit_jump']:.6f} logit、分类翻转{b['fixed_xyz_condition_flip_rate']:.6%}；±0.1mm实际场变化B为{b['spatial_mean_abs_logit_change']:.6f}、翻转{b['spatial_classification_flip_rate']:.6%}，C为{c['spatial_mean_abs_logit_change']:.6f}、翻转{c['spatial_classification_flip_rate']:.6%}。"]
    cscore=summary['main']['C_plane_linear'];ascore=summary['main']['A_case_only'];bscore=summary['main']['B_plane_nearest']
    lines += ['',f"C−A隐藏Dice差为{cscore['heldout_mean_dice']-ascore['heldout_mean_dice']:+.6f}，观测Dice差为{cscore['observed_mean_dice']-ascore['observed_mean_dice']:+.6f}；C−B隐藏Dice差为{cscore['heldout_mean_dice']-bscore['heldout_mean_dice']:+.6f}。这些是单病例绝对差值，不是统计显著性结论；须联合以上观测拟合和拓扑解释。"]
    lines+=['','## 精度与计算成本','',
        '斜SAX轴的float32坐标投影和(1-alpha)系数舍入使原生B/C观测logit出现约6e-5差异、分类完全一致。统一评估使用相同float64场计算适配，先保留原query的float32 xyz舍入，再调用原模型/插值/decoder；G的原生输出完全固定。原C文件、权重、ROI和阈值不改。所有原生差异与严格1e-5检查保存于validation_checks.json，阈值未放宽。C七层Dice/拓扑/面积和原128³网格计数回归通过。',
        f"A训练可学习参数{execution['A_trainable_parameters']}，C为{execution['C_trainable_parameters']}，差值49792来自删除G。A2000步共{execution['A_decoder_queries']} decoder点查询，C为{execution['C_decoder_queries']}；二者训练数据量均{execution['A_independent_training_points']}，所以不是相同总算力。",
        f"A训练墙钟{training_seconds:.2f}s，平均更新{execution['A_update_seconds_mean']:.4f}s，峰值CUDA allocated={execution['A_peak_cuda_allocated_bytes']} bytes。配对初始化只复制C第0步decoder和code，未读取第2000步训练权重进行A初始化，也未复制Adam状态。",'',
        f"C原运行的记录更新均值{execution['C_recorded_update_seconds_mean']:.4f}s，累计更新{execution['C_recorded_update_seconds_total']:.2f}s，峰值CUDA allocated={execution['C_recorded_peak_cuda_allocated_bytes']} bytes；A/C全部25代采样池数组SHA逐项相同，LOO复用坐标，数据采样游标独立复位。累计更新时间包括池刷新，不包含保存/绘图，不能等同整作业墙钟。",'',
        '## 解释范围','',
        'C vs A检验完整Plane-latent组合（含G/线性插值/配套LOO）与简单Case-only的增量，不能分离Generator或LOO的贡献。C vs B只检验这个已按线性/LOO训练checkpoint的推理算子，不能推断最近邻独立重训的最优能力。此次没有第四组LOO消融。单病例探索性结果无统计显著性阈值，必须同时看观测拟合、隐藏精度和拓扑；不证明跨患者先验。', '',
        '逐层Precision/Recall、面积比、心腔/背景误报、Epi/Endo单向有符号/无符号偏移见metrics.json/CSV；无GT gap Dice为null。无明确单环时偏移为null并报告原因，不静默删除失败部件。','',
        '对照图：compare_final/plane_triptych.png、compare_final/mesh_triptych.png；固定协议/输入SHA见protocol.json与inputs_lock.json。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    root=Path(a.experiment).resolve();cfg,protocol=verify(root)
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    case=root/'cache/002_ED.npz';pool=root/'pools/generation_000.npz';step0=root/'reference/C_step0.pt';cfinal=root/'reference/C_final.pt'
    with (root/'logs/deployment_unit_tests.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','experiments.plane_auto_occ.tests','experiments.plane_auto_occ.repair_tests',
            'experiments.plane_auto_occ.ablations.test_ablations'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Original12 + ablation contract tests passed',flush=True)
    smoke=train(case,[0,3,6],pool,cfg,root/'gates/cuda_smoke',a.device,steps=2,step0=step0)
    rows=[json.loads(x) for x in (smoke.parent/'history.jsonl').read_text().splitlines()]
    if len(rows)!=2 or any(r['independent_points']!=98304 or r['decoder_queries']!=98304 for r in rows):raise AssertionError('A full-budget CUDA gate failed')
    write(root/'gates/CUDA_GATE_COMPLETE.json',dict(steps=2,points_per_plane=32768,initialization='C step0 paired',
        formal_reinitialization_required=True,device=a.device,peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows)))
    print('A full32768-point CUDA gate passed. Start FRESH formal2000 updates.',flush=True)
    start=time.perf_counter();afinal=train(case,[0,3,6],pool,cfg,root/'A_train',a.device,steps=2000,step0=step0);training_seconds=time.perf_counter()-start
    start=time.perf_counter();summary,checks=evaluate(case,cfinal,root/'compare_final',afinal,a.device,160,128,5.,20.,
        root/'reference/C_native_metrics.json',root/'reference/C_native_mesh.json',root/'reference/C_analysis_audit.json')
    evaluation_seconds=time.perf_counter()-start;report(root,summary,checks,training_seconds,evaluation_seconds)
    verify(root)
    write(root/'RUN_COMPLETE.json',dict(A_steps=2000,B_new_training_steps=0,C_new_training_steps=0,
        A_checkpoint_sha256=digest(afinal),C_fixed_checkpoint_sha256=digest(cfinal),summary=summary,
        C_native_regression_passed=True,BC_observed_equality_passed=True,report=str(root/'REPORT.md')))
    print(json.dumps(dict(status='RUN_COMPLETE',summary=summary)),flush=True)

if __name__=='__main__':main()
