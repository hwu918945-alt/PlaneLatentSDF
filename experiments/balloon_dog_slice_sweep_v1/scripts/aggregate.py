"""Report every prespecified setting, retaining missing/failed results."""
import csv
import json
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_dog import plot_mesh
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    domain=json.loads((ROOT/'data/evaluation_domain.json').read_text())
    lower=np.asarray(domain['lower_mm']);upper=np.asarray(domain['upper_mm'])
    upper_height=lower[2]+.8*(upper[2]-lower[2])
    rows=[];fig=plt.figure(figsize=(14,17))
    for i,count in enumerate(protocol['slice_counts']):
        group=ROOT/f'slices_{count:03d}'
        logs=group/'runs/main/metrics.jsonl'
        training={r['global_step']:r for r in map(json.loads,logs.read_text().splitlines())} if logs.exists() else {}
        for j,step in enumerate(protocol['milestones']):
            out=group/f'outputs/step_{step:06d}'
            row=dict(observed_planes=count,optimizer_steps=step,status='MISSING',heldout_dice=None,upper_20_percent_heldout_dice=None,
                observed_dice=None,voxel_dice=None,surface_mcd_mm=None,surface_hd95_mm=None,mesh_components=None,
                heldout_contour_coverage=None,gap_gradient_norm_median=None,training_eikonal=None)
            ax=fig.add_subplot(4,3,i*3+j+1,projection='3d')
            marker=out/'EVALUATION_COMPLETE.json'
            if marker.exists():
                m=json.loads(marker.read_text())
                assert m['observed_planes']==count and m['optimization_steps']==step and m['heldout_planes']==121
                slices=json.loads((out/'slice_metrics.json').read_text())['heldout']
                upper_rows=[r['dice'] for r in slices if r['height_mm']>=upper_height]
                row.update(status='COMPLETE',heldout_dice=m['sections']['heldout']['dice'],upper_20_percent_heldout_dice=float(np.mean(upper_rows)),
                    observed_dice=m['sections']['observed']['dice'],voxel_dice=m['voxel_dice_observed_slab'],
                    surface_mcd_mm=m['surface_observed_slab']['mcd_mm'],surface_hd95_mm=m['surface_observed_slab']['hd95_mm'],
                    mesh_components=m['mesh_connected_components'],heldout_contour_coverage=m['sections']['heldout']['contour_coverage'],
                    gap_gradient_norm_median=m['gap_gradient_norm_quantiles'][2],training_eikonal=training.get(step,{}).get('eikonal'))
                mesh=trimesh.load(out/'balloon_dog_reconstruction.obj',process=False)
                plot_mesh(ax,mesh,lower,upper,f'{count} slices / {step} steps\nUnseen Dice {row["heldout_dice"]:.3f}; CC {row["mesh_components"]}')
            else:
                row['status']='EVALUATION_FAILED' if (out/'EVALUATION_FAILED.json').exists() else 'MISSING'
                ax.text2D(.2,.5,f'{count} slices / {step} steps\n{row["status"]}',transform=ax.transAxes);ax.set_axis_off()
            rows.append(row)
    result=ROOT/'results';result.mkdir(exist_ok=True)
    with (result/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (result/'comparison.json').write_text(json.dumps(dict(protocol=protocol,evaluation_domain=domain,
        upper_band_definition='height >= lowest_observed + 0.8 * observed_height_span; proxy, not semantic feature segmentation',
        all_settings_reported=True,selected_checkpoint=None,rows=rows),indent=2,allow_nan=False))
    fig.tight_layout();fig.savefig(result/'comparison_meshes.png',dpi=160,transparent=True);plt.close(fig)
    lines=['# Balloon Dog 切面数量 × 优化步数实测对比','',
        '所有组使用同一批 121 个未见切面。三维评价范围、网格和镜头一致；各组保留全部分量，不封口。',
        '25/50/75 层参考 CrossSDF 的切面数量实验，未更换为 CrossSDF 方法。数值距离为此前约定的 benchmark mm。',
        '每组从头训练 6000 步，在 2000/4000/6000 步留存结果；EMA horizon 都是 6000。所有组使用同一预设 EMA horizon。','',
        '| 可见层数 | 优化步数 | 未见 Dice | 上部 20% Dice | 3D Dice | 3D MCD | 连通分量 | 状态 |',
        '|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):return '—' if x is None else f'{x:.4f}'
    for r in rows:
        lines.append(f'| {r["observed_planes"]} | {r["optimizer_steps"]} | {fmt(r["heldout_dice"])} | {fmt(r["upper_20_percent_heldout_dice"])} | {fmt(r["voxel_dice"])} | {fmt(r["surface_mcd_mm"])} | {r["mesh_components"] if r["mesh_components"] is not None else "—"} | {r["status"]} |')
    lines+=['','上部 20% 是固定高度范围的辅助指标，不能代替人工判断各细长部位是否连接。相同步数下，更多切面会有更多监督点与更长耗时，并非相同计算量对照。',
        '本实验不按这些测试指标挑选 checkpoint；失败或缺失的设置保留在表中。','',
        '可视化：`comparison_meshes.png`。逐组图、OBJ、完整指标在 `../slices_NNN/outputs/step_NNNNNN/`。']
    (result/'README.md').write_text('\n'.join(lines)+'\n')
    complete=all(r['status']=='COMPLETE' for r in rows)
    (result/'SWEEP_COMPLETE.json').write_text(json.dumps(dict(status='EXECUTION_COMPLETE' if complete else 'EXECUTION_INCOMPLETE',
        evaluated_settings=sum(r['status']=='COMPLETE' for r in rows),expected_settings=12,geometric_success_claimed=False),indent=2))
    print(json.dumps(rows,indent=2),flush=True)
    if not complete:raise RuntimeError('Incomplete sweep: all missing/failed settings recorded')


if __name__=='__main__':main()
