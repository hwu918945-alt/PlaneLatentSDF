import argparse
import json
from pathlib import Path
import numpy as np
import torch
from core.data import load_case
from core.model import PlaneField
from core.mesh import stack_basis, volume_to_mesh, write_obj


def extract(model, case, out, resolution=160, chunk=32768, cap_stack=False):
    basis = stack_basis(case['axis'].cpu().numpy())
    contour_stack = case['boundary'].cpu().numpy().reshape(-1, 3) @ basis
    lower, upper = contour_stack.min(0), contour_stack.max(0)
    lower[:2] -= 0.15; upper[:2] += 0.15
    low_s, high_s = float(case['s'][0]), float(case['s'][-1])
    lower[2], upper[2] = low_s, high_s
    if cap_stack:
        pad = (high_s-low_s)/(resolution-3)
        lower[2] -= pad; upper[2] += pad
    shape = (resolution,)*3
    volume = np.empty(np.prod(shape), np.float32)
    with torch.no_grad():
        latent = model.encoder(case['q'],case['sdf2d'],case['phase'])
        for start in range(0,len(volume),chunk):
            stop = min(len(volume),start+chunk)
            index = np.stack(np.unravel_index(np.arange(start,stop),shape),-1)
            coords = lower+index/(resolution-1)*(upper-lower)
            clamped = coords.copy(); clamped[:,2] = np.clip(clamped[:,2],low_s,high_s)
            q = torch.as_tensor(clamped@basis.T,dtype=torch.float32,device=latent.device)
            value = model.field(q,case['s'],latent,case['axis'],case['phase']).cpu().numpy()
            if cap_stack:
                value = np.maximum(value,np.maximum(low_s-coords[:,2],coords[:,2]-high_s))
            volume[start:stop] = value
    vertices,faces,report = volume_to_mesh(volume.reshape(shape),lower,upper,basis,case['center_mm'],case['scale_mm'])
    out = Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    write_obj(out,vertices,faces)
    report.update(case_id=case['case_id'],phase=case['phase'],artificial_axial_caps=cap_stack,resolution=resolution,
                  grid_spacing_mm=((upper-lower)/(resolution-1)*case['scale_mm']).tolist(),
                  anatomy='LV_epi_envelope',full_anatomical_extent_verified=False,
                  coordinate_frame='v11_5static aligned oriented millimetres',
                  bbox_mm=[vertices.min(0).tolist(),vertices.max(0).tolist()])
    out.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resolution', type=int, default=160)
    p.add_argument('--chunk', type=int, default=32768)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cap-stack', action='store_true', help='Explicit artificial end caps; not anatomical completion')
    args = p.parse_args()
    if args.resolution < 8:
        raise ValueError('Resolution must be >=8')
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = PlaneField(**saved['config']['model']).to(args.device).eval()
    model.load_state_dict(saved['student'])  # inference uses the supervised ONLINE model
    case = load_case(args.case, args.device)
    report = extract(model,case,args.output,args.resolution,args.chunk,args.cap_stack)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
