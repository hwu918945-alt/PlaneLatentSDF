"""Run mathematical, coordinate-gradient, EMA, optimizer and mesh checks.
These checks do not establish clinical reconstruction quality.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from core.data import polygon_sdf, load_case, gap_queries
from core.model import PlaneField, make_teacher, ema_update, interpolate_latent, leave_one_out_latent
from core.losses import coordinate_gradient, eikonal_loss, gap_geometry_loss, DistanceWeightedDistillation
from core.training import case_objective
from core.mesh import stack_basis, volume_to_mesh


def run():
    torch.set_num_threads(1); torch.manual_seed(7)
    results = {}
    axis = torch.tensor([0., 0., 1.], dtype=torch.float64)
    s = torch.tensor([0., 1., 4.], dtype=torch.float64)
    z = s[:, None]
    assert torch.allclose(leave_one_out_latent(s, z), torch.tensor([[1.]], dtype=torch.float64))
    results['unequal_gap_interpolation_and_LOO'] = 'PASS'
    q = torch.tensor([[0.2, 0.1, 0.4]], dtype=torch.float64, requires_grad=True)
    ss = torch.tensor([0., 1.], dtype=torch.float64)
    zz = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
    f = q[:, 0] + interpolate_latent(q, ss, zz, axis)[:, 0]
    gradient = coordinate_gradient(f, q)
    assert torch.allclose(gradient, torch.tensor([[1., 0., 2.]], dtype=torch.float64))
    h = 1e-6
    numeric = []
    for dim in range(3):
        delta = torch.zeros_like(q); delta[:, dim] = h
        def analytic_field(x): return x[:, 0] + interpolate_latent(x, ss, zz, axis)[:, 0]
        numeric.append(float(((analytic_field(q+delta)-analytic_field(q-delta))/(2*h)).detach()))
    assert np.allclose(gradient.detach().numpy()[0], numeric, atol=1e-8)
    bad = q[:, 0] + interpolate_latent(q, ss, zz, axis).detach()[:, 0]
    assert not torch.allclose(coordinate_gradient(bad, q), gradient)
    results['Eikonal_total_derivative_matches_finite_difference'] = {'status': 'PASS', 'gradient': numeric}
    try:
        interpolate_latent(torch.tensor([[0., 0., 5.]], dtype=torch.float64), s, z, axis)
        raise AssertionError('Out-of-range must fail')
    except ValueError:
        pass
    results['out_of_stack_queries_rejected'] = 'PASS'
    left = torch.tensor([[0., 0., 1-1e-7]], dtype=torch.float64)
    right = torch.tensor([[0., 0., 1+1e-7]], dtype=torch.float64)
    assert abs(float(interpolate_latent(left, s, z, axis) - interpolate_latent(right, s, z, axis))) < 1e-6
    results['field_condition_C0_at_knots'] = 'PASS'
    polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    d = polygon_sdf(np.array([[0., 0.], [2., 0.], [1., 0.], [2., 2.]]), polygon)
    assert np.allclose(d, [-1, 1, 0, np.sqrt(2)])
    results['continuous_polygon_SDF_sign_and_segment_distance'] = 'PASS'
    sample = torch.tensor([[0.01, 0., 0.3], [-0.02, 0., 0.4]], dtype=torch.float64)
    k = 0.5
    tilt = lambda x: (x[:, 0] - k*x[:, 2]) / np.sqrt(1+k*k)
    gt2 = sample[:, 0] - k*sample[:, 2]
    qr = sample.clone().requires_grad_(True)
    g = coordinate_gradient(tilt(qr), qr)
    tangent = g - (g*axis).sum(-1, keepdim=True)*axis
    assert torch.allclose(tilt(sample), tangent.norm(dim=-1)*gt2)
    assert not torch.allclose(tilt(sample), gt2)
    assert float(eikonal_loss(tilt, sample)) < 1e-20
    assert float(gap_geometry_loss(tilt, sample, axis)) < 1e-20
    sphere = lambda x: x.norm(dim=-1)-0.5
    sphere_sample = torch.tensor([[0.45, 0., 0.2], [0.4, 0.2, 0.2]], dtype=torch.float64)
    assert float(gap_geometry_loss(sphere, sphere_sample, axis)) > 0
    elongation = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    curved = lambda x: torch.stack([x[:,0]/elongation, x[:,1], x[:,2]], -1).norm(dim=-1)-0.5
    curvature = gap_geometry_loss(curved, sphere_sample, axis)
    curvature.backward()
    assert elongation.grad is not None and torch.isfinite(elongation.grad) and elongation.grad.abs() > 0
    results['second_order_geometry_has_nonzero_parameter_gradient'] = 'PASS'
    results['2D_3D_distance_compatibility_and_second_order_geometry'] = 'PASS'
    model = PlaneField(latent_dim=16, width=32, prototypes=32)
    teacher = make_teacher(model)
    original = {n: p.clone() for n, p in teacher.named_parameters()}
    with torch.no_grad():
        for p in model.parameters(): p.add_(0.02)
    ema_update(teacher, model, 0.9)
    for n, p in teacher.named_parameters():
        assert not p.requires_grad
        assert torch.allclose(p, 0.9*original[n]+0.1*dict(model.named_parameters())[n], atol=1e-7)
    results['EMA_direction_and_frozen_teacher'] = 'PASS'
    n, count = 5, 48
    phase = 0
    sax_s = torch.linspace(-0.4, 0.4, n)
    angle = torch.arange(count)/count*2*torch.pi
    boundary = torch.stack([0.3*torch.cos(angle), 0.2*torch.sin(angle), torch.zeros_like(angle)], -1)
    boundary = boundary[None].repeat(n, 1, 1); boundary[..., 2] = sax_s[:, None]
    q_data = boundary.clone(); q_data[..., :2] += torch.randn(n, count, 2)*0.03
    d_data = q_data[..., :2].norm(dim=-1)-0.25  # fixture only, not polygon GT
    case = dict(q=q_data, sdf2d=d_data, boundary=boundary, s=sax_s,
                axis=axis.float(), phase=phase, case_id='fixture', scale_mm=100., center_mm=np.zeros(3))
    distill = DistanceWeightedDistillation(prototypes=32)
    cfg = json.loads((Path(__file__).parent/'config.json').read_text())
    cfg.update(encoder_points=32, supervision_points=16, boundary_points=16, regularization_points=12,
               warmup_epochs=0, ramp_epochs=1)
    queries, intervals, a = gap_queries(case, 64)
    assert bool(((queries @ case['axis']) > sax_s[intervals]).all())
    assert bool(((queries @ case['axis']) < sax_s[intervals+1]).all())
    assert bool(((a > 0)&(a < 1)).all())
    results['unlabeled_sampling_stays_between_real_planes'] = 'PASS'
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.encoder.point[0].weight.detach().clone()
    loss, log, logits = case_objective(model, teacher, distill, case, cfg, 0)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    for group in [model.encoder, model.decoder, model.head]:
        grads = [p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    optimizer.step(); distill.update_center(logits); ema_update(teacher, model, 0.996)
    assert not torch.equal(before, model.encoder.point[0].weight)
    # Independent Eikonal-only path must reach the encoder through plane codes.
    model.zero_grad(set_to_none=True)
    latents = model.encoder(case['q'], case['sdf2d'], phase)
    eik = eikonal_loss(lambda x: model.field(x, sax_s, latents, case['axis'], phase), queries[:12])
    eik.backward()
    assert model.encoder.point[0].weight.grad.abs().sum() > 0
    results['all_losses_backward_teacher_stopgrad_encoder_decoder_head_update'] = {'status':'PASS', 'losses':log}
    results['Eikonal_only_reaches_encoder'] = 'PASS'
    basis = stack_basis(np.array([0.2, 0.3, 0.9]))
    assert np.allclose(basis.T @ basis, np.eye(3)) and np.linalg.det(basis)>0
    size = 48
    bounds = np.array([-0.8]*3), np.array([0.8]*3)
    grid = np.stack(np.meshgrid(*[np.linspace(-0.8,0.8,size)]*3, indexing='ij'), -1)
    volume = np.linalg.norm(grid, axis=-1)-0.5
    center = np.array([12., -9., 40.])
    vertices, faces, report = volume_to_mesh(volume.astype(np.float32), *bounds, basis, center, 100.)
    radius = np.linalg.norm(vertices-center, axis=-1)
    assert np.max(np.abs(radius-50)) < 0.1
    assert report['watertight_edge_test'] and report['nonmanifold_edges']==0
    normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]])
    assert np.mean(np.sum(normals*(vertices[faces].mean(1)-center), axis=-1)) > 0
    results['marching_cubes_axis_spacing_winding_and_mm_inverse'] = {'status':'PASS','max_radius_error_mm':float(np.max(np.abs(radius-50))),**report}
    results['torch_version'] = torch.__version__
    results['clinical_training_performed'] = False
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='verification.json')
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
