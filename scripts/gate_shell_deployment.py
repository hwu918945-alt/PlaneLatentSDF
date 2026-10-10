"""CUDA integration gate on Train 002 and Val 012; all fitted weights discarded."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this gate')
    root = ROOT / 'preflight/shell_deployment/cuda_gate'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'CUDA_GATE_COMPLETE.json').exists():
        raise ValueError('Gate already completed; use a fresh deployment gate directory')
    run('scripts/check_shell_ready.py')
    data = Path('preflight/shell_single_002')
    run('scripts/prepare_single_patient.py', '--data', 'data/plane_sdf_shell', '--output', data)
    run('scripts/smoke_shell.py', '--data', data, '--device', 'cuda', '--output', root / 'full_losses.json')
    config = json.loads((ROOT / 'config.json').read_text())
    config.update(epochs=2, batch_cases=2)
    cfg_path = root / 'config.json'
    cfg_path.write_text(json.dumps(config, indent=2) + '\n')
    continuous, resumed = root / 'continuous', root / 'resumed'
    common = ('train.py', '--config', cfg_path, '--data', data, '--device', 'cuda')
    run(*common, '--output', continuous)
    run(*common, '--output', resumed, '--max-steps', 1)
    run(*common, '--output', resumed, '--resume', resumed / 'last.pt')
    first = torch.load(continuous / 'last.pt', map_location='cpu', weights_only=False)
    second = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)
    assert first['global_step'] == second['global_step'] == 2
    assert first['epoch'] == second['epoch'] == 1
    bitwise = True
    for module in ['student', 'teacher', 'distill']:
        for key, value in first[module].items():
            other = second[module][key]
            torch.testing.assert_close(value, other, rtol=1e-5, atol=1e-7)
            bitwise = bitwise and torch.equal(value, other)
    optimizer_bitwise = True
    for key, state in first['optimizer']['state'].items():
        for name, value in state.items():
            other = second['optimizer']['state'][key][name]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=1e-5, atol=1e-7)
                optimizer_bitwise = optimizer_bitwise and torch.equal(value, other)
            else:
                assert value == other
    assert torch.equal(first['rng_torch'], second['rng_torch'])
    assert first['rng_python'] == second['rng_python']
    import numpy as np
    for left, right in zip(first['rng_numpy'], second['rng_numpy']):
        np.testing.assert_equal(left, right)
    assert len(first['rng_cuda']) == len(second['rng_cuda'])
    assert all(torch.equal(a, b) for a, b in zip(first['rng_cuda'], second['rng_cuda']))
    run('evaluate.py', '--checkpoint', resumed / 'last.pt', '--data', data,
        '--split', 'train', '--device', 'cuda', '--meshes', '--resolution', 64,
        '--output', root / 'evaluation')
    evaluation = json.loads((root / 'evaluation/numeric_eval_summary.json').read_text())
    assert evaluation['patient_count'] == 1 and evaluation['phase_file_count'] == 2
    report = dict(status='PASS', cuda=torch.version.cuda, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), fitted_patient='002', validation_patient='012',
                  test_patient_used_for_optimization=False, full_loss_cuda_backward=True,
                  epoch_boundary_resume_numerically_equal=True,
                  student_teacher_center_bitwise_equal=bitwise,
                  optimizer_bitwise_equal=optimizer_bitwise, rng_states_equal=True,
                  train_and_evaluation_interfaces_verified=True,
                  cardiac_quality_verified=False, formal_weights_initialized_from_gate=False)
    (root / 'CUDA_GATE_COMPLETE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
