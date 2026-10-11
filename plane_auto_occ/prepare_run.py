"""Create a self-contained run: frozen package, cache, config, observed-only pool."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from .area_pool import build_pool
from .data import load_case


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True, help='Occupancy-schema NPZ, copied without modification')
    p.add_argument('--output', required=True, help='New local run directory')
    p.add_argument('--observed', default='0,3,6')
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--reference-initial', help='Optional initialization comparison checkpoint; copied locally')
    a = p.parse_args()
    case = Path(a.case).resolve(); out = Path(a.output).resolve()
    observed = [int(v) for v in a.observed.split(',')]
    c = load_case(case, observed=observed)
    cfg = json.loads(Path(a.config).read_text())
    if cfg.get('sampling_version') != 'observed_area_uniform_v2':
        raise ValueError('prepare-run expects the repaired observed-area sampling config')
    if cfg['steps'] < 1:
        raise ValueError('Training steps must be positive')
    if cfg.get('loo_mix', .25) > 0 and cfg['loo_weight'] > 0 and len(observed) < 3:
        raise ValueError('LOO requires at least three observed planes')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Run directory must be new/empty')
    out.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    runtime = out/'runtime'
    shutil.copytree(package, runtime/'plane_auto_occ', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    cache = out/'cache/train'/case.name; cache.parent.mkdir(parents=True)
    shutil.copy2(case, cache)
    assert digest(cache) == digest(case)
    config = out/'config.json'; config.write_text(json.dumps(cfg, indent=2)+'\n')
    for name in ('gates', 'logs'):
        (out/name).mkdir()
    pool = out/'pools/generation_000.npz'
    build_pool(cache, observed, pool, tuple(cfg['pool_capacities']), cfg['seed'], 0,
               cfg['pool_margin_mm'], cfg['boundary_band_mm'], cfg['near_mm'])
    root = package.parent
    commit = None
    if (root/'.git').exists():
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            commit = r.stdout.strip()
    lock = dict(format='plane_auto_occ_standalone_run_v1', commit=commit,
        files={str(f.relative_to(runtime)):digest(f) for f in sorted((runtime/'plane_auto_occ').rglob('*')) if f.is_file()},
        cache_path=str(cache.relative_to(out)), cache_sha256=digest(cache),
        config_path='config.json', config_sha256=digest(config), pool_sha256=digest(pool),
        observed=observed, strict_heldout=sorted(set(range(c['total_planes']))-set(observed)),
        case_id=c['case_id'], phase=c['phase'])
    if a.reference_initial:
        reference=out/'reference_initial.pt'; shutil.copy2(a.reference_initial, reference)
        lock.update(reference_initial_path=reference.name, reference_initial_sha256=digest(reference))
    (out/'source_lock.json').write_text(json.dumps(lock, indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(original_cache_path=str(case),
        cache_sha256=digest(case), observed=observed, hidden_geometry_used_for_pool=False,
        source_commit=commit, package_version='0.1.0'), indent=2)+'\n')
    print(f'Prepared independent run: {out}')


if __name__ == '__main__':
    main()
