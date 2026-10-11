#!/usr/bin/env python3
"""Freeze only tracked package/config/docs and lock the independent runtime."""
import argparse,json,shutil,subprocess
from pathlib import Path
from plane_auto_occ.area_pool import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path(a.experiment).resolve()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
    if status.strip():raise RuntimeError('Commit source/config/docs before freezing')
    if (root/'runtime').exists():raise FileExistsError('Existing immutable snapshot')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
    for name in tracked:
        if not name:continue
        source=repo/name;target=root/'runtime'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    locked=[p for p in (root/'runtime').rglob('*') if p.is_file()]
    locked += [root/'config.json',root/'plan.json',root/'data_lock.json']
    files=[dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in sorted(locked)]
    (root/'runtime_lock.json').write_text(json.dumps(dict(git_revision=revision,files=files),indent=2))
    (root/'logs').mkdir(exist_ok=True)
    print(json.dumps(dict(experiment=str(root),revision=revision,locked_files=len(files))))

if __name__=='__main__':main()
