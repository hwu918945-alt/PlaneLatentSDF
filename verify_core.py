"""Run behavior checks for the current SAX-only myocardium shell method."""
import argparse
import json
from pathlib import Path
import unittest
import torch


def run():
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'),pattern='test_shell.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),torch_version=torch.__version__,
                device='cpu',clinical_training_performed=False)
    if not result.wasSuccessful():raise RuntimeError(json.dumps(report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='verification.json');args=p.parse_args()
    report=run();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
