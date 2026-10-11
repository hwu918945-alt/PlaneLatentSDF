"""Portability checks for moved pools and frozen standalone runs."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from .paths import resolve_sampling_pool
from .deployment import verify_sources
from .make_synthetic import make_case
from .convert_cache import convert


class MigrationTests(unittest.TestCase):
    def test_moved_pool_is_used_before_stored_original_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); old=root/'old/generation_012.npz'; old.parent.mkdir()
            old.write_bytes(b'locked sampling artifact')
            new=root/'new/outputs/fit/sampling_pools'/old.name;new.parent.mkdir(parents=True)
            shutil.copyfile(old,new)
            ck=root/'new/outputs/fit/step_01000/checkpoint.pt';ck.parent.mkdir();ck.touch()
            ref=dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest())
            self.assertEqual(resolve_sampling_pool(ck,ref),new.resolve())
            old.unlink()
            self.assertEqual(resolve_sampling_pool(ck,ref),new.resolve())
            new.write_bytes(b'wrong pool with same name')
            with self.assertRaises(FileNotFoundError): resolve_sampling_pool(ck,ref)

    def test_frozen_run_survives_move_and_detects_config_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'raw.npz';case=root/'occupancy.npz'
            make_case(raw);convert(raw,case)
            cfg=json.loads(Path(__file__).with_name('config.json').read_text())
            cfg.update(points=128,microbatch_points=32,width=32,steps=2,pool_capacities=[512,256,256])
            config=root/'config.json';config.write_text(json.dumps(cfg))
            run=root/'run'
            result=subprocess.run([sys.executable,'-B','-m','plane_auto_occ.prepare_run',
                '--case',str(case),'--config',str(config),'--output',str(run)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            verify_sources(run)
            moved=root/'moved';shutil.move(str(run),str(moved));case.unlink()
            locked_case,runtime,lock=verify_sources(moved)
            self.assertEqual(runtime,moved/'runtime')
            self.assertTrue(locked_case.exists())
            self.assertTrue(all(k.startswith('plane_auto_occ/') for k in lock['files']))
            (moved/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Locked config changed'):verify_sources(moved)


if __name__=='__main__':unittest.main()
