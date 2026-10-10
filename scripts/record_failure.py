import argparse
import json
import os
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--output',default='runs/shell');args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
(out/'FAILED.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),stage=os.environ.get('FAILURE_STAGE'),utc=datetime.now(timezone.utc).isoformat(),previous_valid_checkpoint_retained=(out/'last.pt').exists(),see_slurm_error_log=True),indent=2))
