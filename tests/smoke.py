"""Run all three examples in fresh directories, then check closed-loop invariants."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workdir', type=Path, required=True, help='new parent directory for test outputs')
args = parser.parse_args()
root = args.workdir.resolve()
root.mkdir(parents=True, exist_ok=False)
reports = []
for example in ('toy', 'adp', 'chignolin', 'chignolin_paper'):
    work = root/example
    subprocess.run([sys.executable, str(REPO/'run.py'), example, '--workdir', str(work),
                    '--smoke', '--timeout', '900'], check=True)
    result = subprocess.run([sys.executable, str(REPO/'tools/verify_run.py'), str(work)],
                            check=True, capture_output=True, text=True)
    reports.append(json.loads(result.stdout))
(root/'results.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps(reports, indent=2))
