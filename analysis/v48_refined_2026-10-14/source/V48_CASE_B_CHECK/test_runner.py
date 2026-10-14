"""Integration test with a tiny fake solver; never invokes the physical RT solver."""
import json, tempfile, shutil, subprocess, sys, platform
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def test():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);package=root/HERE.name;package.mkdir();solver=root/'V48_PAPER_FINAL';solver.mkdir()
  for name in ('run_validation.py','rt_queue.py','cache_utils.py','budget.py','validation_plan.json'):shutil.copyfile(HERE/name,package/name)
  prov={'hashes':{},'numerics':{},'python':platform.python_version(),'numpy':np.__version__,'sha256':'fixture'}
  (package/'seed.json').write_text(json.dumps({'provenance':prov}))
  (solver/'run_final.py').write_text('from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\ndef prepare(): pass\n')
  (solver/'_solver_fine.py').write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output');p.add_argument('--resolution');a=p.parse_args()
time.sleep(.05);d=Path(a.output);c=json.loads(Path(a.config).read_text());root=Path(__file__).resolve().parent.parent
prov=json.loads((root/'V48_CASE_B_CHECK/seed.json').read_text())['provenance']
s={'config':c,'numerics':{},'provenance_sha256':'fixture','fit':{'chi2_total':59.},'static_opacity_gate_passed':True}
(d/'response.csv').write_text('time_obs_days,W1_Jy,W2_Jy\\n0,0,0\\n');(d/'provenance.json').write_text(json.dumps(prov));(d/'summary.json').write_text(json.dumps(s))
''')
  cmd=[sys.executable,str(package/'run_validation.py'),'--workers','4','--minutes','1']
  for i in range(2):
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stdout+r.stderr
   out=root/'results/v48_case_b_validation';report=json.loads((out/'report.json').read_text());b=json.loads((out/'total_budget.json').read_text())
   assert report['completed']==13 and report['validation_complete'];assert not report['confidence_intervals_certified'];assert len(b['reservations'])==13
   if i:assert 'TO RUN 0' in r.stdout
  # Simulate one interrupted attempt, absent result, and stale completion flag.
  plan=json.loads((package/'validation_plan.json').read_text());d=root/'results/v48_paper_final/models/fine'/plan['points'][0]['id'];(d/'summary.json').unlink()
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stdout+r.stderr
  assert len(json.loads((out/'total_budget.json').read_text())['reservations'])==14
  # Exhausted persistent budget cannot launch an uncompleted target on restart.
  (d/'summary.json').unlink();b=json.loads((out/'total_budget.json').read_text())
  b['reservations'] += [{'attempt':i,'model_id':'interrupted'} for i in range(15,19)];(out/'total_budget.json').write_text(json.dumps(b))
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stdout+r.stderr
  report=json.loads((out/'report.json').read_text());assert report['completed']==12 and report['status']=='total_budget_exhausted'
 print('PASS: 13 fixed targets; cached restart; stale checkpoint recovery; persistent cap; no RT used')
if __name__=='__main__':test()
