import importlib.util,json,tempfile,sys,shutil
from pathlib import Path
from unittest.mock import patch
src=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('large',src/'scripts/run_v48_all_stars.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
config=json.loads((src/'configs/v48_all_stars.json').read_text());assert len(m.expand(config))==285525
assert len({m.identity(c) for c in m.expand(config)})==285525
with tempfile.TemporaryDirectory() as td:
 root=Path(td);shutil.copytree(src,root,dirs_exist_ok=True)
 for f in ['scripts/run_v45_direct_absorption.py','scripts/run_v4_model_grid.py','src/v48_angular_rt.py','data/UGC11487_WISE_TDE_model_input_22epochs.csv','STARS_library-master/'+config['base']['stars']]:
  p=root/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
 config.pop('expected_stars_count',None);config['axes']={'eta':[.13,.15,.17]};p=root/'configs/test.json';p.write_text('\ufeff'+json.dumps(config),encoding='utf-8')
 calls=[]
 class Fake:
  def __init__(self,cmd,**kw):
   out=Path(cmd[cmd.index('--output')+1]);c=json.loads((out/'config.json').read_text());calls.append(c);self.returncode=0
   m.atomic(out/'summary.json',{'fit':{'chi2_total':c['eta']*100,'t0_mjd':57000}});(out/'response.csv').write_text('fixture')
  def wait(self):return 0
  def poll(self):return 0
  def terminate(self):pass
 def run(*args):
  with patch.object(sys,'argv',['large','--config',str(p),*args]):m.main()
 with patch('importlib.metadata.version',return_value='test-fixture'),patch.object(m,'ROOT',root),patch.object(m.subprocess,'Popen',Fake),patch.object(m,'report',lambda *a:None):
  run('--dry-run');assert not calls
  run('--workers','2');assert len(calls)==3
  run('--workers','2');assert len(calls)==3
  (root/'scripts/unrelated.py').write_text('unrelated')
  run();assert len(calls)==3
  run('--refine-top','2','--workers','2');assert len(calls)==5
  run('--refine-top','2');assert len(calls)==5
  folder=next((root/'results').glob('v48_all_stars_*'));f=next(folder.glob('models/*/status.json'));f.unlink()
  run();assert len(calls)==6
print('PASS: 285525 unique configs, UTF8 BOM, two workers, resume, unrelated script stable identity, fine selection, incomplete recovery. No physical RT executed.')
