"""Fast scheduler/design tests. NO dust RT, NO grid computation."""
import json,sys,tempfile,unittest,collections
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import v48_staged_plan as plan
import run_v48_staged as runner
from v48_staged_common import identity,atomic,records
ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'configs/v48_staged_domain.json').read_text())
class Tests(unittest.TestCase):
 def test_design(self):
  a=plan.stage1(SPEC);self.assertEqual(len(a),9024);self.assertEqual(len({identity(c) for c in a}),9024)
  self.assertEqual(len({c['stars'] for c in a[:141]}),141)
  self.assertEqual(a,plan.stage1(SPEC))
  d=plan.balanced_design(SPEC)
  for k in plan.AXES:
   counts=collections.Counter(plan.key(s[k]) for s in d)
   self.assertEqual(len(counts),len(SPEC['axes'][k]));self.assertLessEqual(max(counts.values())-min(counts.values()),2)
 def rows(self):
  a=plan.stage1(SPEC)
  return [dict(model_id=identity(c),config=c,status='completed',fit={'chi2_total':i+70.}) for i,c in enumerate(a)]
 def test_followup(self):
  rows=self.rows();second,seeds=plan.stage2(SPEC,rows)
  self.assertLessEqual(len(second),5640);self.assertEqual(len({c['stars'] for c in second}),141)
  self.assertEqual(len(seeds),282)
  self.assertTrue(all(runner.in_domain(c,SPEC) for c in second))
  excluded=[plan.key(s) for s in plan.balanced_design(SPEC)]
  extra=plan.balanced_design(SPEC,16,48102,excluded)
  self.assertTrue(all(plan.key(s) not in excluded for s in extra))
  fine=plan.fine_selection(rows);self.assertLessEqual(len(fine),20);self.assertGreaterEqual(len({c['stars'] for c in fine}),13)
 def test_fail_closed_reuse(self):
  prov={'hashes':{'src/physics.py':'abc'},'python':'3.11','packages':{'numpy':'1.26'}}
  old={**prov,'spec':{'seed':481,'resolution':'coarse'}}
  self.assertTrue(runner.compatible(old,prov))
  self.assertFalse(runner.compatible(dict(old,hashes={}),prov))
  self.assertFalse(runner.compatible(dict(old,python='3.12'),prov))
  self.assertFalse(runner.compatible(dict(old,spec={'seed':482}),prov))
 def test_execute_resume_and_unsupported(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'scripts').mkdir();stub=root/'scripts/stub.py'
   stub.write_text('''import sys,json\nfrom pathlib import Path\na=sys.argv\no=Path(a[a.index('--output')+1]);c=json.loads(Path(a[a.index('--config')+1]).read_text())\nif c['eta']==0.12: raise RuntimeError('SUBLIMATION: test fixture')\n(o/'summary.json').write_text(json.dumps({'fit':{'chi2_total':77.0}}))\n(o/'response.csv').write_text('time_obs_days,W1_Jy,W2_Jy\\n0,0,0\\n')\n''')
   a=plan.stage1(SPEC)[:2];a[1]=dict(a[1],eta=.12);args=SimpleNamespace(workers=2,retry_errors=False)
   out=root/'results/test'
   with patch.object(runner,'ROOT',root),patch.object(runner,'RUNNER','scripts/stub.py'),patch.object(runner,'report',lambda *x:None):
    runner.execute(a,out,'coarse',args)
    rs=records(out);self.assertEqual({r['status'] for r in rs},{'completed','unsupported_sublimation'})
    stamps={p:p.stat().st_mtime_ns for p in out.glob('models/*/status.json')}
    runner.execute(a,out,'coarse',args)
    self.assertEqual(stamps,{p:p.stat().st_mtime_ns for p in stamps})
 def test_import_legacy(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'results/v48_all_stars_fixture';dest=root/'results/v48_staged_fixture';dest.mkdir(parents=True)
   c=plan.stage1(SPEC)[0];mid=identity(c);m=src/'models'/mid;m.mkdir(parents=True)
   sk='STARS_library-master/'+c['stars']
   hashes={'src/physics.py':'abc',runner.RUNNER:'def',sk:'ghi','scripts/v48_large_lock.py':'jkl'}
   prov={'hashes':hashes,'python':'3.11','packages':{'numpy':'1.26'}}
   atomic(src/'manifest.json',dict(prov,spec={'seed':481,'resolution':'coarse'}))
   r={'model_id':mid,'config':c,'status':'completed','fit':{'chi2_total':70.},'output':'models/'+mid}
   atomic(m/'status.json',r);atomic(m/'config.json',c)
   atomic(m/'summary.json',{'config':c,'fit':r['fit'],'numerics':{'seed':481,'dt':40.},'provenance_sha256':'proof'})
   atomic(m/'provenance.json',{'sha256':'proof','hashes':hashes});(m/'response.csv').write_text('fixture')
   with patch.object(runner,'ROOT',root):runner.import_legacy(dest,prov,SPEC)
   imported=records(dest);self.assertEqual(len(imported),1);self.assertEqual(imported[0]['model_id'],mid)
   self.assertTrue((m/'status.json').exists());self.assertTrue((dest/'models'/mid/'response.csv').exists())
 def test_freeze(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan.json';self.assertEqual(runner.freeze(p,[{'a':1}]),[{'a':1}]);self.assertEqual(runner.freeze(p,[{'a':2}]),[{'a':1}])
if __name__=='__main__':unittest.main()
