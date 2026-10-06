"""Fast scheduler tests. Synthetic subprocess outputs; no physical RT/grid run."""
import importlib.util,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scheduler',ROOT/'scripts/run_v48_grid.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    template=json.loads((ROOT/'configs/v48_grid.json').read_text())
    assert len(m.expand(template))==81
    assert len({m.identity(c) for c in m.expand(template)})==81
    bad=json.loads(json.dumps(template));bad['axes']['eta']=[-1]
    try:m.expand(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid efficiency accepted')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for folder in ('src','scripts','data','STARS_library-master/input/m3.0_t0.0'):(root/folder).mkdir(parents=True)
        (root/'STARS_library-master/input/m3.0_t0.0/0.850.dat').write_text('test fixture')
        template['axes']={'eta':[.1,.15,.2]};config=root/'grid.json';config.write_text(json.dumps(template))
        launches=[]
        class FakeProcess:
            def __init__(self,command,**kwargs):
                out=Path(command[command.index('--output')+1]);c=json.loads((out/'config.json').read_text());eta=c['eta'];launches.append(eta)
                self.code=0 if eta==.1 else 1
                if self.code==0:
                    m.atomic(out/'summary.json',{'fit':{'chi2_total':1.,'t0_mjd':57000.}})
                    (out/'response.csv').write_text('synthetic fixture')
                else:kwargs['stdout'].write('RuntimeError: SUBLIMATION: test' if eta==.15 else 'RuntimeError: numerical test')
            def wait(self):return self.code
        def run(*args):
            with patch.object(sys,'argv',['grid','--config',str(config),*args]):m.main()
        with patch.object(m,'ROOT',root),patch.object(m.subprocess,'Popen',FakeProcess),patch.object(m,'report',lambda folder:None):
            run('--dry-run');assert launches==[]
            run();assert launches==[.1,.15,.2]
            run();assert len(launches)==3
            run('--retry-errors');assert launches==[.1,.15,.2,.2]
            folder=next((root/'results').glob('v48_grid_*'))
            rows=m.records(folder);assert {r['status'] for r in rows}=={'completed','unsupported_sublimation','error'}
            # An interrupted model has no committed status: it must resume.
            failed=next(r for r in rows if r['status']=='error')
            (folder/failed['output']/'status.json').unlink();run();assert launches[-1]==.2 and len(launches)==5
            run('--refine-top','1');assert launches[-1]==.1 and len(launches)==6
            run('--refine-top','1');assert len(launches)==6
        with m.lock(root/'lock'):
            try:
                with m.lock(root/'lock'):pass
            except RuntimeError:pass
            else:raise AssertionError('Concurrent writer lock failed')
    print('PASS: 81 unique configs, validation, dry-run, resume, error retry, sublimation status, interrupted model, fine selection, lock. No physical calculations executed.')

if __name__=='__main__':main()
