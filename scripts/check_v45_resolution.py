"""Targeted convergence check, run once before interpreting small chi2 differences."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_v45_direct_absorption import task_grid,evaluate
from concurrent.futures import ProcessPoolExecutor

def run(c):
 r,comps=evaluate(c)
 return dict(config=c,chi2=r['chi2_total'],energy=r['energy'])
if __name__=='__main__':
 c=task_grid('control')[1]
 cases=[c,dict(c,n_radii=64),dict(c,n_mu=48,n_phi=128),dict(c,dt=5.)]
 with ProcessPoolExecutor(max_workers=2) as ex:rows=list(ex.map(run,cases))
 out=Path(__file__).resolve().parents[1]/'results'/'v45_validation';out.mkdir(parents=True,exist_ok=True)
 (out/'resolution.json').write_text(json.dumps(rows,indent=2))
 for r in rows:print('Nr',r['config']['n_radii'],'angles',r['config']['n_mu']*r['config']['n_phi'],'dt',r['config']['dt'],'chi2',r['chi2'])
