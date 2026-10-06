"""Conditional V4.8 likelihood profiles and independent numerical checks.
Run inside the existing project. No change to the physical solver.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, copy, csv
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent
ROOT=P.parent
BOUNDS={'eta':(.10,.22),'tau_visc':(100.,400.),'tau_C':(.2,.8),'tau_E':(1.2,3.4),'theta':(30.,55.),'inc':(30.,55.)}
STEPS=dict(eta=.01,tau_visc=25.,tau_C=.025,tau_E=.15,theta=2.5,inc=2.5,t0_mjd=20.)
LEVELS={'fine':{},'angular':dict(nmu=12,nrays=512,nobserver=1024),'time':dict(dt=15.),'radial':dict(nr=12),'spectral':dict(nwave=288)}
EXPECTED='cff64b3506b865af30293bc338207d162f9a16f0313c2d6e48796a02bcc4e37e'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def prepare():
 source=ROOT/'V48_LOCAL_PAIR/solver.py'
 if not source.exists():raise RuntimeError('Required: V48_LOCAL_PAIR/solver.py in the project.')
 raw=source.read_bytes()
 # Windows newline conversion is allowed, all other differences require review.
 normalized=raw.replace(b'\r\n',b'\n')
 if hashlib.sha256(raw).hexdigest()!=EXPECTED and hashlib.sha256(normalized).hexdigest()!=EXPECTED:
  raise RuntimeError('Solver differs from the archived run. Do not bypass this check; send solver.py for review.')
 text=normalized.decode('utf-8')
 marker='    config.update(dt=10.)'
 if text.count(marker)!=1:raise RuntimeError('Unknown numerical setup.')
 for level,override in LEVELS.items():
  code=text.replace(marker,'    numerics.update('+repr(override)+')\n'+marker)
  target=P/f'_solver_{level}.py'
  if not target.exists() or target.read_text(encoding='utf-8')!=code:target.write_text(code,encoding='utf-8')
def evaluate(config,level='fine',plots=False):
 key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
 folder=ROOT/'results/v48_paper_final/models'/level/key;folder.mkdir(parents=True,exist_ok=True)
 write(folder/'config.json',config)
 cmd=[sys.executable,'-u',str(P/f'_solver_{level}.py'),'--resolution','fine','--config',str(folder/'config.json'),'--output',str(folder)]
 if plots:cmd+=['--plots']
 # Always invoke solver: its full source/data/STARS provenance validates resume.
 with (folder/'run.log').open('a',encoding='utf-8') as f:
  result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'Solver failed; inspect {folder / "run.log"}. No physical rejection inferred.')
 s=read(folder/'summary.json')
 if not s['static_opacity_gate_passed']:raise RuntimeError('Outside static-opacity domain.')
 return s,folder
def fixed_time_loss(folder,t0):
 import numpy as np
 r=np.genfromtxt(folder/'response.csv',delimiter=',',names=True)
 # Recover constant hosts from the archived physical inputs via the solver loader.
 # Loader is imported from installed code, ensuring the same actual observations.
 sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'V48_LOCAL_PAIR'))
 import solver
 obs=solver.g.load_observations();chi=0.
 for band in ('W1','W2'):
  m=np.interp(obs.mjd.to_numpy()-t0,r['time_obs_days'],r[band+'_Jy'],left=0,right=0)
  res=(obs['F_'+band+'_Jy'].to_numpy()-obs['baseline_'+band+'_mJy'].iloc[0]/1000-m)/obs['sigma_'+band+'_Jy'].to_numpy()
  chi+=float(res@res)
 return chi
def optimize(start,fixed,maxfev):
 import numpy as np
 from scipy.optimize import minimize
 keys=[k for k in BOUNDS if k not in fixed];base=copy.deepcopy(start)
 for k,v in fixed.items():
  if k!='t0_mjd':base[k]=v
 best={'chi2':float('inf')};calls=0
 def objective(x):
  nonlocal calls
  cfg=copy.deepcopy(base)
  for k,u in zip(keys,x):
   lo,hi=BOUNDS[k];cfg[k]=float(lo+u*(hi-lo))
  s,folder=evaluate(cfg);chi=fixed_time_loss(folder,fixed['t0_mjd']) if 't0_mjd' in fixed else s['fit']['chi2_total']
  calls+=1
  if chi<best['chi2']:best.update(chi2=chi,config=cfg,path=str(folder),t0_mjd=fixed.get('t0_mjd',s['fit']['t0_mjd']))
  print('EVAL',calls,'chi2',round(chi,5),'best',round(best['chi2'],5),flush=True)
  return chi
 x0=[(base[k]-BOUNDS[k][0])/(BOUNDS[k][1]-BOUNDS[k][0]) for k in keys]
 result=minimize(objective,x0,method='Powell',bounds=[(0,1)]*len(keys),options=dict(maxfev=maxfev,xtol=0.001,ftol=1e-5))
 best.update(optimizer_success=bool(result.success),message=str(result.message),evaluations=calls,fixed=fixed)
 return best
def branch_job(branch,args):
 out=ROOT/'results/v48_paper_final'/branch;out.mkdir(parents=True,exist_ok=True)
 cfg=read(P/f'{branch}.json')
 if args.mode=='check':
  rows=[]
  for level in LEVELS:
   print(branch,level,flush=True);s,folder=evaluate(cfg,level,plots=(level=='fine'))
   rows.append(dict(level=level,chi2=s['fit']['chi2_total'],t0=s['fit']['t0_mjd'],Tmax=s['max_temperature_K'],budget=s['energy']['global_budget_relative'],path=str(folder)))
   if level=='fine':
    post=ROOT/'scripts/v48_paper_products.py'
    if not post.exists():raise RuntimeError('Missing original large-run scripts/v48_paper_products.py')
    subprocess.run([sys.executable,str(post),str(folder)],cwd=ROOT,check=True)
    from plot_all import all_plots
    if all_plots(folder):raise RuntimeError('Incomplete publication figure set: inspect figure_manifest.json')
  for row in rows:row['delta_chi2']=row['chi2']-rows[0]['chi2']
  write(out/'numerical_checks.json',rows)
  return
 if args.mode=='baseline':
  if (out/'baseline.json').exists():cfg=read(out/'baseline.json')['config']
  b=optimize(cfg,{},args.maxfev);write(out/'baseline.json',b);return
 b=read(out/'baseline.json')
 if not b['optimizer_success']:raise RuntimeError('Baseline budget exhausted. Repeat baseline with a larger --maxfev; cached RT models are reused.')
 params=list(STEPS) if args.parameter=='all' else [args.parameter]
 for param in params:
  center=b['t0_mjd'] if param=='t0_mjd' else b['config'][param]
  bounds=(56500.,57500.) if param=='t0_mjd' else BOUNDS[param]
  records=[dict(value=center,chi2=b['chi2'],delta_chi2=0.,optimizer_success=True)]
  for sign in (-1,1):
   start=b['config']
   for i in range(args.points):
    value=max(bounds[0],min(bounds[1],center+sign*STEPS[param]*2**i))
    result=optimize(start,{param:value},args.maxfev);result.update(value=value,delta_chi2=result['chi2']-b['chi2'])
    records.append(result);write(out/f'profile_{param}.json',dict(parameter=param,baseline=b,points=records,scope='conditional; no automatic confidence interval certification'))
    if result['delta_chi2'] < -.1:raise RuntimeError('Profile found a better minimum. Update baseline before interpreting intervals.')
    if not result['optimizer_success']:raise RuntimeError('Profile optimization did not converge. Increase --maxfev; no interval reported.')
    start=result['config']
    if result['delta_chi2']>=3.84 or value in bounds:break
  # Sampled curves only: sparse crossing interpolation must be refined before reporting.
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size':18,'axes.labelsize':18,'axes.titlesize':18,'xtick.labelsize':18,'ytick.labelsize':18})
  rows=sorted(records,key=lambda x:x['value']);fig,ax=plt.subplots(figsize=(8,6))
  ax.plot([r['value'] for r in rows],[r['delta_chi2'] for r in rows],'o-')
  for y in (1.,3.84):ax.axhline(y,color='grey',ls='--',lw=.8)
  ax.set(xlabel=param,ylabel=r'$\Delta\chi^2$',title=branch+' conditional profile');fig.tight_layout();fig.savefig(out/f'profile_{param}.pdf');plt.close(fig)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--mode',choices=['check','baseline','profile'],default='check')
 ap.add_argument('--branch',choices=['local','grid','both'],default='both')
 ap.add_argument('--workers',type=int,choices=[1,2],default=2)
 ap.add_argument('--maxfev',type=int,default=120)
 ap.add_argument('--parameter',choices=['all']+list(STEPS),default='eta')
 ap.add_argument('--points',type=int,default=4)
 args=ap.parse_args()
 if args.maxfev<10 or args.points<1:ap.error('maxfev >=10 and points >=1 required')
 prepare();branches=['local','grid'] if args.branch=='both' else [args.branch]
 with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(lambda b:branch_job(b,args),branches))
 print('FINISHED:',ROOT/'results/v48_paper_final')
if __name__=='__main__':main()
